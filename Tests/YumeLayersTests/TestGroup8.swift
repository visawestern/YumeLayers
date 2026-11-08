// YumeLayers — группа тестов (выделено из main.swift).
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import Metal
import AVFoundation
import UniformTypeIdentifiers
import Accelerate
import VideoToolbox
import IOKit

func testFrameExtractionPlan() {
    section("Извлечение кадров: план чанков, fps и JPEG 85%")

    let full = ParallelExtractor.makePlan(duration: 10, sourceFPS: 24,
                                          maxFrames: 0, cores: 4)
    expectClose(full.sourceFPS, 24, 1e-9, "fps берётся из исходного видео")
    expectEqual(full.allFrames, 240, "все кадры = длительность × fps")
    expectEqual(full.targetFrames, 240, "без лимита извлекаются все кадры")
    expectClose(full.outputFPS, 24, 1e-9, "выходной fps равен исходному")

    // Чанки идут подряд без дыр и перекрытий и покрывают ролик целиком.
    var nextFrame = 0
    for chunk in full.chunks {
        expectEqual(chunk.startFrame, nextFrame,
                    "чанк начинается там, где закончился предыдущий")
        expect(chunk.frameCount > 0, "чанк не пустой")
        nextFrame += chunk.frameCount
    }
    expectEqual(nextFrame, 240, "чанки покрывают весь ролик")

    // Первый чанк маленький: первые кадры видны в интерфейсе почти сразу,
    // а не после декодирования всего ролика.
    expectEqual(full.chunks.first?.frameCount, 60,
                "первый чанк — 60 кадров (быстрый старт)")
    // Дальше чанки растут до базового размера: на длинном ролике нельзя
    // запускать ffmpeg на каждые 60 кадров.
    expect((full.chunks.dropFirst().first?.frameCount ?? 0) > 60,
           "второй чанк крупнее первого")

    let limited = ParallelExtractor.makePlan(duration: 10, sourceFPS: 24,
                                             maxFrames: 100, cores: 4)
    expectEqual(limited.allFrames, 240, "лимит не меняет общее число кадров ролика")
    expectEqual(limited.targetFrames, 100, "лимит ограничивает число кадров")
    // Лимит — это ПЕРВЫЕ N кадров подряд, а не равномерная выборка по всему
    // ролику: частота остаётся исходной, иначе тест на 2000 кадров был бы
    // прореженным по всей длине, а не эмуляцией полной работы на начале.
    expectClose(limited.outputFPS, 24, 1e-9, "при лимите fps остаётся исходным")
    expectEqual(limited.chunks.map { $0.frameCount }.reduce(0, +), 100,
                "лимит делится по чанкам")
    if let last = limited.chunks.last {
        expectClose(limited.timeRange(of: last).end, 100.0 / 24.0, 1e-9,
                    "лимит занимает первые 100 кадров, а не весь ролик")
    } else {
        expect(false, "у лимита есть чанки")
    }

    // Времена чанков берутся из номеров кадров: стык двух чанков — ровно
    // кадр, а не «примерно секунда».
    if full.chunks.count >= 2 {
        let first = full.chunks[0], second = full.chunks[1]
        let r1 = full.timeRange(of: first)
        let r2 = full.timeRange(of: second)
        expectClose(r1.start, 0, 1e-9, "первый чанк начинается с нуля")
        expectClose(r1.end, 60.0 / 24.0, 1e-9, "конец = кадр / fps")
        expectClose(r2.start, r1.end, 1e-9, "следующий чанк начинается в стык")
    }

    // Длинный ролик (≈44 800 кадров, 10 ядер): чанков достаточно много, чтобы
    // рабочие не простаивали, но и не по одному на каждые 60 кадров.
    let long = ParallelExtractor.makePlan(duration: 1866.6, sourceFPS: 24.001,
                                          maxFrames: 0, cores: 10)
    expectEqual(long.targetFrames, long.allFrames, "длинный ролик: все кадры")
    expect(long.chunks.count >= 10,
           "чанков хватает на все ядра (\(long.chunks.count))")
    expect(long.chunks.count <= 120,
           "чанков не слишком много — запуск ffmpeg не съедает время (\(long.chunks.count))")
    expect(long.chunks.first?.frameCount == 60, "быстрый первый кадр на длинном ролике")
    expect(long.chunks.map { $0.frameCount }.reduce(0, +) == long.targetFrames,
           "длинный ролик покрыт без пропусков")
    let maxChunk = long.chunks.map { $0.frameCount }.max() ?? 0
    expect(maxChunk <= 1920, "размер чанка ограничен (память): \(maxChunk)")

    // Маленький ролик не дробится на бессмысленные куски.
    let tiny = ParallelExtractor.makePlan(duration: 1, sourceFPS: 10,
                                          maxFrames: 0, cores: 10)
    expectEqual(tiny.chunks.count, 1, "10 кадров — один чанк")

    expectEqual(ParallelExtractor.jpegQuality, 5,
                "JPEG ≈85% (ffmpeg q:v 5)")
}

/// Собирает эталон: один процесс ffmpeg, непрерывное декодирование.
private func referenceFrames(video: URL, fps: Double, count: Int,
                             dir: URL) -> [Data] {
    let ff = ParallelExtractor.ffmpegPath!
    let p = Process()
    p.executableURL = URL(fileURLWithPath: ff)
    p.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                   "-i", video.path,
                   "-vf", String(format: "fps=%.4f", fps),
                   "-frames:v", "\(count)",
                   "-f", "image2", "-vcodec", "mjpeg", "-q:v", "5",
                   "-vsync", "0", dir.appendingPathComponent("%05d.jpg").path]
    p.standardOutput = FileHandle.nullDevice
    p.standardError = FileHandle.nullDevice
    try? p.run()
    p.waitUntilExit()
    let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
    return files.sorted().compactMap { try? Data(contentsOf: dir.appendingPathComponent($0)) }
}

/// Стыки чанков не должны сдвигать кадры: байты каждого кадра обязаны
/// совпасть с непрерывным декодированием того же ролика.

func testChunkBoundariesAreExact() {
    section("Извлечение: чанки не сдвигают и не теряют кадры")

    guard let ffmpeg = ParallelExtractor.ffmpegPath else {
        skip("ffmpeg не установлен"); return
    }
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_boundary_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let video = root.appendingPathComponent("clip.mp4")

    let gen = Process()
    gen.executableURL = URL(fileURLWithPath: ffmpeg)
    gen.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc=size=96x64:rate=30:duration=20",
                     "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     video.path]
    gen.standardOutput = FileHandle.nullDevice
    gen.standardError = FileHandle.nullDevice
    do { try gen.run() } catch { skip("ffmpeg не запустился"); return }
    gen.waitUntilExit()
    guard gen.terminationStatus == 0 else {
        skip("ffmpeg не собрал тестовое видео"); return
    }

    // 20 с × 30 fps = 600 кадров, план даёт 4+ чанка (несколько стыков).
    func extract(maxFrames: Int) -> (frames: [Data], plan: ParallelExtractor.Plan) {
        let plan = ParallelExtractor.makePlan(duration: 20, sourceFPS: 30,
                                              maxFrames: maxFrames, cores: 10)
        var out: [Data] = []
        let sem = DispatchSemaphore(value: 0)
        let ex = FrameExtractor(url: video,
                                options: FrameExtractor.Options(maxFrames: maxFrames))
        ex.run(onFrame: { _, _, _, jpeg in out.append(jpeg) },
               progress: { _ in },
               completion: { _ in sem.signal() })
        _ = sem.wait(timeout: .now() + 180)
        return (out, plan)
    }

    let full = extract(maxFrames: 0)
    expect(full.plan.chunks.count >= 4,
           "тестовый ролик режется на несколько чанков (\(full.plan.chunks.count))")
    expectEqual(full.frames.count, 600, "получены все кадры ролика")

    let refDir = root.appendingPathComponent("ref_full")
    try? FileManager.default.createDirectory(at: refDir, withIntermediateDirectories: true)
    let ref = referenceFrames(video: video, fps: 30, count: 600, dir: refDir)
    expectEqual(ref.count, 600, "эталон собран")
    expectEqual(full.frames, ref,
                "кадры на стыках чанков байт-в-байт совпадают с эталоном")

    // Лимит кадров — это первые N кадров при исходном fps: они обязаны
    // байт-в-байт совпасть с началом полного извлечения. Раньше лимит
    // пересчитывал fps и прореживал ролик по всей длине — для теста на
    // 2000 кадров это была бы другая выборка, а не эмуляция полной работы.
    let limited = extract(maxFrames: 300)
    expectEqual(limited.frames.count, 300, "лимит: ровно 300 кадров")
    expectClose(limited.plan.outputFPS, 30, 1e-9, "лимит сохраняет исходный fps")
    expectEqual(limited.frames, Array(full.frames.prefix(300)),
                "лимит берёт первые 300 кадров подряд, без прореживания")
}


func testSoftwareFrameExtractionInMemory() {
    section("Извлечение кадров: ffmpeg параллельно, JPEG в оперативке")

    guard let ffmpeg = ParallelExtractor.ffmpegPath else {
        skip("ffmpeg не установлен"); return
    }
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_frame_test_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let video = dir.appendingPathComponent("clip.mp4")

    let gen = Process()
    gen.executableURL = URL(fileURLWithPath: ffmpeg)
    gen.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc=size=64x48:rate=20:duration=2",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", video.path]
    gen.standardOutput = FileHandle.nullDevice
    gen.standardError = FileHandle.nullDevice
    do { try gen.run() } catch { skip("ffmpeg не запустился"); return }
    gen.waitUntilExit()
    guard gen.terminationStatus == 0 else {
        skip("ffmpeg не собрал тестовое видео"); return
    }

    let extractor = FrameExtractor(url: video)
    expectEqual(extractor.estimatedCount(), 40, "оценка = 2 с × 20 fps")

    let sem = DispatchSemaphore(value: 0)
    var frames: [Data] = []
    var ordinals: [Int] = []
    var outcome: Result<Void, Error>?
    extractor.run(onFrame: { _, _, ordinal, jpeg in
        frames.append(jpeg)
        ordinals.append(ordinal)
    }, progress: { _ in }, completion: { r in
        outcome = r
        sem.signal()
    })
    _ = sem.wait(timeout: .now() + 60)

    switch outcome {
    case .success?:
        expect(frames.count >= 38 && frames.count <= 42,
               "кадров ≈ 40, получено \(frames.count)")
        expect(frames.allSatisfy { $0.count > 100 && $0.first == 0xFF && $0.dropFirst().first == 0xD8 },
               "каждый кадр — JPEG (FFD8) в памяти")
        expect(ordinals == Array(0..<frames.count), "порядок кадров строгий")
        // Тот же путь, которым обучается VAE: JPEG из памяти декодируется
        // сразу в размер обучения (80×56), исходный кадр остаётся полным.
        if let first = frames.first, let rgb = loadImageRGB(data: first, w: 80, h: 56) {
            expectEqual(rgb.count, 80 * 56 * 3,
                        "кадр декодируется в размер обучения 80×56")
            expect(rgb.allSatisfy { $0 >= 0 && $0 <= 1 },
                   "значения после декодирования в 80×56 нормализованы")
        } else {
            expect(false, "кадр не декодировался в 80×56")
        }
    case .failure(let e):
        expect(false, "извлечение упало: \(e)")
    case nil:
        expect(false, "извлечение не завершилось за 60 с")
    }
}


func testParallelRGBExtraction() {
    section("Извлечение RGB: программный декод параллельно")

    guard let ffmpeg = ParallelExtractor.ffmpegPath else {
        skip("ffmpeg не установлен"); return
    }
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_rgb_test_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let video = dir.appendingPathComponent("clip.mp4")

    let gen = Process()
    gen.executableURL = URL(fileURLWithPath: ffmpeg)
    gen.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", video.path]
    gen.standardOutput = FileHandle.nullDevice
    gen.standardError = FileHandle.nullDevice
    do { try gen.run() } catch { skip("ffmpeg не запустился"); return }
    gen.waitUntilExit()
    guard gen.terminationStatus == 0 else {
        skip("ffmpeg не собрал тестовое видео"); return
    }

    let w = 32, h = 24
    let times = [0.1, 0.5, 1.0, 1.5]
    let frames = try! FrameRGBExtractor.loadFrames(url: video, times: times,
                                                   w: w, h: h)
    expectEqual(frames.count, times.count, "кадров столько же, сколько моментов")
    expect(frames.allSatisfy { $0.count == w * h * 3 },
           "каждый кадр — HWC RGB нужного размера")
    expect(frames.allSatisfy { $0.allSatisfy { $0 >= 0 && $0 <= 1 } },
           "значения нормализованы в 0...1")
    expect(frames[0] != frames[3], "кадры из разных моментов различаются")

    // Тот же путь, что используют датасет и шоты (без синхронного моста).
    let datasetFrames = try! loadVideoFrames(url: video, w: 16, h: 16, count: 4)
    expectEqual(datasetFrames.count, 4, "датасет получает все кадры")
    expect(datasetFrames.allSatisfy { $0.count == 16 * 16 * 3 },
           "датасет получает кадры нужного размера")

    let shotFrames = try! loadShotFrames(url: video, w: 16, h: 16,
                                         start: 0.2, end: 1.8, count: 3)
    expectEqual(shotFrames.count, 3, "шот получает все кадры")
}


func testExtractionStreamsFramesAndProgress() {
    section("Извлечение: прогресс и кадры идут во время декодирования")

    guard let ffmpeg = ParallelExtractor.ffmpegPath else {
        skip("ffmpeg не установлен"); return
    }
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_stream_test_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let video = dir.appendingPathComponent("clip.mp4")

    let gen = Process()
    gen.executableURL = URL(fileURLWithPath: ffmpeg)
    gen.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc=size=480x270:rate=30:duration=10",
                     "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     video.path]
    gen.standardOutput = FileHandle.nullDevice
    gen.standardError = FileHandle.nullDevice
    do { try gen.run() } catch { skip("ffmpeg не запустился"); return }
    gen.waitUntilExit()
    guard gen.terminationStatus == 0 else {
        skip("ffmpeg не собрал тестовое видео"); return
    }

    let start = Date()
    var firstFrameAt: TimeInterval?
    var completionAt: TimeInterval?
    var sawIntermediate = false
    var frames = 0
    let sem = DispatchSemaphore(value: 0)
    let extractor = FrameExtractor(url: video,
                                   options: FrameExtractor.Options(maxFrames: 120))
    extractor.run(onFrame: { _, _, _, _ in
       frames += 1
        if firstFrameAt == nil { firstFrameAt = Date().timeIntervalSince(start) }
    }, progress: { p in
        if p > 0.02 && p < 0.98 { sawIntermediate = true }
    }, completion: { _ in
        completionAt = Date().timeIntervalSince(start)
        sem.signal()
    })
    _ = sem.wait(timeout: .now() + 120)

    expectEqual(frames, 120, "все 120 кадров получены")
    expect(sawIntermediate, "прогресс меняется до завершения (0 < p < 1)")
    if let first = firstFrameAt, let done = completionAt {
        expect(first < done, "первый кадр приходит раньше завершения")
    } else {
        expect(false, "извлечение не завершилось")
    }
}

// MARK: - Асинхронная выдача кадров в интерфейс


func testExtractBatchPolicy() {
    section("Извлечение: пачки кадров и первая страница коллажа")
    let policy = ExtractBatchPolicy()

    // Первая пачка — ровно страница коллажа (10 кадров), чтобы первые
    // кадры появились сразу, а не после всех 44 000.
    expect(policy.shouldFlush(pending: 9, sinceLastFlushMs: 0,
                              flushedFirstBatch: false) == false,
           "9 кадров меньше первой страницы — не сбрасываем")
    expect(policy.shouldFlush(pending: 10, sinceLastFlushMs: 0,
                              flushedFirstBatch: false),
           "10 кадров — первая пачка готова")

    // Дальше крупные пачки: main thread не дёргается на каждый кадр.
    expect(policy.shouldFlush(pending: 63, sinceLastFlushMs: 0,
                              flushedFirstBatch: true) == false,
           "63 кадра — ещё не полная пачка")
    expect(policy.shouldFlush(pending: 64, sinceLastFlushMs: 0,
                              flushedFirstBatch: true),
           "64 кадра — пачка готова")

    // Даже неполная пачка уходит по таймеру, иначе последние кадры
    // висят в буфере до конца извлечения.
    expect(policy.shouldFlush(pending: 1, sinceLastFlushMs: 250,
                              flushedFirstBatch: true),
           "неполная пачка уходит по времени")
    expect(policy.shouldFlush(pending: 1, sinceLastFlushMs: 50,
                              flushedFirstBatch: true) == false,
           "неполная пачка ждёт таймер")
    expect(policy.shouldFlush(pending: 0, sinceLastFlushMs: 999,
                              flushedFirstBatch: true) == false,
           "пустую пачку не сбрасываем")

    // Коллаж перерисовывается только на первой странице и только пока
    // она заполняется; остальные кадры — по пагинации.
    expect(policy.shouldRenderCollage(page: 0, totalBefore: 0, totalAfter: 10,
                                      pageSize: 10),
           "первая страница заполнилась — рендер")
    expect(policy.shouldRenderCollage(page: 0, totalBefore: 5, totalAfter: 10,
                                      pageSize: 10),
           "кадры первой страницы появляются по мере поступления")
    expect(policy.shouldRenderCollage(page: 0, totalBefore: 10, totalAfter: 74,
                                      pageSize: 10) == false,
           "после 10 кадров коллаж не дёргаем")
    expect(policy.shouldRenderCollage(page: 2, totalBefore: 10, totalAfter: 74,
                                      pageSize: 10) == false,
           "пока пользователь листает, коллаж не перерисовывается")
    expect(policy.shouldRenderCollage(page: 0, totalBefore: 10, totalAfter: 10,
                                      pageSize: 10) == false,
           "без новых кадров рендера нет")
}

// MARK: - Проверка (тест) текущих кадров в памяти


func testFrameValidation() {
    section("Проверка кадров: лимит и целостность")

    let ok = Data([0xFF, 0xD8, 0x00, 0x01, 0xFF, 0xD9])
    let frames = (0..<5).map { i -> MemoryFrame in
        MemoryFrame(name: String(format: "frame_%04d.jpg", i), jpeg: ok)
    }

    // Все кадры валидны, лимит 0 = проверить всё.
    let all = frameValidation(frames: frames, limit: 0, decode: { _ in true })
    expectEqual(all.checked, 5, "проверены все кадры")
    expectEqual(all.passed, 5, "все прошли")
    expectEqual(all.failed, 0, "провалов нет")
    expectEqual(all.issues.count, 0, "проблем нет")

    // Лимит ограничивает число проверяемых кадров.
    let limited = frameValidation(frames: frames, limit: 2, decode: { _ in true })
    expectEqual(limited.checked, 2, "лимит 2: проверены 2 кадра")
    expectEqual(limited.passed, 2, "оба прошли")
    expectEqual(limited.failed, 0, "провалов нет")

    // Лимит больше числа кадров — проверяются все.
    let over = frameValidation(frames: frames, limit: 999, decode: { _ in true })
    expectEqual(over.checked, 5, "лимит больше массива — все кадры")

    // Пустой массив — ничего не проверяем.
    let empty = frameValidation(frames: [], limit: 0, decode: { _ in true })
    expectEqual(empty.checked, 0, "пустой массив: 0 проверок")
    expectEqual(empty.failed, 0, "пустой массив: 0 провалов")

    // Битая сигнатура JPEG ловится.
    var broken = frames
    broken[2] = MemoryFrame(name: "frame_0002.jpg", jpeg: Data([0x00, 0x01, 0x02]))
    let b = frameValidation(frames: broken, limit: 0, decode: { _ in true })
    expectEqual(b.failed, 1, "битая сигнатура — 1 провал")
    expect(b.issues.contains { $0.contains("JPEG") },
           "причина указана: JPEG-сигнатура")

    // Сигнатура есть, но декодирование не проходит.
    let undecodable = frameValidation(frames: frames, limit: 0,
                                      decode: { _ in false })
    expectEqual(undecodable.failed, 5, "все не декодируются")
    expect(undecodable.issues.first?.contains("декод") == true,
           "причина указана: декодирование")
}

// MARK: - Математика и защита от краевых входов (ревью 2026-11)


func testMathAndGuards() {
    section("Математика: smoothstep, метрики кадра, нормализация")

    // smoothstep: обычный порядок краёв (монотонный рост).
    expect(smoothstep(0, 1, 0) == 0, "smoothstep: левый край = 0")
    expect(smoothstep(0, 1, 1) == 1, "smoothstep: правый край = 1")
    expect(smoothstep(0, 1, 0.5) == 0.5, "smoothstep: центр = 0.5")
    expect(smoothstep(0, 1, -1) == 0, "smoothstep: кламп слева")
    expect(smoothstep(0, 1, 2) == 1, "smoothstep: кламп справа")

    // Обращённые края (e0 > e1) — так рендерер зовёт солнце:
    // smoothstep(sunR, sunR*0.6, d). Раньше знаменатель клампился как
    // max(e1-e0, 1e-6): обращённый диапазон инвертировался — диск солнца
    // не закрашивался, а всё небо вокруг заливалось солнечным цветом.
    expect(smoothstep(1, 0.5, 1) == 0, "smoothstep (обратный): t=e0 → 0")
    expect(smoothstep(1, 0.5, 0.5) == 1, "smoothstep (обратный): t=e1 → 1")
    expect(smoothstep(1, 0.5, 2) == 0, "smoothstep (обратный): кламп справа")
    expect(smoothstep(1, 0.5, 0) == 1, "smoothstep (обратный): кламп слева")
    // Вырожденные края — без деления на ноль.
    expect(smoothstep(0.5, 0.5, 0.4) == 0, "smoothstep: e0==e1, t меньше")
    expect(smoothstep(0.5, 0.5, 0.6) == 1, "smoothstep: e0==e1, t больше")

    // Метрики кадра (FrameMetrics).
    expect(lumaAt([1, 1, 1], 0) == 1, "lumaAt: белый пиксель = 1")
    expect(abs(lumaAt([1, 0, 0], 0) - 0.299) < 1e-6, "lumaAt: красный = 0.299")
    expect(maxPixelDelta([0, 1, 0.5], [1, 1, 0.5]) == 1, "maxPixelDelta: максимум разности")
    expect(maxPixelDelta([0], [0, 1, 2]) == 0, "maxPixelDelta: короткий кадр безопасен")

    // rgbToFloats: короткий буфер не читает за границей (остаток — нули).
    let short = FrameRGBExtractor.rgbToFloats(Data([255, 128, 0]), pixels: 2)
    expect(short[0] == 1 && abs(short[1] - 128.0 / 255) < 1e-6 && short[2] == 0,
           "rgbToFloats: доступные байты нормализованы")
    expect(short[3] == 0 && short[4] == 0 && short[5] == 0,
           "rgbToFloats: недостающие пиксели — нули (без крэша)")

    // padToMultipleOf8: продление края (не растяжение) до кратного 8.
    // Вход — RGB: на пиксель три значения, иначе функция читает за границей.
    let (padded, pw, ph) = padToMultipleOf8([Float](repeating: 0.5, count: 3 * 3 * 3), w: 3, h: 3)
    expectEqual(pw, 8, "pad: ширина до кратного 8")
    expectEqual(ph, 8, "pad: высота до кратного 8")
    expectEqual(padded.count, 8 * 8 * 3, "pad: размер кадра")
    expect(padded.allSatisfy { $0 == 0.5 }, "pad: край продлён значением края")
}

// MARK: - Планировщик чанков и бюджеты памяти


