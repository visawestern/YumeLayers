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

func testParamAPI() {
    section("Параметры API: приведение значений JSON → поля/переключатели")

    // paramStringValue: число/строка/bool → строка текстового поля.
    expectEqual(paramStringValue("0.01"), "0.01", "строка передаётся как есть")
    expectEqual(paramStringValue(0.01), "0.010000", "дробное число → 6 знаков")
    expectEqual(paramStringValue(80), "80", "целое число → без точки")
    expectEqual(paramStringValue(2000), "2000", "крупное целое → без точки")
    expectEqual(paramStringValue(true), "1", "bool → '1'")
    expectEqual(paramStringValue(3.5), "3.500000", "дробное 3.5 → 6 знаков")
    expect(paramStringValue([1]) == nil, "несовместимый тип → nil")

    // paramBoolValue: bool/0/1/строка → Bool.
    expectEqual(paramBoolValue(true), true, "bool true")
    expectEqual(paramBoolValue(false), false, "bool false")
    expectEqual(paramBoolValue(1), true, "число 1 → true")
    expectEqual(paramBoolValue(0), false, "число 0 → false")
    expectEqual(paramBoolValue("true"), true, "'true' → true")
    expectEqual(paramBoolValue("off"), false, "'off' → false")
    expectEqual(paramBoolValue("yes"), true, "'yes' → true")
    expect(paramBoolValue("abc") == nil, "непарсимо → nil")
    expect(paramBoolValue(2) == nil, "2 не является bool-значением → nil")
}


func testTensorBufferPool() {
    section("TensorBufferPool: переиспользование приватных буферов")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // 1. Пул: take → retire → reclaim отдаёт ТОТ ЖЕ буфер.
    do {
        let b1 = engine.takeBuffer(4096)
        let b2 = engine.takeBuffer(4096)
        // Оба nil: пул пуст.
        expect(b1 == nil && b2 == nil, "пустой пул не отдаёт буферы")
        // Создаём два буфера вручную и кладём их в пул.
        if let m1 = engine.device.makeBuffer(length: 4096, options: .storageModePrivate),
           let m2 = engine.device.makeBuffer(length: 4096, options: .storageModePrivate) {
            engine.retireBuffer(m1)
            engine.retireBuffer(m2)
            engine.reclaimRetiredBuffers()
            let r1 = engine.takeBuffer(4096)
            let r2 = engine.takeBuffer(4096)
            // Переиспользование: вернулись те же объекты (в любом порядке).
            expect((r1 === m1 || r1 === m2), "takeBuffer вернул буфер из пула (1)")
            expect((r2 === m1 || r2 === m2), "takeBuffer вернул буфер из пула (2)")
            expect(r1 !== r2, "два takeBuffer дают разные буферы")
        }
    }

    // 2. Поведенческий: на длинном прогоне число РЕАЛЬНЫХ аллокаций не растёт
    //    линейно. Моделируем ритм тренировки: каждый шаг = begin/create/end,
    //    и create/end синхронизирует GPU и возвращает освобождённые буферы
    //    в пул переиспользования.
    do {
        let outerSteps = 40      // «шаги» обучения
        let innerPerStep = 10    // тензоров на шаг
        let iterations = outerSteps * innerPerStep

        // Прогрев: пара итераций наполняет пул.
        for _ in 0..<2 {
            session.begin()
            let t = GPUTensor(engine: engine, rows: 64, cols: 64)
            _ = t.n
            _ = session.end()
        }
        let startAlloc = engine.totalBufferAllocations
        for _ in 0..<outerSteps {
            session.begin()
            for _ in 0..<innerPerStep {
                let t = GPUTensor(engine: engine, rows: 64, cols: 64)
                _ = t.n
            }
            _ = session.end()
        }
        let delta = engine.totalBufferAllocations - startAlloc
        // После прогрева аллокации должны быть ~первый шаг (внутри шага буферы
        // живут одновременно и не переиспользуются), а не расти линейно.
        expect(delta < iterations / 8,
               String(format: "пул: %d созданий породили %d аллокаций (< %d)",
                      iterations, delta, iterations / 8))
    }
}


func testTensorStagingPool() {
    section("TensorStagingPool: staging-буферы переиспользуются")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let t = GPUTensor(engine: engine, rows: 8, cols: 8)
    let vals = (0..<64).map { Float($0) * 0.01 }

    // Прогрев: первый setData/toArray могут выделить staging.
    t.setData(vals)
    _ = t.toArray()

    let before = engine.totalStagingAllocations
    let calls = 100
    var lastCount = 0
    for _ in 0..<calls {
        t.setData(vals)
        let a = t.toArray()
        lastCount = a.count
    }
    let delta = engine.totalStagingAllocations - before
    expectEqual(lastCount, 64, "toArray вернул 64 значения")
    expect(delta <= 2,
           String(format: "%d циклов setData+toArray: %d новых staging-буферов (ожидается ≤ 2)",
                  calls, delta))
}


func testGpuWaitPolicy() {
    section("GpuWaitPolicy: показания драйвера и ожидание command buffer")

    // 1) Правдоподобное показание драйвера берём как есть.
    expectEqual(boundedGpuMs(reportedMs: 2, wallMs: 10), 2,
                "правдоподобное показание не меняется")

    // 2) Драйвер сообщил 18 с при 12 мс стенного времени. GPU физически
    //    не мог быть занят дольше самого шага — такое показание отбрасываем,
    //    иначе регулятор превращает его в паузу в 2 с на каждом шаге
    //    (обучение замедляется в 100 раз при полностью простаивающей GPU).
    expectEqual(boundedGpuMs(reportedMs: 18_000, wallMs: 12), 12,
                "невозможное показание ограничивается стенным временем")

    // 3) Драйвер не дал времени вовсе — берём стенное.
    expectEqual(boundedGpuMs(reportedMs: 0, wallMs: 8), 8,
                "без показания драйвера — стенное время")

    // 4) Ни показания, ни стенного времени — ноль (не NaN!).
    expectEqual(boundedGpuMs(reportedMs: .nan, wallMs: 0), 0,
                "NaN и отсутствие стенного времени дают 0")

    // 5) Небольшое превышение стенного времени — норма: буфер мог ждать
    //    в очереди (WindowServer, другой клиент).
    expectEqual(boundedGpuMs(reportedMs: 30, wallMs: 12, toleranceMs: 50), 30,
                "превышение в пределах допуска не трогаем")

    // 6) Главное свойство: пауза регулятора не может взорваться из-за
    //    показания драйвера. 18 000 мс при duty 0.9 дали бы 2000 мс паузы;
    //    после ограничения — около 1.3 мс.
    let pause = dutySleepMs(gpuMs: boundedGpuMs(reportedMs: 18_000, wallMs: 12),
                            duty: 0.9, maxPauseMs: 2000)
    expect(pause < 5,
           String(format: "пауза по ограниченному показанию: %.2f мс (ожидается < 5)", pause))

    // 7) Обработчик сработал — ждать больше нечего.
    expectEqual(gpuWaitOutcome(state: .running, handlerSignaled: true,
                               elapsedSeconds: 0.01, deadlineSeconds: 30), .done,
                "сработавший обработчик — готово")

    // 8) Обработчик потерялся, но буфер уже выполнен — это и есть путь
    //    спасения от «GPU не ответила»: ждать handler'а бессмысленно.
    expectEqual(gpuWaitOutcome(state: .completed, handlerSignaled: false,
                               elapsedSeconds: 0.01, deadlineSeconds: 30), .done,
                "выполненный буфер без обработчика — готово")

    // 9) Буфер ещё выполняется, время есть — продолжаем ждать.
    expectEqual(gpuWaitOutcome(state: .running, handlerSignaled: false,
                               elapsedSeconds: 1, deadlineSeconds: 30), .keepWaiting,
                "идущий буфер — продолжаем ждать")

    // 10) Время вышло — таймаут.
    expectEqual(gpuWaitOutcome(state: .running, handlerSignaled: false,
                               elapsedSeconds: 30, deadlineSeconds: 30), .timedOut,
                "превышен предел — таймаут")

    // 11) Ошибка драйвера — дальше ждать нечего.
    expectEqual(gpuWaitOutcome(state: .failed, handlerSignaled: false,
                               elapsedSeconds: 1, deadlineSeconds: 30), .done,
                "ошибка буфера — ждать бессмысленно")

    // 12) Пауза регулятора режется на короткие срезы: «Стоп» должен
    //     срабатывать сразу, а не через 2 с.
    expectEqual(pauseSlices(totalMs: 250), [100, 100, 50],
                "пауза 250 мс делится на срезы по 100 мс")
    expectEqual(pauseSlices(totalMs: 0), [Double](),
                "нулевая пауза — без срезов")
    expectEqual(pauseSlices(totalMs: 40), [40],
                "пауза меньше среза — один срез")
}


func testReconOutput() {
    section("ReconOutput: сохранение реконструкций и выбор кадров")

    // 1) Кадры с реконструкцией — по именам, в порядке кадров набора.
    expectEqual(reconstructedFrameIndices(frameNames: ["a", "b", "c", "d"],
                                          reconNames: ["c", "a"]),
                [0, 2],
                "индексы кадров с реконструкцией: по порядку кадров")
    expectEqual(reconstructedFrameIndices(frameNames: ["a"], reconNames: []),
                [], "нет реконструкций — пустой список")
    expectEqual(reconstructedFrameIndices(frameNames: [], reconNames: ["x"]),
                [], "пустой набор кадров")

    // 2) Сохранение на диск: файлы пишутся, имена сохраняются.
    let dir = NSTemporaryDirectory() + "yume-recons-\(UUID().uuidString)"
    defer { try? FileManager.default.removeItem(atPath: dir) }
    let recons = ["frame_0001.png": Data([0x89, 0x50, 0x4E, 0x47]),
                  "frame_0002.png": Data([0x89, 0x50, 0x4E, 0x47])]
    let saved = try? saveReconstructions(recons, to: dir)
    expectEqual(saved ?? 0, 2, "сохранено 2 реконструкции")
    let files = (try? FileManager.default.contentsOfDirectory(atPath: dir))?.sorted() ?? []
    expectEqual(files, ["frame_0001.png", "frame_0002.png"], "файлы на диске с теми же именами")

    // 3) Пустой набор — папка создаётся, файлов 0.
    let emptyDir = NSTemporaryDirectory() + "yume-recons-empty-\(UUID().uuidString)"
    defer { try? FileManager.default.removeItem(atPath: emptyDir) }
    let savedEmpty = try? saveReconstructions([:], to: emptyDir)
    expectEqual(savedEmpty ?? -1, 0, "пустой набор — 0 файлов")
    expect(FileManager.default.fileExists(atPath: emptyDir),
           "папка для реконструкций создана")
}


func testInterpolationPlan() {
    section("Interpolation: план кадров, easing, интерполяция латента")

    // 1) Число кадров = длительность × fps. Минимум 2: кадр A и кадр B.
    expectEqual(interpolationFrameCount(seconds: 2, fps: 24), 48,
                "2 с × 24 fps = 48 кадров")
    expectEqual(interpolationFrameCount(seconds: 0.5, fps: 24), 12,
                "0.5 с × 24 fps = 12 кадров")
    expectEqual(interpolationFrameCount(seconds: 1, fps: 1), 2,
                "1 с × 1 fps = 2 кадра")
    expectEqual(interpolationFrameCount(seconds: 0, fps: 24), 2,
                "нулевая длительность — минимум 2 кадра")
    expectEqual(interpolationFrameCount(seconds: -3, fps: 24), 2,
                "отрицательная длительность — минимум 2 кадра")
    expectEqual(interpolationFrameCount(seconds: 2, fps: 0), 2,
                "fps = 0 — минимум 2 кадра")

    // 2) Моменты времени — равномерно от 0 до 1 включительно.
    let times = interpolationTimes(frameCount: 5)
    expectEqual(times.count, 5, "5 моментов времени")
    expectClose(times.first ?? -1, 0, 1e-9, "первый момент — 0 (исходный кадр A)")
    expectClose(times.last ?? -1, 1, 1e-9, "последний момент — 1 (исходный кадр B)")
    expectClose(times[2], 0.5, 1e-9, "середина — 0.5")

    // 3) Easing smoothstep: на концах 0 и 1, в середине 0.5, монотонный.
    expectClose(smoothstep(0), 0, 1e-9, "smoothstep(0) = 0")
    expectClose(smoothstep(1), 1, 1e-9, "smoothstep(1) = 1")
    expectClose(smoothstep(0.5), 0.5, 1e-9, "smoothstep(0.5) = 0.5")
    expectClose(smoothstep(-1), 0, 1e-9, "smoothstep вне [0,1] обрезается снизу")
    expectClose(smoothstep(2), 1, 1e-9, "smoothstep вне [0,1] обрезается сверху")
    expect(smoothstep(0.25) < 0.25,
           "smoothstep(0.25) < 0.25 — начало движения мягче линейного")
    var monotone = true
    var prev = -1.0
    for i in 0...20 {
        let v = smoothstep(Double(i) / 20)
        if v < prev { monotone = false }
        prev = v
    }
    expect(monotone, "smoothstep монотонен")

    // 4) Лерп латента — ровно середина между A и B.
    expectEqual(interpolateScalars([0, 10], [10, 20], t: 0), [0, 10],
                "t = 0 — латент A")
    expectEqual(interpolateScalars([0, 10], [10, 20], t: 1), [10, 20],
                "t = 1 — латент B")
    expectEqual(interpolateScalars([0, 10], [10, 20], t: 0.5), [5, 15],
                "t = 0.5 — ровно посередине")
    expectEqual(interpolateScalars([2], [6], t: 0.25), [3],
                "t = 0.25 — четверть пути")

    // 5) План целиком: длительность, fps, число кадров и моменты.
    let plan = interpolationPlan(seconds: 2, fps: 24)
    expectEqual(plan.frameCount, 48, "план: 48 кадров")
    expectEqual(plan.fps, 24, "план: 24 fps")
    expectClose(plan.seconds, 2, 1e-9, "план: 2 секунды")
    expectEqual(plan.times.count, 48, "план: 48 моментов времени")
    expectClose(plan.times.last ?? -1, 1, 1e-9, "план: последний момент — 1")
}

/// Полный сквозной путь: две картинки → латент → промежуточные кадры → mp4.

func testInterpolateVideo() {
    section("InterpolateVideo: А → Б, длительность → видео")

    guard gHasMetal else { skip("Metal недоступен"); return }

    let tmp = NSTemporaryDirectory() + "yume-interp-\(UUID().uuidString)"
    defer { try? FileManager.default.removeItem(atPath: tmp) }
    try? FileManager.default.createDirectory(atPath: tmp, withIntermediateDirectories: true)

    // 1) Две опорные картинки (левый и правый кадр «сцены»).
    let w = 16, h = 8
    var a = [Float](repeating: 0, count: w * h * 3)
    var b = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            let i = (y * w + x) * 3
            a[i] = Float(x) / Float(w)
            a[i + 1] = Float(y) / Float(h)
            a[i + 2] = 0.25
            b[i] = 1 - Float(x) / Float(w)
            b[i + 1] = 0.75
            b[i + 2] = 0.5
        }
    }
    let pathA = tmp + "/a.png", pathB = tmp + "/b.png"
    guard encodePNG(width: w, height: h, rgb: a) != nil,
          encodePNG(width: w, height: h, rgb: b) != nil else {
        expect(false, "опорные картинки созданы"); return
    }
    try? encodePNG(width: w, height: h, rgb: a)!.write(to: URL(fileURLWithPath: pathA))
    try? encodePNG(width: w, height: h, rgb: b)!.write(to: URL(fileURLWithPath: pathB))

    // 2) Чекпоинт: случайные веса модели (важно не качество, а форма
    //    данных: энкодер/декодер должны отработать целиком).
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine)
    let ckpt = tmp + "/conv.safetensors"
    do {
        try saveState(vae.matrices, to: ckpt)
    } catch {
        expect(false, "чекпоинт сохранён: \(error)"); return
    }

    // 3) Латент A кодируется, латент B кодируется — это разные векторы.
    let muA = (try? encodeLatent(vae: vae, image: a, w: w, h: h)) ?? []
    let muB = (try? encodeLatent(vae: vae, image: b, w: w, h: h)) ?? []
    expectEqual(muA.count, ConvVAEGPU.latentC * (h / 8) * (w / 8),
                "латент A: C × (h/8) × (w/8) значений")
    expectEqual(muB.count, muA.count, "латент B: та же форма")
    var same = muA.count == muB.count && !muA.isEmpty
    if same {
        for i in 0..<muA.count where abs(muA[i] - muB[i]) > 1e-6 {
            same = false; break
        }
    }
    expect(!same, "латенты разных картинок различаются")

    // 4) Декодирование латента даёт кадр нужного размера и диапазона 0…1.
    let decoded = (try? decodeLatent(vae: vae, latent: muA, w: w, h: h)) ?? []
    expectEqual(decoded.count, w * h * 3, "декодированный кадр: w×h×3")
    expect(decoded.allSatisfy { $0 >= 0 && $0 <= 1 },
           "декодированный кадр в диапазоне 0…1")
    let decodedAgain = (try? decodeLatent(vae: vae, latent: muA, w: w, h: h)) ?? []
    expectEqual(decodedAgain, decoded, "декодирование детерминировано")

    // 5) Сквозная сборка видео: 1 секунда × 12 fps = 12 кадров.
    let out = tmp + "/out.mp4"
    let req = InterpolationRequest(imageAPath: pathA,
                                   imageBPath: pathB,
                                   seconds: 1,
                                   fps: 12,
                                   outputPath: out,
                                   modelWidth: w,
                                   modelHeight: h,
                                   checkpointPath: ckpt)
    do {
        let res = try interpolateVideo(req)
        expectEqual(res.frameCount, 12, "видео: 12 кадров")
        expectEqual(res.fps, 12, "видео: 12 fps")
        expectClose(res.seconds, 1, 1e-9, "видео: 1 секунда")
        expectEqual(res.width, w, "видео: ширина как у опорной картинки")
        expectEqual(res.height, h, "видео: высота как у опорной картинки")
        expect(FileManager.default.fileExists(atPath: out), "mp4-файл создан")

        // 6) Файл — настоящее видео: длительность ≈ кадры/fps.
        let asset = AVURLAsset(url: URL(fileURLWithPath: out))
        let duration = try loadAssetDuration(asset)
        expectClose(CMTimeGetSeconds(duration), 1.0, 0.2, "длительность mp4 ≈ 1 с")
        let tracks = try loadVideoTracks(of: asset)
        expectEqual(tracks.count, 1, "в mp4 одна видеодорожка")
    } catch {
        expect(false, "интерполяция выполнена: \(error)")
    }

    // 7) Плохие входные данные — понятная ошибка, а не падение.
    let bad = InterpolationRequest(imageAPath: tmp + "/нет-такого.png",
                                   imageBPath: pathB,
                                   seconds: 1,
                                   fps: 12,
                                   outputPath: tmp + "/bad.mp4",
                                   modelWidth: w,
                                   modelHeight: h,
                                   checkpointPath: ckpt)
    expectThrows("нет файла A — ошибка") { try interpolateVideo(bad) }

    let badCkpt = InterpolationRequest(imageAPath: pathA,
                                       imageBPath: pathB,
                                       seconds: 1,
                                       fps: 12,
                                       outputPath: tmp + "/bad2.mp4",
                                       modelWidth: w,
                                       modelHeight: h,
                                       checkpointPath: tmp + "/нет-чекпоинта.safetensors")
    expectThrows("нет чекпоинта — ошибка") { try interpolateVideo(badCkpt) }
}

/// Ориентация картинки при записи в видео: верхний левый пиксель кадра
/// должен попасть в верхний левый угол буфера (а не вниз).

func testVideoWriterOrientation() {
    section("VideoWriter: ориентация кадра в пиксель-буфере")

    // Кадр 2×2: каждый угол своего цвета.
    //   верхний левый  — красный (1, 0, 0)
    //   верхний правый — зелёный (0, 1, 0)
    //   нижний левый   — синий  (0, 0, 1)
    //   нижний правый  — жёлтый (1, 1, 0)
    let frame: [Float] = [
        1, 0, 0,   0, 1, 0,
        0, 0, 1,   1, 1, 0,
    ]
    guard let buffer = makePixelBuffer(frame: frame, width: 2, height: 2,
                                      outputWidth: 2, outputHeight: 2) else {
        expect(false, "пиксель-буфер создан"); return
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
        expect(false, "адрес буфера получен"); return
    }
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    func pixel(_ row: Int, _ col: Int) -> (b: Int, g: Int, r: Int) {
        let ptr = base.advanced(by: row * stride + col * 4)
            .assumingMemoryBound(to: UInt8.self)
        return (Int(ptr[0]), Int(ptr[1]), Int(ptr[2]))
    }
    // makePixelBuffer копирует строки БЕЗ переворота (битовый контекст
    // bottom-up + загрузчик кадров низ-вверх). Поэтому верх кадра на входе
    // оказывается в НИЗУ буфера, низ — вверху.
    let topLeft = pixel(0, 0)
    let topRight = pixel(0, 1)
    let bottomLeft = pixel(1, 0)
    let bottomRight = pixel(1, 1)
    expect(topLeft.b > 200 && topLeft.r < 60,
           "верхний левый угол буфера — синий (низ кадра)")
    expect(topRight.r > 200 && topRight.g > 200 && topRight.b < 60,
           "верхний правый угол буфера — жёлтый (низ кадра)")
    expect(bottomLeft.r > 200 && bottomLeft.g < 60 && bottomLeft.b < 60,
           "нижний левый угол буфера — красный (верх кадра)")
    expect(bottomRight.g > 200 && bottomRight.r < 60,
           "нижний правый угол буфера — зелёный (верх кадра)")
}

/// Конвенция загрузчика кадров: строка 0 = НИЗ кадра (низ-вверх).
/// Модель обучена именно на этой конвенции; видео компенсирует её при записи
/// (см. makePixelBuffer). Тест фиксирует конвенцию, чтобы её случайно не
/// «исправили» — иначе сломается согласованность модель ↔ отображение.

func testLoadImageOrientation() {
    section("loadImageRGB: конвенция низ-вверх")

    // Самодостаточность: тест не зависит от внешнего /tmp/orient.png.
    // loadImageRGB возвращает top-down (верх первой строкой), поэтому эталон
    // задаём с чёрным верхом и белым низом: строка 0 → чёрный, строка 31 →
    // белый.
    var gradient = [Float](repeating: 0, count: 32 * 32 * 3)
    for y in 0..<32 {
        let v = Float(y) / 31
        for x in 0..<32 {
            let i = (y * 32 + x) * 3
            gradient[i] = v; gradient[i + 1] = v; gradient[i + 2] = v
        }
    }
    try? savePNG(path: "/tmp/orient.png", width: 32, height: 32, rgb: gradient)

    guard let frame = loadImageRGB(path: "/tmp/orient.png", w: 32, h: 32) else {
        expect(false, "картинка прочитана"); return
    }
    func luma(_ y: Int) -> Float {
        var s: Float = 0
        for x in 0..<32 {
            let i = (y * 32 + x) * 3
            s += (frame[i] + frame[i + 1] + frame[i + 2]) / 3
        }
        return s / 32
    }
    let top = luma(0)
    let bottom = luma(31)
    print(String(format: "конвенция loadImageRGB: строка0=%.2f строка31=%.2f", top, bottom))
    expect(top < 0.1 && bottom > 0.9,
           String(format: "строка 0 = низ (чёрный), строка 31 = верх (белый) (получено %.2f/%.2f)", top, bottom))
}

/// Временная диагностика: качество реконструкции обученной модели.

