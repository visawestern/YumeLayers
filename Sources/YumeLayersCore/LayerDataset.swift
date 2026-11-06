// YumeLayers — итерация 6: НАСТОЯЩЕЕ видео → датасет слоёв → обучение под ключ.
//
// До этого layered-VAE учился исключительно на процедурной синтетике
// (`makeLayeredScene`). Это доказывало математику градиентов, но не доказывало
// продукт: на реальном MP4 ни разметки, ни обучения, ни чекпоинта не было.
// Здесь контур замыкается одной функцией `runLayerPipeline`:
//
//   MP4 ──AVAssetImageGenerator──> RGB-кадры [Float]
//       ──LayerAnnotator─────────> [LayeredScene] (8 каналов, точная сборка)
//       ──trainLayeredVAE────────> веса + PSNR/SSIM + safetensors
//       ──savePNG────────────────> превью: кадр / слои / реконструкция
//
// Никакого Python, никаких внешних моделей сегментации, никаких зависимостей
// кроме системных AVFoundation / CoreGraphics / ImageIO.
//
// ОГРАНИЧЕНИЕ ПАМЯТИ СОБЛЮДЕНО СОЗНАТЕЛЬНО. Раньше GUI держал 43 187 кадров
// в RAM и машина уходила в exit 137. Здесь кадры берутся РАВНОМЕРНО по
// длительности (по умолчанию 16), а не «все подряд», и сразу освобождаются
// после аннотации. Увеличение `frameCount` — осознанный выбор пользователя.
import Foundation
import AVFoundation
import CoreMedia
import CoreGraphics

// MARK: - Стабилизация камеры

/// Отчёт о выравнивании: сколько кадров выброшено как склейка/другая сцена
/// и насколько хорошо сошлись оставшиеся. Без этого «тихого» выравнивания
/// нельзя понять, почему маска внезапно стала сплошной.
public struct FrameAlignmentReport {
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let width: Int
    public let height: Int
    public let shifts: [(dx: Int, dy: Int)]
    public let droppedFrames: Int
    /// Средняя невязка после выравнивания (по яркости, 0…1).
    public let meanResidual: Double
}

/// Выровнять стек кадров по ЦЕЛОЧИСЛЕННОМУ сдвигу камеры и обрезать до общей
/// области. Никакой интерполяции: пиксели только переставляются, поэтому
/// инвариант точной сборки `compose(слои) == кадр` сохраняется побайтно.
///
/// Зачем: в реальном аниме камера панорамирует/зумит. Временная медиана по
/// невыровненным кадрам даёт «призрачный» фон, маска различия раздувается до
/// 80% кадра, и модель учится не на слоях, а на мусоре. Сдвиг ищется
/// перебором ±m по минимуму SAD яркости относительно центрального кадра;
/// кадры с аномально большой невязкой отбрасываются как склейка сцен.
public func stabilizeFrames(_ frames: [[Float]], w: Int, h: Int,
                            maxShift: Int = 8)
    -> (frames: [[Float]], w: Int, h: Int, report: FrameAlignmentReport) {
    precondition(w > 0 && h > 0, "stabilizeFrames: неверный размер кадра")
    guard !frames.isEmpty else {
        return (frames, w, h,
                FrameAlignmentReport(sourceWidth: w, sourceHeight: h, width: w,
                                     height: h, shifts: [], droppedFrames: 0,
                                     meanResidual: 0))
    }
    let m = min(max(maxShift, 0), min(w, h) / 4)
    guard m > 0 else {
        return (frames, w, h,
                FrameAlignmentReport(sourceWidth: w, sourceHeight: h, width: w,
                                     height: h,
                                     shifts: frames.map { _ in (0, 0) },
                                     droppedFrames: 0, meanResidual: 0))
    }

    let x0 = m, x1 = w - m, y0 = m, y1 = h - m
    let availW = x1 - x0, availH = y1 - y0
    // Кратно 8: латент VAE /8, иначе формы не сойдутся.
    let cw = max(8, (availW / 8) * 8)
    let ch = max(8, (availH / 8) * 8)
    let offX = (availW - cw) / 2
    let offY = (availH - ch) / 2
    let sampleCount = max(1, ((y1 - y0 + 1) / 2) * ((x1 - x0 + 1) / 2))
    let ref = frames[frames.count / 2]

    var shifts: [(dx: Int, dy: Int)] = []
    var residual: [Double] = []
    shifts.reserveCapacity(frames.count)
    residual.reserveCapacity(frames.count)

    for f in frames {
        var best = Double.infinity
        var bdx = 0, bdy = 0
        for dy in -m...m {
            for dx in -m...m {
                var sad = 0.0
                var y = y0
                while y < y1 {
                    var x = x0
                    while x < x1 {
                        let a = lumaAt(f, (y + dy) * w + (x + dx))
                        let b = lumaAt(ref, y * w + x)
                        sad += Double(abs(a - b))
                        x += 2
                    }
                    y += 2
                }
                if sad < best { best = sad; bdx = dx; bdy = dy }
            }
        }
        shifts.append((bdx, bdy))
        residual.append(best / Double(sampleCount))
    }

    let sorted = residual.sorted()
    let median = sorted[sorted.count / 2]
    // Порог относительный: если ВЕСЬ отрывок динамичный, медиана невязки
    // тоже высокая — не выбрасывать же все кадры. Но одиночная склейка
    // выделяется на её фоне в разы.
    let cut = max(median * 2.5, 0.02)
    var keep: [Int] = []
    for i in 0..<frames.count where residual[i] <= cut { keep.append(i) }
    if keep.isEmpty { keep = [frames.count / 2] }

    var aligned: [[Float]] = []
    aligned.reserveCapacity(keep.count)
    var sumResidual = 0.0
    var keptShifts: [(dx: Int, dy: Int)] = []
    for i in keep {
        let f = frames[i]
        let (dx, dy) = shifts[i]
        var out = [Float](repeating: 0, count: cw * ch * 3)
        for oy in 0..<ch {
            for ox in 0..<cw {
                let sx = x0 + offX + ox + dx
                let sy = y0 + offY + oy + dy
                let si = (sy * w + sx) * 3
                let di = (oy * cw + ox) * 3
                out[di] = f[si]; out[di + 1] = f[si + 1]; out[di + 2] = f[si + 2]
            }
        }
        aligned.append(out)
        sumResidual += residual[i]
        keptShifts.append((dx, dy))
    }
    let report = FrameAlignmentReport(sourceWidth: w, sourceHeight: h,
                                      width: cw, height: ch,
                                      shifts: keptShifts,
                                      droppedFrames: frames.count - keep.count,
                                      meanResidual: sumResidual / Double(keep.count))
    return (aligned, cw, ch, report)
}

// MARK: - Датасет

/// Датасет аннотированных кадров. `compositeError` — максимальное расхождение
/// `compose(слои)` с исходным RGB: это НЕ метрика обучения, а проверка, что
/// разметка вообще корректна. Если она не ~0, учиться на таком таргете нельзя.
public struct LayerDataset {
    public let w: Int
    public let h: Int
    public let frames: [[Float]]
    public let scenes: [LayeredScene]
    /// Средняя доля пикселей-объектов по стеку.
    public let meanCoverage: Double
    public let compositeError: Float
    /// nil — stabilization не применялась (готовый стек кадров в тестах).
    public let alignment: FrameAlignmentReport?

    public init(w: Int, h: Int, frames: [[Float]], scenes: [LayeredScene],
                meanCoverage: Double, compositeError: Float,
                alignment: FrameAlignmentReport? = nil) {
        self.w = w; self.h = h
        self.frames = frames; self.scenes = scenes
        self.meanCoverage = meanCoverage
        self.compositeError = compositeError
        self.alignment = alignment
    }
}

/// Разметить стек RGB-кадров и честно измерить точность обратной сборки.
///
/// - Parameter stabilize: включить выравнивание по сдвигу камеры. Для
///   настоящего видео — обязательно; для синтетического стека в тестах
///   выключено, чтобы не менять размер кадра.
public func makeLayerDataset(frames: [[Float]], w: Int, h: Int,
                             params: LayerAnnotationParams = LayerAnnotationParams(),
                             stabilize: Bool = false,
                             maxShift: Int = 8) -> LayerDataset {
    precondition(!frames.isEmpty, "makeLayerDataset: пустой стек кадров")
    var src = frames
    var dw = w, dh = h
    var alignment: FrameAlignmentReport? = nil
    if stabilize {
        let s = stabilizeFrames(frames, w: w, h: h, maxShift: maxShift)
        src = s.frames; dw = s.w; dh = s.h; alignment = s.report
    }
    for f in src {
        precondition(f.count >= dw * dh * 3,
                     "makeLayerDataset: кадр меньше \(dw)×\(dh)×3")
    }
    let scenes = annotateFrames(src, w: dw, h: dh, params: params)
    var coverage = 0.0
    var worst: Float = 0
    for (i, scene) in scenes.enumerated() {
        coverage += objectCoverage(scene)
        let got = composeLayers(w: dw, h: dh, background: scene.background,
                                objects: scene.objects, mask: scene.mask,
                                light: scene.light)
        worst = max(worst, maxPixelDelta(got, src[i]))
    }
    return LayerDataset(w: dw, h: dh, frames: src, scenes: scenes,
                        meanCoverage: coverage / Double(scenes.count),
                        compositeError: worst, alignment: alignment)
}

// MARK: - Видео → RGB-кадры

/// Моменты времени для КОРОТКОГО непрерывного отрывка, а не для всего ролика.
///
/// Почему так: фон считается одной временной медианой по стеку кадров. Если
/// надергать 16 кадров из 30-минутного фильма, это будут 16 разных сцен и
/// «медианный фон» станет бессмысленной кашей. Поэтому по умолчанию берётся
/// 6-секундный отрывок с отступом 3% от начала (первые кадры часто чёрные —
/// fade-in / артефакт декодера). Для аниме 6 секунд = стабильный задник и
/// несколько положений движущегося персонажа, что разметке и нужно.
///
/// Чистая функция — тестируется без видеофайла и без Metal.
public func sampleVideoTimes(count: Int, duration: Double,
                             windowSeconds: Double = 2.0,
                             startSeconds: Double? = nil) -> [Double] {
    let n = max(count, 1)
    let d = max(duration, 1e-3)
    let win = min(max(windowSeconds, 1.0 / 30.0), d * 0.94)
    let start = min(max(startSeconds ?? d * 0.03, 0), max(d - win, 0))
    if n == 1 { return [start + win * 0.5] }
    return (0..<n).map { start + win * Double($0) / Double(n - 1) }
}

public enum LayerDatasetError: Error, LocalizedError {
    case unreadableVideo(String)
    case badFrame(String)
    case emptySource

    public var errorDescription: String? {
        switch self {
        case .unreadableVideo(let p): return "не удалось прочитать видео: \(p)"
        case .badFrame(let p): return "кадр не декодировался: \(p)"
        case .emptySource: return "не задан ни видеофайл, ни стек кадров"
        }
    }
}

/// Извлечь `count` кадров из MP4/MOV и сразу привести к RGB [0,1] ровно w×h.
///
/// `AVAssetImageGenerator` тянет только запрошенные моменты, а не весь ролик:
/// при 16 кадрах это ~16 декодирований вместо 43 тысяч из прежнего GUI-пути.
public func loadVideoFrames(url: URL, w: Int, h: Int,
                            count: Int = 16,
                            windowSeconds: Double = 2.0,
                            startSeconds: Double? = nil) throws -> [[Float]] {
    let asset = AVURLAsset(url: url)
    guard let time = try? loadAssetDuration(asset) else {
        throw LayerDatasetError.unreadableVideo(url.path)
    }
    let duration = CMTimeGetSeconds(time)
    guard duration.isFinite, duration > 0.01 else {
        throw LayerDatasetError.unreadableVideo(url.path)
    }
    guard let tracks = try? loadVideoTracks(of: asset), !tracks.isEmpty else {
        throw LayerDatasetError.unreadableVideo(url.path)
    }
    // Жёсткий потолок на число кадров. Прежний GUI держал 43 187 кадров в
    // оперативке и ронял машину (exit 137); датасет хранит и RGB, и 8 слоёв,
    // так что 64 кадра — осознанная граница, а не случайное число.
    let n = min(max(count, 1), 64)
    let times = sampleVideoTimes(count: n, duration: duration,
                                 windowSeconds: windowSeconds,
                                 startSeconds: startSeconds)
    // Многопоточный программный декод: кадры тянутся параллельно, а не по одному.
    do {
        let frames = try FrameRGBExtractor.loadFrames(url: url, times: times,
                                                      w: w, h: h)
        guard frames.count == times.count, frames.allSatisfy({ !$0.isEmpty }) else {
            throw LayerDatasetError.badFrame(url.lastPathComponent)
        }
        return frames
    } catch {
        throw LayerDatasetError.badFrame(url.lastPathComponent)
    }
}

// MARK: - Конвейер под ключ

/// Всё, что нужно знать о прогоне, одной структурой. Никаких скрытых
/// «успехов»: `train.acceptedByFloor` может быть false, и это нормальный
/// честный результат короткого прогона на слабой машине.
public struct LayerPipelineReport {
    public let sourcePath: String
    public let width: Int
    public let height: Int
    public let dataset: LayerDataset
    public let train: LayeredTrainReport
    public let checkpointPath: String?
    public let previewPaths: [String]

    public var frameCount: Int { dataset.scenes.count }
}

/// Одна команда от видеофайла до чекпоинта и превью.
///
/// - Parameters:
///   - videoPath: путь к MP4/MOV; если nil, берётся готовый стек `frames`
///     (этот путь используется тестами и позволяет гонять конвейер без
///     декодера).
///   - frames: готовые RGB-кадры HWC, уже приведённые к `w×h`.
///   - w/h: рабочее разрешение, округляется вверх до кратного 8.
///   - latentChannels: ёмкость латента (32 по умолчанию — ответ на плато
///     28.8 дБ при C=16).
///   - checkpointPath: куда писать safetensors; nil — не писать.
///   - previewPrefix: префикс PNG-файлов; nil — `~/YumeLayers/outputs/gen_pipeline`.
public func runLayerPipeline(
    videoPath: String? = nil,
    frames: [[Float]]? = nil,
    w: Int = 96, h: Int = 72,
    frameCount: Int = 16,
    steps: Int = 200,
    latentChannels: Int = 32,
    lr: Float = 0.01,
    batchSize: Int = 1,
    windowSeconds: Double = 2.0,
    maxShift: Int = 8,
    checkpointPath: String? = nil,
    previewPrefix: String? = nil,
    annotation: LayerAnnotationParams = LayerAnnotationParams(),
    powerPercent: Int = defaultGpuPowerPercent,
    overlap: Bool = false
) throws -> LayerPipelineReport {

    let pw = ((max(w, 8) + 7) / 8) * 8
    let ph = ((max(h, 8) + 7) / 8) * 8

    let sourceFrames: [[Float]]
    let sourceName: String
    if let v = videoPath {
        sourceFrames = try loadVideoFrames(url: URL(fileURLWithPath: v),
                                           w: pw, h: ph, count: frameCount,
                                           windowSeconds: windowSeconds)
        sourceName = v
    } else if let f = frames {
        sourceFrames = f
        sourceName = "frames:\(f.count)"
    } else {
        throw LayerDatasetError.emptySource
    }
    guard !sourceFrames.isEmpty else { throw LayerDatasetError.emptySource }

    let dataset = makeLayerDataset(frames: sourceFrames, w: pw, h: ph,
                                   params: annotation,
                                   stabilize: videoPath != nil,
                                   maxShift: maxShift)
    // После стабилизации кадр ОБРЕЗАН до общей области: все превью и
    // реконструкция обязаны жить в этом же размере, иначе PNG перекосит.
    let dw = dataset.w, dh = dataset.h

    let train = trainLayeredVAE(steps: steps, w: dw, h: dh,
                                scenes: dataset.scenes,
                                lr: lr,
                                latentChannels: latentChannels,
                                cosineSchedule: true,
                                batchSize: batchSize,
                                checkpointPath: checkpointPath,
                                powerPercent: powerPercent,
                                overlap: overlap)

    // Превью: исходный кадр, разметка по слоям и то, что выдала модель.
    // Сравнивать их можно глазами — это и есть «не хуже входа» для человека.
    let prefix = previewPrefix ?? (Params().outPrefix + "_pipeline")
    let dir = (prefix as NSString).deletingLastPathComponent
    try? FileManager.default.createDirectory(atPath: dir,
                                              withIntermediateDirectories: true)
    let scene = dataset.scenes[0]
    let plane = dw * dh
    let gray: [Float] = {
        var g = [Float](repeating: 0, count: plane * 3)
        for i in 0..<plane {
            let v = scene.light[i]
            g[i * 3] = v; g[i * 3 + 1] = v; g[i * 3 + 2] = v
        }
        return g
    }()
    let maskRGB: [Float] = {
        var g = [Float](repeating: 0, count: plane * 3)
        for i in 0..<plane {
            let v = scene.mask[i]
            g[i * 3] = v; g[i * 3 + 1] = v; g[i * 3 + 2] = v
        }
        return g
    }()
    let images: [(String, [Float])] = [
        ("original", dataset.frames[0]),
        ("composite", scene.composite),
        ("background", scene.background),
        ("objects", scene.objects),
        ("mask", maskRGB),
        ("light", gray),
        ("reconstruction", train.reconstructionHWC)
    ]
    var paths: [String] = []
    for (name, rgb) in images {
        let path = prefix + "_" + name + ".png"
        do {
            try savePNG(path: path, width: dw, height: dh, rgb: rgb)
            paths.append(path)
        } catch {
            yumeLog("превью \(name) не сохранено: \(error.localizedDescription)")
        }
    }

    return LayerPipelineReport(sourcePath: sourceName,
                               width: dw, height: dh,
                               dataset: dataset, train: train,
                               checkpointPath: train.checkpointPath,
                               previewPaths: paths)
}
