// YumeLayers — интерполяция между двумя опорными картинками.
//
// Продуктовая задача (docs/PLAN-CHECKLIST.md, Этап B):
//     «Интерполяция между двумя кадрами: А → Б + длительность → N промежуточных кадров»
//
// Как это работает с тем, что у нас реально обучено:
//
//   1. Опорные картинки A и B уменьшаются до рабочего размера модели
//      (80×56 по умолчанию) и кодируются обученным VAE в латенты.
//   2. Латенты смешиваются по кривой smoothstep: z(t) = lerp(muA, muB, t).
//      Именно в латентном пространстве — это стандартный способ интерполяции
//      для VAE: пиксели двух картинок складывать бессмысленно (получается
//      каша из двух сцен), а латенты одного и того же VAE — это координаты
//      в одном пространстве признаков.
//   3. Каждый промежуточный латент декодируется обратно в картинку.
//   4. Первый и последний кадры — исходные опорные картинки (точные, не
//      реконструкции), остальные — сгенерированные моделью.
//   5. Кадры собираются в MP4 (H.264) системным AVAssetWriter.
//
// Честно о границах: это интерполяция в пространстве обученного VAE, а не
// диффузионная генерация движения. Пока нет диффузионной модели (Этап B
// целиком), промежуточные кадры несут в себе качество реконструкции VAE
// (PSNR ≈ 26.9 дБ на полном наборе) и не «дорисовывают» новое движение.

import Foundation
import ImageIO

// ---------- Чистая часть: план, easing, смешивание ----------

/// План интерполяции: сколько кадров, с какой частотой и в какие моменты.
public struct InterpolationPlan: Equatable {
    /// Запрошенная длительность, с.
    public let seconds: Double
    /// Частота кадров, fps.
    public let fps: Int
    /// Число кадров в видео (включая первый и последний).
    public let frameCount: Int
    /// Моменты времени 0…1 (линейные, без easing) — по одному на кадр.
    public let times: [Double]
}

/// Число кадров видео по длительности и частоте.
///
/// Минимум 2: первый кадр — опорная картинка A, последний — B.
/// Некорректные аргументы (длительность ≤ 0, fps ≤ 0) дают те же 2 кадра:
/// пустое или вырожденное видео — это ошибка, а не «видео из одного кадра».
public func interpolationFrameCount(seconds: Double, fps: Int) -> Int {
    guard seconds.isFinite, seconds > 0, fps > 0 else { return 2 }
    let count = Int((seconds * Double(fps)).rounded())
    return max(2, count)
}

/// Моменты времени кадров: 0…1 равномерно, оба конца включены.
public func interpolationTimes(frameCount: Int) -> [Double] {
    guard frameCount >= 2 else { return [0, 1] }
    let last = Double(frameCount - 1)
    return (0..<frameCount).map { Double($0) / last }
}

/// Плавная кривая (smoothstep): на концах 0 и 1, в середине 0.5.
///
/// Линейная интерполяция даёт резкий старт и резкую остановку — в видео это
/// читается как «рывок». Smoothstep разгоняет и тормозит движение плавно.
public func smoothstep(_ t: Double) -> Double {
    guard t.isFinite else { return 0 }
    let x = min(max(t, 0), 1)
    return x * x * (3 - 2 * x)
}

/// Смешивание двух векторов: `a + (b − a)·t`.
///
/// Чистая функция: ею считается промежуточный латент (латент у модели
/// маленький — 16×7×10 = 1120 чисел, CPU тут быстрее любого GPU-ядра).
public func interpolateScalars(_ a: [Float], _ b: [Float], t: Double) -> [Float] {
    guard a.count == b.count, !a.isEmpty, t.isFinite else { return a }
    let k = Float(min(max(t, 0), 1))
    var out = [Float](repeating: 0, count: a.count)
    for i in 0..<a.count {
        out[i] = a[i] + (b[i] - a[i]) * k
    }
    return out
}

/// Полный план интерполяции.
public func interpolationPlan(seconds: Double, fps: Int) -> InterpolationPlan {
    let count = interpolationFrameCount(seconds: seconds, fps: fps)
    return InterpolationPlan(seconds: seconds,
                             fps: max(fps, 1),
                             frameCount: count,
                             times: interpolationTimes(frameCount: count))
}

// ---------- Отмена ----------

/// Флаг отмены интерполяции.
///
/// Отдельный от `TrainingControl`: кнопка «Стоп» у интерполяции не должна
/// отменять обучение (и наоборот).
public final class InterpolationControl {
    public static let shared = InterpolationControl()

    private let lock = NSLock()
    private var cancelled = false

    public var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelled
    }

    /// Сброс флага перед новым запуском.
    public func reset() {
        lock.lock(); cancelled = false; lock.unlock()
    }

    /// Запросить остановку текущего прогона.
    public func cancel() {
        lock.lock(); cancelled = true; lock.unlock()
    }
}

// ---------- GPU: кодирование и декодирование латента ----------

/// Ошибки интерполяции — с понятным человеку текстом.
public enum InterpolationError: Error, LocalizedError {
    case missingFile(String)
    case badImage(String)
    case badDuration(Double)
    case badFps(Int)
    case badSize(Int, Int)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .missingFile(let path):
            return "файл не найден: \(path)"
        case .badImage(let path):
            return "не удалось прочитать картинку: \(path)"
        case .badDuration(let seconds):
            return "длительность \(seconds) с недопустима (нужно > 0)"
        case .badFps(let fps):
            return "частота кадров \(fps) недопустима (нужно 1…120)"
        case .badSize(let w, let h):
            return "размер кадра \(w)×\(h) недопустим"
        case .cancelled:
            return "интерполяция остановлена"
        }
    }
}

/// Кодирует картинку в латент обученной модели (детерминированно, без шума).
///
/// - Параметры:
///   - vae: обученная модель;
///   - image: RGB [0,1], раскладка HWC;
///   - w, h: размер кадра (кратен 8).
/// - Возвращает: латент — C × (h/8 × (w/8)) значений.
public func encodeLatent(vae: ConvVAEGPU, image: [Float], w: Int, h: Int) throws -> [Float] {
    guard w >= 8, h >= 8, w % 8 == 0, h % 8 == 0,
          image.count >= w * h * 3 else {
        throw InterpolationError.badSize(w, h)
    }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let input = GPUTensor(engine: engine, rows: 3, cols: w * h)
    session.begin()
    session.upload(input, from: hwcToChw(image))
    let encoded = vae.encode(session, image: input, w: w, h: h, deterministic: true)
    session.end()
    return encoded.mu.toArray()
}

/// Декодирует латент в картинку (RGB [0,1], раскладка HWC).
public func decodeLatent(vae: ConvVAEGPU, latent: [Float], w: Int, h: Int) throws -> [Float] {
    guard w >= 8, h >= 8, w % 8 == 0, h % 8 == 0 else {
        throw InterpolationError.badSize(w, h)
    }
    let channels = max(vae.latentChannels, 1)
    guard latent.count >= channels, latent.count % channels == 0 else {
        throw InterpolationError.badSize(w, h)
    }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let z = GPUTensor(engine: engine, rows: channels, cols: latent.count / channels)
    session.begin()
    session.upload(z, from: latent)
    let decoded = vae.decode(session, z: z, w: w, h: h)
    session.end()
    return chwToHwc(decoded.recon.toArray())
}

// ---------- Продуктовый путь: А → Б + длительность → видео ----------

/// Запрос на интерполяцию.
public struct InterpolationRequest {
    /// Путь к опорной картинке A (первый кадр).
    public var imageAPath: String
    /// Путь к опорной картинке B (последний кадр).
    public var imageBPath: String
    /// Длительность видео, с.
    public var seconds: Double
    /// Частота кадров.
    public var fps: Int
    /// Куда сохранить .mp4.
    public var outputPath: String
    /// Рабочий размер модели (ширина), кратен 8.
    public var modelWidth: Int
    /// Рабочий размер модели (высота), кратен 8.
    public var modelHeight: Int
    /// Путь к чекпоинту обученного VAE.
    public var checkpointPath: String

    public init(imageAPath: String, imageBPath: String, seconds: Double, fps: Int,
                outputPath: String, modelWidth: Int, modelHeight: Int, checkpointPath: String) {
        self.imageAPath = imageAPath
        self.imageBPath = imageBPath
        self.seconds = seconds
        self.fps = fps
        self.outputPath = outputPath
        self.modelWidth = modelWidth
        self.modelHeight = modelHeight
        self.checkpointPath = checkpointPath
    }
}

/// Результат интерполяции.
public struct InterpolationResult {
    /// Путь к готовому .mp4.
    public let path: String
    /// Число кадров в видео.
    public let frameCount: Int
    /// Частота кадров.
    public let fps: Int
    /// Длительность, с.
    public let seconds: Double
    /// Ширина видео (как у опорной картинки A).
    public let width: Int
    /// Высота видео.
    public let height: Int
}

/// Размер картинки в пикселях (без декодирования данных).
public func imagePixelSize(path: String) -> (width: Int, height: Int)? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
          let width = props[kCGImagePropertyPixelWidth] as? Int,
          let height = props[kCGImagePropertyPixelHeight] as? Int,
          width > 0, height > 0 else { return nil }
    return (width, height)
}

/// Интерполирует между двумя картинками и собирает видео.
///
/// - Параметры:
///   - request: параметры (см. `InterpolationRequest`);
///   - onProgress: (готово, всего) — прогресс по кадрам.
/// - Возвращает: результат (путь, число кадров, размеры).
public func interpolateVideo(_ request: InterpolationRequest,
                             onProgress: ((Int, Int) -> Void)? = nil) throws -> InterpolationResult {
    let fm = FileManager.default
    guard fm.fileExists(atPath: request.imageAPath) else {
        throw InterpolationError.missingFile(request.imageAPath)
    }
    guard fm.fileExists(atPath: request.imageBPath) else {
        throw InterpolationError.missingFile(request.imageBPath)
    }
    guard fm.fileExists(atPath: request.checkpointPath) else {
        throw InterpolationError.missingFile(request.checkpointPath)
    }
    guard request.seconds.isFinite, request.seconds > 0 else {
        throw InterpolationError.badDuration(request.seconds)
    }
    guard request.fps > 0, request.fps <= 120 else {
        throw InterpolationError.badFps(request.fps)
    }

    // Рабочий размер модели: кратен 8 (требование свёрток).
    let pw = max(8, ((max(request.modelWidth, 8) + 7) / 8) * 8)
    let ph = max(8, ((max(request.modelHeight, 8) + 7) / 8) * 8)

    guard let imageA = loadImageRGB(path: request.imageAPath, w: pw, h: ph) else {
        throw InterpolationError.badImage(request.imageAPath)
    }
    guard let imageB = loadImageRGB(path: request.imageBPath, w: pw, h: ph) else {
        throw InterpolationError.badImage(request.imageBPath)
    }

    // Размер видео — как у опорной картинки A (H.264 требует чётные стороны).
    let sourceSize = imagePixelSize(path: request.imageAPath) ?? (pw, ph)
    let outW = max(2, sourceSize.width - sourceSize.width % 2)
    let outH = max(2, sourceSize.height - sourceSize.height % 2)

    // Модель: обученный VAE из чекпоинта. Архитектуру (residual/groupNorm по
    // archVersion, ширину каналов, число каналов латента) ОБЯЗАТЕЛЬНО берём из
    // метаданных чекпоинта — иначе `load` бросит «несовпадение формы слоя» для
    // любой non-plain модели (регрессия, вскрытая живым E2E: /api/train с
    // residual=true, затем /api/interpolate падал без выходного mp4).
    let st = try loadStateWithMeta(path: request.checkpointPath)
    let vae = ConvVAEGPU(engine: MetalEngine.shared,
                         latentChannels: st.latentChannels,
                         width: st.modelWidth,
                         residual: st.archVersion >= 2,
                         groupNorm: st.archVersion >= 3)
    try vae.load(from: st.mats)

    // Латенты опорных картинок.
    let latentA = try encodeLatent(vae: vae, image: imageA, w: pw, h: ph)
    let latentB = try encodeLatent(vae: vae, image: imageB, w: pw, h: ph)

    // Кадры.
    let plan = interpolationPlan(seconds: request.seconds, fps: request.fps)
    var frames: [[Float]] = []
    frames.reserveCapacity(plan.frameCount)
    for (index, linearTime) in plan.times.enumerated() {
        if InterpolationControl.shared.isCancelled { throw InterpolationError.cancelled }
        let eased = smoothstep(linearTime)
        let latent = interpolateScalars(latentA, latentB, t: eased)
        let frame: [Float]
        if index == 0 {
            frame = imageA                       // первый кадр — точная опора A
        } else if index == plan.frameCount - 1 {
            frame = imageB                       // последний — точная опора B
        } else {
            frame = try decodeLatent(vae: vae, latent: latent, w: pw, h: ph)
        }
        frames.append(frame)
        onProgress?(index + 1, plan.frameCount)
    }

    // Видео.
    let written = try writeVideoMP4(frames: frames, width: pw, height: ph, fps: plan.fps,
                                    path: request.outputPath,
                                    outputWidth: outW, outputHeight: outH)
    return InterpolationResult(path: request.outputPath,
                               frameCount: written,
                               fps: plan.fps,
                               seconds: request.seconds,
                               width: outW,
                               height: outH)
}
