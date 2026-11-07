// YumeLayers Core — извлечение кадров из видеофайла (MP4/MOV) через AVFoundation.
// Чистый нативный код: без сторонних ML-фреймворков. Кадры пишутся в JPEG
// в ~/YumeLayers/frames и сразу отдаются по одному (для мгновенного показа).

import Foundation
import AVFoundation
import ImageIO
import UniformTypeIdentifiers

public enum FrameExtractError: Error, LocalizedError {
    case unreadableAsset
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .unreadableAsset: return "Не удалось прочитать видеофайл"
        case .cancelled: return "Извлечение отменено"
        }
    }
}

public final class FrameExtractor {
    public struct Options {
        /// Сколько ПЕРВЫХ кадров извлечь на одно видео. 0 — авто: лимит
        /// считается от доступной памяти (сколько кадров влезет в RAM,
        /// ~60 КБ на кадр), т.е. без произвольных «отсебятиных» значений.
        /// Лимит не меняет частоту: кадры берутся подряд при fps исходника.
        public var maxFrames: Int = 0
        /// `true` — извлекать ВСЕ кадры с частотой исходного видео (fps из трека).
        /// `false` — использовать `everySeconds` (шаг в секундах).
        public var useSourceFps = true
        /// Базовый шаг между кадрами в секундах (используется при useSourceFps=false).
        public var everySeconds: Double = 1.0
        /// Смещение нумерации кадров (для строгого порядка при нескольких видео).
        public var startIndex: Int = 0
        public init(maxFrames: Int = 0, useSourceFps: Bool = true,
                    everySeconds: Double = 1.0, startIndex: Int = 0) {
            self.maxFrames = maxFrames
            self.useSourceFps = useSourceFps
            self.everySeconds = everySeconds
            self.startIndex = startIndex
        }
    }

    /// Авто-лимит кадров от доступной памяти: ~60 КБ на кадр (JPEG),
    /// используем до 3/4 физической RAM.
    private func autoMaxFrames() -> Int {
        let perFrameBytes = 64 * 1024
        let ram = UInt64(ProcessInfo.processInfo.physicalMemory)
        let frames = Int(ram * 3 / 4 / UInt64(perFrameBytes))
        return max(10_000, frames)
    }

    private let url: URL
    private let options: Options
    private let lock = NSLock()
    private var _cancelled = false
    /// Параллельный (на ~10 ядрах) извлекатель кадров — для отмены.
    private var parallelExtractor: ParallelExtractor?
    private var softwareDecoder: SoftwareFrameDecoder?
    /// Сколько лидирующих полностью чёрных кадров пропущено (артефакт декодирования).
    public private(set) var skippedLeadingBlack = 0

    public init(url: URL, options: Options = Options()) {
        self.url = url
        self.options = options
    }

    public func cancel() {
        lock.lock(); _cancelled = true
        parallelExtractor?.cancel()
        softwareDecoder?.cancel()
        lock.unlock()
    }

    private var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _cancelled
    }

    /// Частота кадров исходного видео (fps из видео-трека; 24 по умолчанию).
    private func sourceFrameRate() -> Double {
        let asset = AVURLAsset(url: url)
        guard let track = try? loadVideoTracks(of: asset).first,
              let fr = try? loadNominalFrameRate(of: track) else { return 24 }
        return fr > 0 ? Double(fr) : 24
    }

    /// Публичный доступ к fps исходного видео (для отбора контрольной выборки:
    /// 1 кадр в секунду = каждый round(fps)-й кадр).
    public func frameRate() -> Double {
        sourceFrameRate()
    }

    /// Оценка числа кадров для этого видео (без извлечения).
    public func estimatedCount() -> Int {
        let asset = AVURLAsset(url: url)
        guard let time = try? loadAssetDuration(asset) else { return 0 }
        let duration = CMTimeGetSeconds(time)
        guard duration.isFinite, duration > 0.01 else { return 0 }
        let rate = options.useSourceFps ? sourceFrameRate() : 1.0 / max(options.everySeconds, 1e-6)
        let effectiveMax = options.maxFrames > 0 ? options.maxFrames : autoMaxFrames()
        return max(1, min(effectiveMax, Int(duration * rate)))
    }

    /// Извлекает кадры в память, БЕЗ записи на диск.
    /// Основной путь — FFmpeg (установлен локально), программный/аппаратный
    /// декод + максимальная скорость. Кадры идут MJPEG-потоком через pipe
    /// и разбираются в памяти (без промежуточных файлов).
    /// - `onFrame`: вызывается для каждого кадра строго по порядку
    ///   (номер по порядку, всего, глобальный порядковый номер, JPEG-данные).
    public func run(
        onFrame: @escaping (_ index: Int, _ total: Int, _ ordinal: Int, _ jpeg: Data) -> Void,
        progress: @escaping (Double) -> Void,
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        let asset = AVURLAsset(url: url)
        guard let time = try? loadAssetDuration(asset) else {
            completion(.failure(FrameExtractError.unreadableAsset))
            return
        }
        let duration = CMTimeGetSeconds(time)
        guard duration.isFinite, duration > 0.01 else {
            completion(.failure(FrameExtractError.unreadableAsset))
            return
        }

        // ВСЕ кадры исходника: частота берётся из видео-трека (например, 24 fps).
        let rate = options.useSourceFps ? sourceFrameRate() : 1.0 / max(options.everySeconds, 1e-6)
        let effectiveMax = options.maxFrames > 0 ? options.maxFrames : autoMaxFrames()
        let total = max(1, min(effectiveMax, Int(duration * rate)))

        if ParallelExtractor.ffmpegPath != nil {
            // Параллельное извлечение через ffmpeg (на ~10 ядрах).
            let parallel = ParallelExtractor(
                url: url,
                options: ParallelExtractor.Options(maxFrames: effectiveMax,
                                                    startIndex: options.startIndex,
                                                    framesPerSecond: rate))
            lock.lock()
            parallelExtractor = parallel
            lock.unlock()
            parallel.run(onFrame: onFrame, progress: progress, completion: completion)
        } else {
            // Fallback: программный декодер VideoToolbox (без зависимости).
            let decoder = SoftwareFrameDecoder(url: url)
            lock.lock()
            softwareDecoder = decoder
            if _cancelled { decoder.cancel() }
            lock.unlock()
            decoder.run(targetCount: total, startIndex: options.startIndex,
                        onFrame: onFrame, progress: progress, completion: completion)
        }
    }

}
