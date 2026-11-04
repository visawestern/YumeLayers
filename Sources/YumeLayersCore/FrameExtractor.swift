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
        /// Максимальное число кадров на одно видео. 0 — авто: лимит считается
        /// от доступной памяти (сколько кадров влезет в RAM, ~60 КБ на кадр),
        /// т.е. без произвольных «отсебятиных» значений.
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
    /// Активный FFmpeg-процесс (для отмены).
    private var ffmpegProcess: Process?
    private var softwareDecoder: SoftwareFrameDecoder?
    /// Сколько лидирующих полностью чёрных кадров пропущено (артефакт декодирования).
    public private(set) var skippedLeadingBlack = 0

    public init(url: URL, options: Options = Options()) {
        self.url = url
        self.options = options
    }

    public func cancel() {
        lock.lock(); _cancelled = true
        if let process = ffmpegProcess, process.isRunning { process.terminate() }
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
        guard let track = asset.tracks(withMediaType: .video).first else { return 24 }
        let fr = track.nominalFrameRate
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
        let duration = CMTimeGetSeconds(asset.duration)
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
        let duration = CMTimeGetSeconds(asset.duration)
        guard duration.isFinite, duration > 0.01 else {
            completion(.failure(FrameExtractError.unreadableAsset))
            return
        }

        // ВСЕ кадры исходника: частота берётся из видео-трека (например, 24 fps).
        let rate = options.useSourceFps ? sourceFrameRate() : 1.0 / max(options.everySeconds, 1e-6)
        let effectiveMax = options.maxFrames > 0 ? options.maxFrames : autoMaxFrames()
        let total = max(1, min(effectiveMax, Int(duration * rate)))

        // Итоговая частота выборки: сколько кадров/с нужно на выходе.
        let outFps = Double(total) / duration

        if Self.ffmpegPath != nil {
            runFFmpeg(outFps: outFps, total: total,
                       onFrame: onFrame, progress: progress, completion: completion)
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

    private static let ffmpegPath: String? = {
        let candidates = [
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "/usr/bin/ffmpeg",
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }()

    /// Запуск ffmpeg: видео → MJPEG-поток в stdout.
    private func runFFmpeg(outFps: Double,
                           total: Int,
                           onFrame: @escaping (Int, Int, Int, Data) -> Void,
                           progress: @escaping (Double) -> Void,
                           completion: @escaping (Result<Void, Error>) -> Void) {
        let process = Process()
        guard let path = Self.ffmpegPath else { completion(.failure(FrameExtractError.unreadableAsset)); return }
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = [
            "-hide_banner", "-loglevel", "error",
            "-threads", "12",
            "-i", url.path,
            "-vf", String(format: "fps=%.4f", outFps),
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "3",
            "-vsync", "0",
            "-"
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        lock.lock(); ffmpegProcess = process; lock.unlock()

        process.terminationHandler = { [weak self] _ in
            guard let self = self else { return }
            self.lock.lock(); self.ffmpegProcess = nil; self.lock.unlock()
        }

        do {
            guard !isCancelled else { completion(.failure(FrameExtractError.cancelled)); return }
            try process.run()
            if isCancelled && process.isRunning { process.terminate() }
        } catch {
            completion(.failure(error))
            return
        }

        // Чтение потока и разбор JPEG-кадров.
        let handle = pipe.fileHandleForReading
        var buffer = Data()
        var frameIndex = 0
        var ordinal = 0
        var lastProgress = 0

        // Поток чтения: асинхронно, чтобы не блокировать.
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            while true {
                if self.isCancelled {
                    process.terminate()
                    completion(.failure(FrameExtractError.cancelled))
                    return
                }
                let chunk = handle.readData(ofLength: 262144)
                if chunk.isEmpty { break }
                buffer.append(chunk)

                // Разбор: ищем полные JPEG (FFD8 … FFD9/следующий FFD8).
                while true {
                    guard let start = Self.findMarker(0xFFD8, in: buffer, from: 0) else { break }
                    // Ищем следующий FFD8 после старта — это граница следующего кадра.
                    if let nextStart = Self.findMarker(0xFFD8, in: buffer, from: start + 2) {
                        let frameData = buffer.subdata(in: start..<nextStart)
                        if frameData.count > 100 {
                            let ordinalOut = ordinal
                            ordinal += 1
                            frameIndex += 1
                            onFrame(frameIndex - 1, total, ordinalOut + self.options.startIndex, frameData)
                            if frameIndex % 50 == 0 || frameIndex >= total {
                                let p = Double(frameIndex) / Double(max(total, 1))
                                if Int(p * 1000) != lastProgress {
                                    lastProgress = Int(p * 1000)
                                    progress(p)
                                }
                            }
                            if frameIndex >= total {
                                process.terminate()
                                progress(1.0)
                                completion(.success(()))
                                return
                            }
                        }
                        buffer.removeSubrange(0..<nextStart)
                    } else {
                        // Нет следующего старта — ждём ещё данных.
                        break
                    }
                }
            }
            // Поток закрыт. Обрабатываем остаток буфера: последний JPEG без
            // следующего маркера начала (иначе теряем ровно один кадр).
            if frameIndex < total,
               let start = Self.findMarker(0xFFD8, in: buffer, from: 0),
               let end = Self.findMarker(0xFFD9, in: buffer, from: start + 2) {
                let frameData = buffer.subdata(in: start..<(end + 2))
                if frameData.count > 100 {
                    let ordinalOut = ordinal
                    onFrame(frameIndex, total, ordinalOut + self.options.startIndex, frameData)
                    frameIndex += 1
                    progress(Double(frameIndex) / Double(max(total, 1)))
                }
            }

            if self.isCancelled {
                completion(.failure(FrameExtractError.cancelled))
            } else if frameIndex == 0 {
                completion(.failure(FrameExtractError.unreadableAsset))
            } else {
                progress(1.0)
                completion(.success(()))
            }
        }
    }

    /// Ищет маркер (2 байта) в Data, начиная с `from`.
    private static func findMarker(_ marker: UInt16, in data: Data, from: Int) -> Int? {
        guard from + 1 < data.count else { return nil }
        let hi = UInt8(marker >> 8)
        let lo = UInt8(marker & 0xFF)
        let bytes = [UInt8](data)
        var i = from
        while i + 1 < bytes.count {
            if bytes[i] == hi && bytes[i + 1] == lo { return i }
            i += 1
        }
        return nil
    }
}
