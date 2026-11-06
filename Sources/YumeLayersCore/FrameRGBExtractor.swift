// YumeLayers Core — многопоточное программное извлечение RGB-кадров.
//
// Раньше датасет/шоты/кадры шотов тянулись через `AVAssetImageGenerator`
// и синхронный мост `loadCGImage`, который декодировал кадры ПО ОДНОМУ.
// Здесь декодирование идёт параллельно: несколько процессов ffmpeg
// (программный декод) работают одновременно, каждый отдаёт кадр в
// rawvideo RGB24, который сразу переводится в [Float] 0...1.

import Foundation
import AVFoundation
import CoreMedia

/// Ошибка извлечения RGB-кадров.
public struct FrameRGBError: Error, LocalizedError {
    public let message: String
    public var errorDescription: String? { message }
    public init(_ message: String) { self.message = message }
}

/// Потокобезопасный сборщик кадров: все блокировки спрятаны в
/// синхронные методы, поэтому их можно вызывать из async-задач.
private final class FrameCollector {
    private let lock = NSLock()
    private var frames: [[Float]]
    private var firstError: Error?

    init(count: Int) {
        frames = [[Float]](repeating: [], count: count)
    }

    func set(_ rgb: [Float], at index: Int) {
        lock.lock(); frames[index] = rgb; lock.unlock()
    }

    func fail(_ error: Error) {
        lock.lock(); if firstError == nil { firstError = error }; lock.unlock()
    }

    var failed: Bool {
        lock.lock(); defer { lock.unlock() }
        return firstError != nil
    }

    func snapshot() -> (frames: [[Float]], error: Error?) {
        lock.lock(); defer { lock.unlock() }
        return (frames, firstError)
    }
}

public enum FrameRGBExtractor {
    /// Путь к ffmpeg (общий с параллельным извлекателем GUI).
    static var ffmpegPath: String? { ParallelExtractor.ffmpegPath }

    /// Извлекает кадры для указанных моментов времени, строго в том же
    /// порядке. Декодирование распараллелено на все доступные ядра.
    public static func loadFrames(url: URL, times: [Double],
                                  w: Int, h: Int) throws -> [[Float]] {
        guard !times.isEmpty, w > 0, h > 0 else { return [] }
        let cores = max(1, min(ProcessInfo.processInfo.activeProcessorCount, 10))

        if ffmpegPath != nil {
            return try loadWithFFmpeg(url: url, times: times,
                                      w: w, h: h, cores: cores)
        }
        return try loadWithAVFoundation(url: url, times: times,
                                        w: w, h: h, cores: cores)
    }

    // MARK: - ffmpeg (программный параллельный декод)

    private static func loadWithFFmpeg(url: URL, times: [Double],
                                       w: Int, h: Int,
                                       cores: Int) throws -> [[Float]] {
        guard let ffmpeg = ffmpegPath else {
            throw FrameRGBError("ffmpeg не найден")
        }
        let count = times.count
        let frameBytes = w * h * 3
        let collector = FrameCollector(count: count)

        DispatchQueue.concurrentPerform(iterations: min(cores, count)) { worker in
            var i = worker
            while i < count {
                if collector.failed { return }
                let time = times[i]
                do {
                    let rgb = try decodeFrame(ffmpeg: ffmpeg, url: url,
                                              time: time, w: w, h: h,
                                              expectedBytes: frameBytes)
                    collector.set(rgb, at: i)
                } catch {
                    collector.fail(error)
                    return
                }
                i += cores
            }
        }

        let result = collector.snapshot()
        if let failure = result.error { throw failure }
        return result.frames
    }

    /// Один кадр в момент `time`: отдельный процесс ffmpeg, rawvideo RGB24.
    /// Если после запрошенной точки уже нет кадров (конец ролика), берём
    /// ближайший предыдущий — так же, как это делал AVAssetImageGenerator.
    private static func decodeFrame(ffmpeg: String, url: URL, time: Double,
                                    w: Int, h: Int,
                                    expectedBytes: Int) throws -> [Float] {
        var candidates = [max(0, time)]
        if time > 0.3 { candidates.append(time - 0.3) }
        if time > 1.0 { candidates.append(time * 0.5) }
        candidates.append(0)

        for t in candidates {
            if let data = try decode(ffmpeg: ffmpeg, url: url, time: t,
                                     w: w, h: h, expectedBytes: expectedBytes) {
                return rgbToFloats(data.prefix(expectedBytes), pixels: w * h)
            }
        }
        throw FrameRGBError("кадр не декодировался (t=\(time))")
    }

    private static func decode(ffmpeg: String, url: URL, time: Double,
                               w: Int, h: Int,
                               expectedBytes: Int) throws -> Data? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpeg)
        process.arguments = [
            "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", String(format: "%.4f", max(0, time)),
            "-i", url.path,
            "-frames:v", "1",
            "-vf", "scale=\(w):\(h):flags=bicubic",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-",
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            throw FrameRGBError("ffmpeg не запустился: \(error.localizedDescription)")
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0, data.count >= expectedBytes else {
            return nil
        }
        return data
    }

    // MARK: - AVFoundation (параллельно, без поштучного моста)

    private static func loadWithAVFoundation(url: URL, times: [Double],
                                             w: Int, h: Int,
                                             cores: Int) throws -> [[Float]] {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: w, height: h)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let collector = FrameCollector(count: times.count)

        // Мост один на весь пакет: внутри кадры декодируются конкурентно.
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            await withTaskGroup(of: (Int, Result<[Float], Error>).self) { group in
                var next = 0
                let parallel = max(1, min(cores, times.count))
                func submit(_ i: Int) {
                    let t = times[i]
                    group.addTask {
                        do {
                            let cg = try await generator.image(
                                at: CMTime(seconds: t, preferredTimescale: 600)).image
                            guard let rgb = decodeImageRGB(cg, w: w, h: h) else {
                                throw FrameRGBError("кадр не декодировался (t=\(t))")
                            }
                            return (i, .success(rgb))
                        } catch {
                            return (i, .failure(error))
                        }
                    }
                }
                for _ in 0..<parallel { submit(next); next += 1 }
                for await (i, result) in group {
                    switch result {
                    case .success(let rgb):
                        collector.set(rgb, at: i)
                    case .failure(let error):
                        collector.fail(error)
                    }
                    if next < times.count { submit(next); next += 1 }
                }
            }
            semaphore.signal()
        }
        semaphore.wait()

        let collected = collector.snapshot()
        if let failure = collected.error { throw failure }
        return collected.frames
    }

    // MARK: - Байты → Float

    static func rgbToFloats(_ bytes: Data, pixels: Int) -> [Float] {
        var out = [Float](repeating: 0, count: pixels * 3)
        bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for i in 0..<(pixels * 3) {
                out[i] = Float(base[i]) / 255.0
            }
        }
        return out
    }
}
