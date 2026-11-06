import Foundation
import AVFoundation
import VideoToolbox
import ImageIO
import UniformTypeIdentifiers

/// Программный (софтверный) декодер H.264 через VideoToolbox.
///
/// В отличие от AVAssetImageGenerator (аппаратный видеодекодер, один на
/// систему, сериализованный), здесь декодирование идёт на CPU — и его
/// реально можно распараллелить, а в последовательном варианте он
/// значительно быстрее на большом числе ядер (i9-9880H -> 100+ fps).
///
/// Потокобезопасность: один экземпляр — один поток.
final class SoftwareFrameDecoder {
    private let asset: AVURLAsset
    private var session: VTDecompressionSession?
    private let lock = NSLock()
    private var _cancelled = false
    private(set) var totalFrames = 0

    init(url: URL) {
        self.asset = AVURLAsset(url: url)
    }

    func cancel() {
        lock.lock(); _cancelled = true; lock.unlock()
    }

    private var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _cancelled
    }

    /// Извлекает кадры последовательно (программным декодером).
    /// - Parameters:
    ///   - targetCount: сколько кадров нужно итого (0 — все).
    ///   - startIndex: смещение глобальной нумерации.
    ///   - onFrame: (порядковый номер выбранного кадра, всего, глобальный
    ///     порядковый номер, JPEG-данные)
    ///   - progress: 0...1 по мере чтения потока.
    func run(targetCount: Int,
             startIndex: Int = 0,
             onFrame: @escaping (_ index: Int, _ total: Int, _ ordinal: Int, _ jpeg: Data) -> Void,
             progress: @escaping (Double) -> Void,
             completion: @escaping (Result<Void, Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.decode(targetCount: targetCount, startIndex: startIndex,
                                onFrame: onFrame, progress: progress, completion: completion)
            } catch {
                completion(.failure(error))
            }
        }
    }

    private func decode(targetCount: Int,
                        startIndex: Int,
                        onFrame: @escaping (Int, Int, Int, Data) -> Void,
                        progress: @escaping (Double) -> Void,
                        completion: @escaping (Result<Void, Error>) -> Void) throws {
        guard !isCancelled else {
            completion(.failure(FrameExtractError.cancelled)); return
        }

        guard let time = try? loadAssetDuration(asset) else {
            completion(.failure(FrameExtractError.unreadableAsset)); return
        }
        let duration = CMTimeGetSeconds(time)
        guard duration.isFinite, duration > 0.01 else {
            completion(.failure(FrameExtractError.unreadableAsset)); return
        }

        let tracks = (try? loadVideoTracks(of: asset)) ?? []
        // fps из видео-трека
        let fps = tracks.first.flatMap { try? loadNominalFrameRate(of: $0) } ?? 24
        let allFrames = max(1, Int(duration * Double(fps)))

        // сколько реально берём
        let effectiveMax = targetCount > 0 ? targetCount : allFrames
        let target = min(allFrames, effectiveMax)

        // stride: берём каждый k-й кадр
        let stride = max(1, allFrames / max(target, 1))

        let reader = try AVAssetReader(asset: asset)
        guard let track = tracks.first else {
            completion(.failure(FrameExtractError.unreadableAsset)); return
        }
        // БЕЗ outputSettings: читаем сжатые семплы (как есть), декодируем сами
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            completion(.failure(reader.error ?? FrameExtractError.unreadableAsset)); return
        }

        var frameNo = 0        // счётчик всех кадров в потоке
        var taken = 0         // сколько уже взяли
        var writtenSeen = 0   // глобальная нумерация
        totalFrames = allFrames

        while let sample = output.copyNextSampleBuffer() {
            if isCancelled { reader.cancelReading(); completion(.failure(FrameExtractError.cancelled)); return }

            // Термо-контроль: каждые 32 кадра проверяем тепловой уровень.
            if frameNo % 32 == 0 {
                switch ProcessInfo.processInfo.thermalState {
                case .critical:
                    yumeLog("⚠️ извлечение: критический перегрев — пауза 60 с")
                    Thread.sleep(forTimeInterval: 60)
                    while ProcessInfo.processInfo.thermalState == .critical {
                        Thread.sleep(forTimeInterval: 30)
                    }
                case .serious:
                    Thread.sleep(forTimeInterval: 0.02)
                default:
                    break
                }
            }

            let take = (frameNo % stride == 0) && (taken < target)
            if take {
                // Лениво создаём сессию декодера на первом сэмпле
                if session == nil {
                    guard let fmt = CMSampleBufferGetFormatDescription(sample) else {
                        reader.cancelReading()
                        completion(.failure(FrameExtractError.unreadableAsset)); return
                    }
                    session = try Self.makeSoftwareSession(formatDescription: fmt)
                }
                if let img = try Self.decompress(sample: sample, session: session!),
                   let jpeg = Self.jpegData(img) {
                    let ordinal = writtenSeen + startIndex
                    writtenSeen += 1
                    taken += 1
                    onFrame(frameNo, target, ordinal, jpeg)
                }
            }

            frameNo += 1
            if frameNo % 25 == 0 || frameNo >= allFrames {
                progress(Double(frameNo) / Double(allFrames))
            }
        }

        session = nil
        if reader.status == .cancelled {
            completion(.failure(FrameExtractError.cancelled))
        } else if taken == 0 {
            completion(.failure(FrameExtractError.unreadableAsset))
        } else {
            progress(1.0)
            completion(.success(()))
        }
    }

    /// Сессия программного декодера: аппаратное ускорение запрещено.
    private static func makeSoftwareSession(formatDescription fmt: CMFormatDescription) throws -> VTDecompressionSession {
        let spec: [CFString: Any] = [
            kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: false
        ]
        var outSession: VTDecompressionSession?
        let status = VTDecompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            formatDescription: fmt,
            decoderSpecification: spec as CFDictionary,
            imageBufferAttributes: nil,
            outputCallback: nil,
            decompressionSessionOut: &outSession
        )
        guard status == noErr, let s = outSession else {
            throw NSError(domain: "Yume", code: 30,
                          userInfo: [NSLocalizedDescriptionKey: "не удалось создать программный декодер (\(status))"])
        }
        return s
    }

    /// Декодирует один сэмпл в CGImage (синхронно).
    private static func decompress(sample: CMSampleBuffer, session: VTDecompressionSession) throws -> CGImage? {
        let sem = DispatchSemaphore(value: 0)
        var resultImage: CGImage?
        var decodeError: Error?

        let status = VTDecompressionSessionDecodeFrame(
            session,
            sampleBuffer: sample,
            flags: [],
            infoFlagsOut: nil
        ) { status, _, imageBuffer, _, _ in
            if status == noErr, let buf = imageBuffer {
                var cg: CGImage?
                let cStatus = VTCreateCGImageFromCVPixelBuffer(buf, options: nil, imageOut: &cg)
                if cStatus == noErr { resultImage = cg }
                else { decodeError = NSError(domain: "Yume", code: 31,
                                             userInfo: [NSLocalizedDescriptionKey: "VTCreateCGImage (\(cStatus))"]) }
            } else {
                decodeError = NSError(domain: "Yume", code: 32,
                                      userInfo: [NSLocalizedDescriptionKey: "декод статус \(status)"])
            }
            sem.signal()
        }
        guard status == noErr else {
            throw NSError(domain: "Yume", code: 33,
                          userInfo: [NSLocalizedDescriptionKey: "VTDecode не принял сэмпл (\(status))"])
        }
        sem.wait()
        if let e = decodeError { throw e }
        return resultImage
    }

    private static func jpegData(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }
}
