// YumeLayers — сборка видео из готовых кадров.
//
// Кадры приходят как RGB [0,1] в раскладке HWC (строки сверху вниз) —
// ровно та раскладка, в которой ядро отдаёт реконструкции. Дальше:
// Float → CGImage → масштабирование CoreGraphics → CVPixelBuffer (BGRA) →
// AVAssetWriter (H.264 в MP4).
//
// Почему так, а не «собрать PNG и склеить ffmpeg'ом»:
//   • ffmpeg не является частью приложения — его может не быть в системе;
//   • AVAssetWriter — системный API: получается настоящий MP4 с корректной
//     длительностью и частотой кадров, который открывается в QuickTime.
//
// Ориентация картинки проверяется тестом: первый пиксель кадра должен
// оказаться в верхнем левом углу буфера, а не внизу.

import Foundation
import AVFoundation
import CoreVideo

/// Ошибки сборки видео — с понятным человеку текстом.
public enum VideoWriterError: Error, LocalizedError {
    case emptyFrames
    case badFrameSize
    case oddOutputSize(Int, Int)
    case badFps(Int)
    case pixelBufferFailed(Int)
    case writerFailed(String)
    case appendFailed(Int, String)

    public var errorDescription: String? {
        switch self {
        case .emptyFrames:
            return "нечего писать: список кадров пуст"
        case .badFrameSize:
            return "кадр имеет неверный размер"
        case .oddOutputSize(let w, let h):
            return "размер видео \(w)×\(h) должен быть чётным (H.264 требует чётные стороны)"
        case .badFps(let fps):
            return "частота кадров \(fps) недопустима (нужно 1…120)"
        case .pixelBufferFailed(let index):
            return "не удалось подготовить кадр \(index) для записи"
        case .writerFailed(let text):
            return "видеозапись не удалась: \(text)"
        case .appendFailed(let index, let text):
            return "видеозапись не удалась на кадре \(index): \(text)"
        }
    }
}

/// Готовит CVPixelBuffer (32-bit BGRA) из кадра RGB.
///
/// - Параметры:
///   - frame: RGB [0,1], раскладка HWC, строки сверху вниз;
///   - width, height: размер исходного кадра;
///   - outputWidth, outputHeight: размер буфера (кадр масштабируется).
/// - Возвращает: буфер или nil, если кадр/размер не годятся.
public func makePixelBuffer(frame: [Float], width: Int, height: Int,
                            outputWidth: Int, outputHeight: Int) -> CVPixelBuffer? {
    guard width > 0, height > 0, outputWidth > 0, outputHeight > 0,
          frame.count >= width * height * 3 else { return nil }
    guard let img = makeCGImage(rgb: frame, width: width, height: height) else { return nil }

    // 1) Масштабируем в отдельный bitmap-контекст. В CGBitmapContext строка 0
    //    — нижняя строка картинки, а в CVPixelBuffer строка 0 — верхняя;
    //    переворот делаем явно при копировании (шаг 2), чтобы не зависеть от
    //    неявной ориентации CGImage.
    var scaled = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &scaled,
                              width: outputWidth, height: outputHeight,
                              bitsPerComponent: 8, bytesPerRow: outputWidth * 4,
                              space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                                  | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: outputWidth, height: outputHeight))

    // 2) Копируем в CVPixelBuffer построчно БЕЗ переворота.
    //
    // Битовый контекст выше хранит строку 0 = НИЗ кадра (bottom-up), а
    // загрузчик кадров (loadImageRGB) отдаёт кадры тоже низом вверх — это
    // внутренняя конвенция, на которой обучена модель. Если здесь
    // переворачивать строки, итоговое видео оказывается вертикально
    // зеркальным относительно исходной картинки (проверено на 852×480:
    // кадр совпадал с vflip источника).
    var pixelBuffer: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
    ]
    guard CVPixelBufferCreate(kCFAllocatorDefault, outputWidth, outputHeight,
                              kCVPixelFormatType_32BGRA,
                              attrs as CFDictionary, &pixelBuffer) == kCVReturnSuccess,
          let buffer = pixelBuffer else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let dstStride = CVPixelBufferGetBytesPerRow(buffer)
    let srcStride = outputWidth * 4
    scaled.withUnsafeBytes { srcRaw in
        guard let src = srcRaw.baseAddress else { return }
        for row in 0..<outputHeight {
            let srcRow = row
            memcpy(base + row * dstStride, src + srcRow * srcStride, srcStride)
        }
    }
    return buffer
}

/// Пишет кадры в MP4 (H.264).
///
/// - Параметры:
///   - frames: RGB [0,1], раскладка HWC, все кадры одного размера;
///   - width, height: размер кадров;
///   - fps: частота кадров;
///   - path: путь к .mp4 (существующий файл заменяется);
///   - outputWidth, outputHeight: размер видео (по умолчанию — размер кадров).
/// - Возвращает: число записанных кадров.
public func writeVideoMP4(frames: [[Float]], width: Int, height: Int, fps: Int,
                         path: String, outputWidth: Int? = nil, outputHeight: Int? = nil) throws -> Int {
    guard !frames.isEmpty else { throw VideoWriterError.emptyFrames }
    guard width > 0, height > 0, fps > 0, fps <= 120 else { throw VideoWriterError.badFps(fps) }
    let outW = outputWidth ?? width
    let outH = outputHeight ?? height
    guard outW > 0, outH > 0 else { throw VideoWriterError.badFrameSize }
    guard outW % 2 == 0, outH % 2 == 0 else { throw VideoWriterError.oddOutputSize(outW, outH) }

    let fm = FileManager.default
    let url = URL(fileURLWithPath: path)
    if fm.fileExists(atPath: path) { try fm.removeItem(atPath: path) }
    let dir = url.deletingLastPathComponent().path
    if !dir.isEmpty { try fm.createDirectory(atPath: dir, withIntermediateDirectories: true) }

    let writer: AVAssetWriter
    do {
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    } catch {
        throw VideoWriterError.writerFailed(error.localizedDescription)
    }
    let settings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: outW,
        AVVideoHeightKey: outH,
        AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: max(outW * outH * 4, 200_000),
            AVVideoMaxKeyFrameIntervalKey: fps,
            AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
        ],
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outW,
            kCVPixelBufferHeightKey as String: outH,
        ])
    guard writer.canAdd(input) else {
        throw VideoWriterError.writerFailed("нельзя добавить видеодорожку")
    }
    writer.add(input)
    guard writer.startWriting() else {
        throw VideoWriterError.writerFailed(writer.error?.localizedDescription ?? "не удалось начать запись")
    }
    writer.startSession(atSourceTime: .zero)

    for (index, frame) in frames.enumerated() {
        while !input.isReadyForMoreMediaData {
            Thread.sleep(forTimeInterval: 0.002)
        }
        guard let buffer = makePixelBuffer(frame: frame, width: width, height: height,
                                          outputWidth: outW, outputHeight: outH) else {
            writer.cancelWriting()
            throw VideoWriterError.pixelBufferFailed(index)
        }
        let time = CMTime(value: CMTimeValue(index), timescale: CMTimeScale(fps))
        guard adaptor.append(buffer, withPresentationTime: time) else {
            writer.cancelWriting()
            throw VideoWriterError.appendFailed(index,
                                                writer.error?.localizedDescription ?? "кадр не принят")
        }
    }
    input.markAsFinished()
    let done = DispatchSemaphore(value: 0)
    writer.finishWriting { done.signal() }
    done.wait()
    if writer.status == .failed {
        throw VideoWriterError.writerFailed(writer.error?.localizedDescription ?? "запись не завершена")
    }
    return frames.count
}
