import Foundation
import AVFoundation
import CoreMedia

/// Вытащить `count` кадров ПЛОТНО из одного шота `[start, end]`.
///
/// В отличие от `loadVideoFrames` (одно окно), здесь кадры берутся внутри
/// найденной сцены, где фон стабилен — временная медиана даёт корректный
/// фон, а не кашу из разных сцен.
public func loadShotFrames(url: URL, w: Int, h: Int,
                           start: Double, end: Double,
                           count: Int = 12) throws -> [[Float]] {
    let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
    gen.appliesPreferredTrackTransform = true
    gen.maximumSize = CGSize(width: w, height: h)
    gen.requestedTimeToleranceBefore = .zero
    gen.requestedTimeToleranceAfter = .zero
    let n = max(count, 2)
    var frames: [[Float]] = []
    frames.reserveCapacity(n)
    for i in 0..<n {
        let t = start + (end - start) * Double(i) / Double(n - 1)
        let time = CMTime(seconds: t, preferredTimescale: 600)
        if let cg = try? gen.copyCGImage(at: time, actualTime: nil),
           let rgb = decodeImageRGB(cg, w: w, h: h) {
            frames.append(rgb)
        }
    }
    return frames
}
