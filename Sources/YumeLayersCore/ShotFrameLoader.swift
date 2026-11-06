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
    let n = max(count, 2)
    let times = (0..<n).map { start + (end - start) * Double($0) / Double(n - 1) }
    // Многопоточный программный декод: кадры шота тянутся параллельно.
    let frames = try FrameRGBExtractor.loadFrames(url: url, times: times,
                                                  w: w, h: h)
    return frames.filter { !$0.isEmpty }
}
