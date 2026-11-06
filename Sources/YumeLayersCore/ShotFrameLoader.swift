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
    // Полуинтервал [start, end): сэмпл с индексом `cut` у детектора — это
    // ПЕРВЫЙ кадр новой сцены, и сэмпл ровно в `end` давал сцене чужой кадр
    // (маска против медианы ≈ весь кадр).
    let times = (0..<n).map { start + (end - start) * Double($0) / Double(n) }
    // Многопоточный программный декод: кадры шота тянутся параллельно.
    let frames = try FrameRGBExtractor.loadFrames(url: url, times: times,
                                                  w: w, h: h)
    return frames.filter { !$0.isEmpty }
}
