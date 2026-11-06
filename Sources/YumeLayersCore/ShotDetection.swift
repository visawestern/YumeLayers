import Foundation
import AVFoundation
import CoreMedia

/// Детектор склеек (смены сцен) в фильме.
///
/// Проблема прямой выборки «N кадров по всему фильму»: фон в `LayerAnnotator`
/// считается ОДНОЙ временной медианой по стеку. Если надергать кадры из разных
/// сцен, медианный фон становится бессмысленной кашей и маска раздувается до
/// 80% кадра. Поэтому перед аннотацией нужно разбить фильм на однородные
/// «шоты» (где задник стабилен), и внутри каждого шота уже аннотировать плотно.
///
/// Сканируем фильм грубо (`scanCount` моментов), считаем среднюю яркость кадра
/// и ищем скачки от соседних — это и есть склейки. Порог относительный
/// (×6 от медианы), чтобы динамичные сцены не дробились на куски.
public func detectSceneShots(url: URL, w: Int, h: Int,
                             scanCount: Int = 240,
                             cutThreshold: Double = 0.18) throws -> [(start: Double, end: Double)] {
    let asset = AVURLAsset(url: url)
    guard let time = try? loadAssetDuration(asset) else {
        throw LayerDatasetError.unreadableVideo(url.path)
    }
    let duration = CMTimeGetSeconds(time)
    guard duration.isFinite, duration > 0.5 else {
        throw LayerDatasetError.unreadableVideo(url.path)
    }
    let n = max(scanCount, 8)
    let times = (0..<n).map { duration * Double($0) / Double(n - 1) }
    // Многопоточный программный декод всех точек сканирования сразу.
    let frames = try FrameRGBExtractor.loadFrames(url: url, times: times,
                                                  w: w, h: h)

    var lumas: [Float] = []
    lumas.reserveCapacity(n)
    let px = w * h
    for rgb in frames {
        guard !rgb.isEmpty else { lumas.append(0); continue }
        var s: Float = 0
        for p in 0..<px { s += lumaAt(rgb, p) }
        lumas.append(s / Float(px))
    }
    var diffs: [Double] = []
    diffs.reserveCapacity(n)
    for i in 1..<lumas.count { diffs.append(Double(abs(lumas[i] - lumas[i - 1]))) }
    let sorted = diffs.sorted()
    let median = sorted[sorted.count / 2]
    let thr = max(median * 6.0, cutThreshold)

    var cuts: [Int] = [0]
    for i in 1..<diffs.count where diffs[i] > thr {
        if i - cuts.last! >= 2 { cuts.append(i) }
    }
    cuts.append(n - 1)

    var shots: [(Double, Double)] = []
    for k in 0..<(cuts.count - 1) {
        let a = duration * Double(cuts[k]) / Double(n - 1)
        let b = duration * Double(cuts[k + 1]) / Double(n - 1)
        if b - a > 0.4 { shots.append((a, b)) }
    }
    return shots
}
