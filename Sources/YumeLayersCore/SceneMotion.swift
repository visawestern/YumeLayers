// YumeLayers — профиль движения и «законченности» сцены по последовательным
// кадрам (Фаза 3: фундамент временных окон движения).
//
// Сцена считается «законченной», когда по стеку кадров мы видели движение на
// большей части кадра (traveledCoverage) при стабильном фоне. Это даёт сигнал
// для диффузии движения: на таких окнах есть и статика (фон), и динамика.

import Foundation

/// Сводный профиль движения по последовательности кадров.
public struct MotionSummary {
    /// Доля пикселей, стабильных по всему стеку (фон).
    public let staticFraction: Double
    /// Максимальная доля движущихся пикселей в одном кадре.
    public let peakMotion: Double
    /// Доля кадра, где хоть раз наблюдалось движение (полнота покрытия).
    public let traveledCoverage: Double
    /// Вывод о законченности сцены.
    public let sceneComplete: Bool
}

/// Считает профиль движения по стеку кадров (RGB HWC, значения 0…1).
public func motionSummary(frames: [[Float]], w: Int, h: Int) -> MotionSummary {
    let px = w * h
    guard !frames.isEmpty else {
        return MotionSummary(staticFraction: 1, peakMotion: 0,
                             traveledCoverage: 0, sceneComplete: false)
    }
    let thr: Float = 0.06

    // Медианный фон по каждому пикселю (яркость усреднена по каналам).
    let n = frames.count
    var bg = [Float](repeating: 0, count: px)
    var pixelStack = [[Float]](repeating: [], count: px)
    for img in frames {
        guard img.count == px * 3 else { continue }
        for p in 0..<px {
            let lum = (img[p * 3] + img[p * 3 + 1] + img[p * 3 + 2]) / 3
            pixelStack[p].append(lum)
        }
    }
    var seen = [Bool](repeating: false, count: px)
    var peak = 0.0
    var changedSum = 0.0
    for p in 0..<px {
        let st = pixelStack[p].sorted()
        bg[p] = st.isEmpty ? 0 : st[st.count / 2]
        for lum in st where abs(lum - bg[p]) > thr {
            seen[p] = true
        }
    }
    var totalChanged = 0
    for img in frames {
        guard img.count == px * 3 else { continue }
        var changed = 0
        for p in 0..<px {
            let lum = (img[p * 3] + img[p * 3 + 1] + img[p * 3 + 2]) / 3
            if abs(lum - bg[p]) > thr { changed += 1 }
        }
        let f = Double(changed) / Double(px)
        if f > peak { peak = f }
        changedSum += f
    }
    for p in 0..<px where seen[p] { totalChanged += 1 }
    let traveled = Double(totalChanged) / Double(px)
    let staticFraction = 1 - changedSum / Double(max(n, 1))
    // Сцена закончена: видели движение почти по всему кадру при стабильном фоне.
    let sceneComplete = traveled >= 0.9 && staticFraction >= 0.5
    return MotionSummary(staticFraction: staticFraction, peakMotion: peak,
                         traveledCoverage: traveled, sceneComplete: sceneComplete)
}
