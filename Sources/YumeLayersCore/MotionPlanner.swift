// YumeLayers — Фаза 4: планирование движения между опорными кадрами.
//
// Чистые функции (без Metal/UI): выбор «каких кадров» из сегмента реального
// видео и построение обучающих пар для сети движения. Покрыты юнит-тестами.
import Foundation

/// Равномерная выборка `targetCount` индексов из последовательности
/// `0..<total` с сохранением обоих концов (0 и total−1 всегда включены).
///
/// Нужна для случая «между опорными кадрами 10 секунд, а запрошено 5»:
/// вместо выбрасывания кадров подряд мы берём равномерную пропорциональную
/// выборку — основные переходы движения сохраняются, сцена не «рвётся».
///
/// - Если `targetCount >= total` — возвращаются все индексы (растягивать
///   нечего; недостающая длина добивается интерполяцией модели).
/// - Если `targetCount < 2` — возвращается [0].
public func resampleSegmentIndices(total: Int, targetCount: Int) -> [Int] {
    guard total > 0 else { return [] }
    guard targetCount >= 2, total >= 2 else { return [0] }
    guard targetCount < total else { return Array(0..<total) }
    let last = Double(total - 1)
    let denom = Double(targetCount - 1)
    var picked: [Int] = []
    picked.reserveCapacity(targetCount)
    for i in 0..<targetCount {
        picked.append(Int((Double(i) * last / denom).rounded()))
    }
    // Редкая коллизия округления даёт дубликаты — заполняем недостающие
    // позиции средними между соседями с наибольшим зазором.
    var uniq = picked
    while uniq.count < targetCount {
        var bestGap = 0
        var bestIdx = -1
        for i in 0..<(uniq.count - 1) {
            let gap = uniq[i + 1] - uniq[i]
            if gap > bestGap {
                bestGap = gap
                bestIdx = i
            }
        }
        guard bestIdx >= 0, bestGap > 1 else { break }
        uniq.insert((uniq[bestIdx] + uniq[bestIdx + 1]) / 2, at: bestIdx + 1)
    }
    return uniq
}

/// Обучающая пара сети движения: внутри окна [a, b] целевой кадр k.
public struct MotionPair: Equatable {
    /// Индекс кадра A (начало окна).
    public let a: Int
    /// Индекс кадра B (конец окна).
    public let b: Int
    /// Индекс целевого промежуточного кадра (a < k < b).
    public let k: Int

    public init(a: Int, b: Int, k: Int) {
        self.a = a
        self.b = b
        self.k = k
    }

    /// Нормированное время целевого кадра в окне: 0…1.
    public var t: Double {
        guard b > a else { return 0 }
        return Double(k - a) / Double(b - a)
    }
}

/// Все обучающие пары внутри одной непрерывной серии кадров одного источника.
///
/// Окно скользит с шагом `stride` (по умолчанию 1 — полное перекрытие),
/// внутри каждого окна целевыми являются ВСЕ внутренние кадры.
public func motionPairs(frameCount: Int, window: Int, stride: Int = 1) -> [MotionPair] {
    let w = max(3, window)
    let s = max(1, stride)
    guard frameCount >= w else { return [] }
    var pairs: [MotionPair] = []
    var start = 0
    while start + w <= frameCount {
        let a = start
        let b = start + w - 1
        guard b - a >= 2 else { start += s; continue }
        for k in (a + 1)..<b {
            pairs.append(MotionPair(a: a, b: b, k: k))
        }
        start += s
    }
    return pairs
}

/// Синусоидальное временное эмбеддинг для t ∈ [0, 1]:
/// emb[2i] = sin(t·1000^(i/d)), emb[2i+1] = cos(t·1000^(i/d)).
public func motionTimeEmbedding(_ t: Double, dim: Int) -> [Float] {
    precondition(dim > 0 && dim % 2 == 0, "dim чётный и > 0")
    var out = [Float](repeating: 0, count: dim)
    let tt = min(max(t.isFinite ? t : 0, 0), 1)
    for i in stride(from: 0, to: dim, by: 2) {
        let w = pow(1000.0, Double(i) / Double(dim))
        out[i] = Float(sin(tt * w))
        out[i + 1] = Float(cos(tt * w))
    }
    return out
}
