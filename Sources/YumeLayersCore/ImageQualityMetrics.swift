// YumeLayers — Фаза 7: приближённый FID (диагональная ковариация).
// Полный FID требует инцепции и собственных значений; здесь — честная
// диагональная аппроксимация для сравнения распределений фич.
import Foundation

/// Статистика выборки фич: среднее и диагональная дисперсия.
/// nil при пустой выборке или неодинаковых длинах.
public func featureStats(samples: [[Float]]) -> (mean: [Double], diagVar: [Double])? {
    guard let first = samples.first, !first.isEmpty else { return nil }
    let d = first.count
    guard samples.allSatisfy({ $0.count == d }) else { return nil }
    var mean = [Double](repeating: 0, count: d)
    for s in samples {
        for i in 0..<d {
            mean[i] += Double(s[i])
        }
    }
    for i in 0..<d { mean[i] /= Double(samples.count) }
    var variance = [Double](repeating: 0, count: d)
    for s in samples {
        for i in 0..<d {
            let diff = Double(s[i]) - mean[i]
            variance[i] += diff * diff
        }
    }
    for i in 0..<d { variance[i] /= Double(samples.count) }
    return (mean, variance)
}

/// Fréchet-расстояние между гауссианами с диагональными ковариациями:
///   d² = Σ_i (m1_i − m2_i)² + Σ_i (√v1_i − √v2_i)²
public func frechetDistanceDiagonal(mean1: [Double], var1: [Double],
                                   mean2: [Double], var2: [Double]) -> Double {
    precondition(mean1.count == var1.count && mean2.count == var2.count
                 && mean1.count == mean2.count, "размерности совпадают")
    var sum = 0.0
    for i in 0..<mean1.count {
        let dm = mean1[i] - mean2[i]
        let ds = sqrt(max(var1[i], 0)) - sqrt(max(var2[i], 0))
        sum += dm * dm + ds * ds
    }
    return sum
}
