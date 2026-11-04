// YumeLayers — гарантия качества и покадровое планирование слоёв-экспертов.
//
// 1) КАЧЕСТВО НЕ ХУЖЕ ВХОДА. У модели есть «пол»: реконструкция кадра, который
//    мы ей скормили, обязана быть не хуже порога (PSNR/SSIM). Чекпоинт, который
//    ухудшил реконструкцию, не принимается. Генерация идёт ПОВЕРХ этого пола:
//    диффузия предсказывает остаток к реконструкции, а не «рисует с нуля», так
//    что выход не может быть хуже качества, которое модель уже доказала на входе.
//
// 2) УСКОРЕНИЕ ДВИЖЕНИЯ. На последовательности кадров статичные эксперты
//    (небо, земля, свет) пересчитываются только на ключевых кадрах, а движущиеся
//    (фигуры, тени) — покадрово. Это прямое следствие факторизации по слоям:
//    гигантская модель не гоняется целиком на каждом кадре.
import Foundation

// ---------- Метрики ----------

/// PSNR в дБ по RGB HWC [0…1]. Идентичные массивы → +infinity.
public func psnr(reference: [Float], candidate: [Float]) -> Float {
    guard reference.count == candidate.count, !reference.isEmpty else { return 0 }
    var sum: Double = 0
    for i in 0..<reference.count {
        let d = Double(reference[i]) - Double(candidate[i])
        sum += d * d
    }
    let mse = sum / Double(reference.count)
    if mse <= 0 { return .infinity }
    return Float(-10 * log10(mse))
}

/// Глобальный SSIM по яркости (упрощённая, но монотонная метрика сходства
/// структуры). Идентичные кадры → 1.
public func luminanceSSIM(reference: [Float], candidate: [Float]) -> Float {
    guard reference.count == candidate.count, reference.count >= 3 else { return 0 }
    let n = reference.count / 3
    var a = [Double](repeating: 0, count: n)
    var b = [Double](repeating: 0, count: n)
    for i in 0..<n {
        a[i] = 0.2126 * Double(reference[i * 3]) + 0.7152 * Double(reference[i * 3 + 1])
             + 0.0722 * Double(reference[i * 3 + 2])
        b[i] = 0.2126 * Double(candidate[i * 3]) + 0.7152 * Double(candidate[i * 3 + 1])
             + 0.0722 * Double(candidate[i * 3 + 2])
    }
    func mean(_ x: [Double]) -> Double { x.reduce(0, +) / Double(x.count) }
    let ma = mean(a), mb = mean(b)
    var va: Double = 0, vb: Double = 0, cov: Double = 0
    for i in 0..<n {
        let da = a[i] - ma, db = b[i] - mb
        va += da * da; vb += db * db; cov += da * db
    }
    va /= Double(n); vb /= Double(n); cov /= Double(n)
    let c1 = 0.01 * 0.01, c2 = 0.03 * 0.03
    let num = (2 * ma * mb + c1) * (2 * cov + c2)
    let den = (ma * ma + mb * mb + c1) * (va + vb + c2)
    guard den > 0 else { return 1 }
    return Float(min(1, num / den))
}

// ---------- Ворот качества ----------

/// Чекпоинт принимается, только если реконструкция держит «пол качества».
public struct QualityFloor {
    public let minPSNR: Float
    public let minSSIM: Float

    /// Порог по умолчанию: 30 дБ + SSIM 0.90. Сравнивать всегда надо и с
    /// билинейным бейзлайном (аудит §6.3): «не хуже входа» = не хуже тривиальной
    /// реконструкции + запас.
    public init(minPSNR: Float = 30, minSSIM: Float = 0.90) {
        self.minPSNR = minPSNR
        self.minSSIM = minSSIM
    }

    public func accepts(reference: [Float], reconstruction: [Float]) -> Bool {
        let p = psnr(reference: reference, candidate: reconstruction)
        let s = luminanceSSIM(reference: reference, candidate: reconstruction)
        return p >= minPSNR && s >= minSSIM
    }

    /// Отклоняет чекпоинт, если он стал хуже ПРЕДЫДУЩЕГО принятого.
    public func accepts(current: Float, previous: Float, toleranceDB: Float = 0.5) -> Bool {
        current >= minPSNR && current >= previous - toleranceDB
    }

    public func report(reference: [Float], reconstruction: [Float]) -> String {
        String(format: "PSNR %.2f дБ · SSIM %.4f · порог %.1f дБ / %.2f → %@",
               psnr(reference: reference, candidate: reconstruction),
               luminanceSSIM(reference: reference, candidate: reconstruction),
               minPSNR, minSSIM,
               accepts(reference: reference, reconstruction: reconstruction) ? "принят" : "отклонён")
    }
}

// ---------- Покадровый план ----------

/// Планировщик пересчёта экспертов на клипе. Статичные слои — на ключевых
/// кадрах, движущиеся — на каждом. Смена сцены (motionScore) сбрасывает кэш.
public struct FramePlan {
    public let keyframeInterval: Int

    public init(keyframeInterval: Int = 8) {
        self.keyframeInterval = max(1, keyframeInterval)
    }

    public func needsRecompute(frame: Int, kind: LayerKind, motionScore: Float = 0) -> Bool {
        if motionScore > 0.5 { return true }          // склейка/смена плана → всё заново
        switch kind {
        case .figure, .shadow:
            return true                               // объекты движутся покадрово
        case .clouds:
            return frame % max(1, keyframeInterval / 2) == 0
        case .sky, .ground, .sun, .lightGradient:
            return frame % keyframeInterval == 0
        }
    }

    /// Оценка ускорения: во сколько раз меньше экспертных прогонов на кадр.
    /// Считаем стоимость каждого слоя равной 1 (потом заменить на реальные FLOPs).
    public func estimatedSpeedup(layers: [LayerSpec]) -> Float {
        guard !layers.isEmpty else { return 1 }
        var cost: Float = 0
        for spec in layers {
            let perFrame: Float
            switch spec.kind {
            case .figure, .shadow: perFrame = 1
            case .clouds: perFrame = 1 / Float(max(1, keyframeInterval / 2))
            case .sky, .ground, .sun, .lightGradient: perFrame = 1 / Float(keyframeInterval)
            }
            cost += perFrame * Float(spec.count)
        }
        let full = Float(layers.reduce(0) { $0 + $1.count })
        return full / max(cost, 1e-6)
    }
}
