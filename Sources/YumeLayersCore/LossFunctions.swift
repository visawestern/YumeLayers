// YumeLayers — чистые функции потерь (Этап 1: переключатель L1/Sobel).
//
// MSE-лосс размывает резкие края (аниме-контуры). Для «резкости» до 480p
// добавляем переключатель: L1 вместо MSE и Sobel edge-loss (штраф за расхождение
// краевых карт). Всё чистое и тестируемое (без Metal/UI), чтобы можно было
// подобрать лосс до интеграции в GPU-backward.
import Foundation

/// Вид реконструкционной части потери.
public enum LossKind: Equatable {
    /// Среднеквадратичная ошибка (текущий продуктовый лосс).
    case mse
    /// Средняя абсолютная ошибка (L1) — мягче к выбросам, лучше сохраняет
    /// резкие контуры.
    case l1
}

/// Средняя абсолютная ошибка (L1) между двумя массивами.
public func meanL1(_ a: [Float], _ b: [Float]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return .infinity }
    var sum = 0.0
    for i in 0..<a.count { sum += abs(Double(a[i] - b[i])) }
    return sum / Double(a.count)
}

/// Магнитуда градиента яркости по оператору Собеля (краевая карта).
/// Вход — плоский RGB в формате **HWC** (строка-мажор, 3 канала, значения 0…1).
/// Возвращает массив длиной `w·h` (одна яркость на пиксель).
public func sobelLumaGradient(_ img: [Float], w: Int, h: Int) -> [Float] {
    precondition(img.count >= w * h * 3, "sobelLumaGradient: мал вход")
    func luma(_ i: Int) -> Double {
        0.2126 * Double(img[i]) + 0.7152 * Double(img[i + 1]) + 0.0722 * Double(img[i + 2])
    }
    var out = [Float](repeating: 0, count: w * h)
    // Собель: Gx/Gy, свернутые в 3×3, строка за строкой (dy=-1…1, dx=-1…1).
    let gx: [Double] = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    let gy: [Double] = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    for y in 1..<(h - 1) {
        for x in 1..<(w - 1) {
            var sx = 0.0
            var sy = 0.0
            var k = 0
            for dy in -1...1 {
                for dx in -1...1 {
                    let li = ((y + dy) * w + (x + dx)) * 3
                    sx += gx[k] * luma(li)
                    sy += gy[k] * luma(li)
                    k += 1
                }
            }
            out[y * w + x] = Float(sqrt(sx * sx + sy * sy))
        }
    }
    return out
}

/// Edge-loss: MSE между краевыми картами реконструкции и цели (обе — HWC).
/// Штрафует размытие резких границ.
public func sobelEdgeLoss(recon: [Float], target: [Float], w: Int, h: Int) -> Double {
    let eR = sobelLumaGradient(recon, w: w, h: h)
    let eT = sobelLumaGradient(target, w: w, h: h)
    var sum = 0.0
    for i in 0..<eR.count {
        let d = Double(eR[i] - eT[i])
        sum += d * d
    }
    return sum / Double(eR.count)
}

/// Полная потеря VAE с переключателем вида реконструкционного лоссса:
/// `recLoss + β·KL`. Формат `target` — CHW (как выход энкодера).
public func vaeLossGeneric(cache: VAECache, target: [Float], beta: Float = 1.0,
                           kind: LossKind = .mse) -> Double {
    let r = cache.recon.toArray()
    let n = min(r.count, target.count)
    guard n > 0 else { return .infinity }
    let rec: Double
    switch kind {
    case .mse:
        var s = 0.0
        for i in 0..<n {
            let d = Double(r[i] - target[i])
            s += d * d
        }
        rec = s / Double(n)
    case .l1:
        var s = 0.0
        for i in 0..<n { s += abs(Double(r[i] - target[i])) }
        rec = s / Double(n)
    }
    return rec + Double(beta) * ConvVAEGPU.klMean(cache)
}
