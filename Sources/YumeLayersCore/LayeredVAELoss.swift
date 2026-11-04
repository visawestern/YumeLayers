// YumeLayers — потеря по слоям + потеря композиции для 8-канального VAE.
//
// ЗАЧЕМ. Прежний ConvVAE учился восстанавливать 3 канала RGB: модель «пихала
// всё в один кадр и потом разбиралась». Новая архитектура (docs/CRITIQUE.md,
// итерация 4) факторизует сцену по экспертам: декодер обязан выдать
// 8 каналов ground-truth — фон RGB | объекты RGB | маска | свет — из которых
// кадр собирается ТОЧНОЙ формулой
//
//     rgb = clamp( (фон·(1−m) + объекты·m) · свет , 0, 1 ).
//
// Поэтому потеря состоит из двух частей:
//   1) L_layer  = MSE(8 каналов, таргет)         — каждый эксперт учит свой слой;
//   2) L_compose = MSE(собранный RGB, кадр)      — слои обязаны СОБИРАТЬСЯ.
//
// Второй член — ключевой: без него можно идеально выучить каждый слой по
// отдельности и всё равно получить несовпадающий кадр. Именно он замыкает
// «пол качества» (SceneQuality.swift): качество измеряется по СОБРАННОМУ
// кадру, а не по слоям.
//
// Все массивы здесь в раскладке CHW: канал c, пиксель i → c·n + i, n = w·h.
// Рендерер отдаёт HWC, поэтому `LayeredTarget` делает перекладку один раз
// при подготовке батча, а не на каждом шаге.
//
// Чистый Swift, без Python. CPU-функции — эталон для Metal-ядра
// `layered_grad_kernel`: тест сверяет их поэлементно.
import Foundation
import Metal

// MARK: - Таргет батча

/// 8-канальный ground-truth + собранный кадр, оба в CHW.
public struct LayeredTarget {
    public let w: Int
    public let h: Int
    /// 8·w·h: bgR bgG bgB | objR objG objB | mask | light.
    public let channelsCHW: [Float]
    /// 3·w·h: R, G, B плоскости итогового кадра.
    public let compositeCHW: [Float]
    /// Тот же кадр в HWC — для PSNR/SSIM и сохранения превью.
    public let compositeHWC: [Float]

    public init(w: Int, h: Int, channelsCHW: [Float], compositeCHW: [Float]) {
        precondition(w > 0 && h > 0, "размер кадра должен быть положительным")
        let n = w * h
        precondition(channelsCHW.count == LayeredScene.layerChannels * n,
                     "таргет: ожидалось 8·w·h значений")
        precondition(compositeCHW.count == 3 * n,
                     "композит: ожидалось 3·w·h значений")
        self.w = w
        self.h = h
        self.channelsCHW = channelsCHW
        self.compositeCHW = compositeCHW
        var hwc = [Float](repeating: 0, count: 3 * n)
        for i in 0..<n {
            for c in 0..<3 { hwc[i * 3 + c] = compositeCHW[c * n + i] }
        }
        self.compositeHWC = hwc
    }

    public init(scene: LayeredScene) {
        let n = scene.w * scene.h
        var comp = [Float](repeating: 0, count: 3 * n)
        for i in 0..<n {
            for c in 0..<3 { comp[c * n + i] = scene.composite[i * 3 + c] }
        }
        self.init(w: scene.w, h: scene.h,
                  channelsCHW: scene.channelsCHW(), compositeCHW: comp)
    }

    public var plane: Int { w * h }
}

/// Тот же таргет, но уже в приватной памяти GPU. Создаётся один раз на батч:
/// на каждом шаге гонять 8·w·h float'ов через PCIe было бы дороже ядра.
public final class LayeredTargetGPU {
    public let channels: GPUTensor    // rows = 8
    public let composite: GPUTensor   // rows = 3
    public let plane: Int

    public init(engine: MetalEngine, target: LayeredTarget) {
        self.plane = target.plane
        self.channels = GPUTensor(engine: engine, data: target.channelsCHW,
                                  rows: LayeredScene.layerChannels)
        self.composite = GPUTensor(engine: engine, data: target.compositeCHW, rows: 3)
    }
}

// MARK: - Точная сборка кадра

/// Сборка кадра из 8 предсказанных каналов, CHW → CHW.
///
/// Побитово повторяет `composeLayers` из рендерера (HWC). Инвариант
/// `composeCHW(target.channelsCHW) == target.compositeCHW` проверяется тестом
/// для любого рецепта сцены.
public func composeCHW(w: Int, h: Int, recon: [Float]) -> [Float] {
    let n = w * h
    precondition(recon.count == LayeredScene.layerChannels * n,
                 "composeCHW: ожидалось 8·w·h значений")
    var out = [Float](repeating: 0, count: 3 * n)
    for i in 0..<n {
        let m = recon[6 * n + i]
        let l = recon[7 * n + i]
        let im = 1 - m
        for c in 0..<3 {
            let v = (recon[c * n + i] * im + recon[(3 + c) * n + i] * m) * l
            out[c * n + i] = min(1, max(0, v))
        }
    }
    return out
}

/// Сборка сразу в HWC — для метрик качества и PNG-превью.
public func composeHWC(w: Int, h: Int, recon: [Float]) -> [Float] {
    let chw = composeCHW(w: w, h: h, recon: recon)
    let n = w * h
    var out = [Float](repeating: 0, count: 3 * n)
    for i in 0..<n {
        for c in 0..<3 { out[i * 3 + c] = chw[c * n + i] }
    }
    return out
}

// MARK: - Потеря и её аналитический градиент (CPU-эталон)

/// `layerWeight·MSE(8 каналов) + composeWeight·MSE(собранный RGB)`.
///
/// Оба члена нормируются на СВОЁ число элементов (8n и 3n), иначе при
/// изменении числа каналов вес композиции неявно менялся бы.
public func layeredLoss(recon: [Float], target: LayeredTarget,
                        layerWeight: Float = 1, composeWeight: Float = 1) -> Double {
    let n = target.plane
    guard recon.count == LayeredScene.layerChannels * n else { return .infinity }
    var layer = 0.0
    for i in 0..<recon.count {
        let d = Double(recon[i] - target.channelsCHW[i])
        layer += d * d
    }
    layer /= Double(LayeredScene.layerChannels * n)

    let comp = composeCHW(w: target.w, h: target.h, recon: recon)
    var compose = 0.0
    for i in 0..<comp.count {
        let d = Double(comp[i] - target.compositeCHW[i])
        compose += d * d
    }
    compose /= Double(3 * n)

    return Double(layerWeight) * layer + Double(composeWeight) * compose
}

/// `dL/d(recon)` по всем 8 каналам. Возвращает 8·w·h в CHW.
///
/// Производные композиции (кламп активен только на границе 0/1, в рабочей
/// области [0,1] он тождественен):
///   ∂y/∂bg_c  = l·(1−m)      ∂y/∂obj_c = l·m
///   ∂y/∂m     = l·(obj_c−bg_c)         ∂y/∂l = bg_c·(1−m)+obj_c·m
/// Сумма по c даёт градиенты скалярных каналов маски и света.
public func layeredLossGradient(recon: [Float], target: LayeredTarget,
                                layerWeight: Float = 1,
                                composeWeight: Float = 1) -> [Float] {
    let n = target.plane
    precondition(recon.count == LayeredScene.layerChannels * n,
                 "layeredLossGradient: ожидалось 8·w·h значений")
    var g = [Float](repeating: 0, count: LayeredScene.layerChannels * n)
    let dLayer = 2.0 * Double(layerWeight) / Double(LayeredScene.layerChannels * n)
    let dCompose = 2.0 * Double(composeWeight) / Double(3 * n)

    for i in 0..<n {
        let m = recon[6 * n + i]
        let l = recon[7 * n + i]
        let im = 1 - m
        var dMask = 0.0
        var dLight = 0.0
        for c in 0..<3 {
            let bg = recon[c * n + i]
            let ob = recon[(3 + c) * n + i]
            let base = bg * im + ob * m
            let v = base * l
            let y = min(Float(1), max(Float(0), v))
            let e = Double(y - target.compositeCHW[c * n + i])
            // Производная clamp: вне (0,1) она равна нулю. Внутри — единица.
            let active: Double = (v > 0 && v < 1) ? 1.0 : 0.0
            let gc = dCompose * e * active

            g[c * n + i] = Float(dLayer * Double(bg - target.channelsCHW[c * n + i])
                                 + gc * Double(l * im))
            g[(3 + c) * n + i] = Float(dLayer * Double(ob - target.channelsCHW[(3 + c) * n + i])
                                       + gc * Double(l * m))
            dMask += gc * Double(l) * Double(ob - bg)
            dLight += gc * Double(base)
        }
        g[6 * n + i] = Float(dLayer * Double(m - target.channelsCHW[6 * n + i]) + dMask)
        g[7 * n + i] = Float(dLayer * Double(l - target.channelsCHW[7 * n + i]) + dLight)
    }
    return g
}

/// Полная потеря VAE: layered + β·KL. Должна в точности совпадать с тем, что
/// дифференцирует `backwardLayered`, иначе конечные разности разойдутся.
public func layeredVAELoss(cache: VAECache, target: LayeredTarget,
                           layerWeight: Float = 1, composeWeight: Float = 1,
                           beta: Float = 1) -> Double {
    let recon = cache.recon.toArray()
    return layeredLoss(recon: recon, target: target,
                       layerWeight: layerWeight, composeWeight: composeWeight)
         + Double(beta) * ConvVAEGPU.klMean(cache)
}

// MARK: - Пол качества по СОБРАННОМУ кадру

/// PSNR/SSIM собранного кадра против входа. «Не хуже входа» проверяется
/// здесь, а не по отдельным слоям: пользователю важен кадр, а не разметка.
public func composedQuality(recon: [Float], target: LayeredTarget)
    -> (psnr: Float, ssim: Float) {
    let rgb = composeHWC(w: target.w, h: target.h, recon: recon)
    return (psnr(reference: target.compositeHWC, candidate: rgb),
            luminanceSSIM(reference: target.compositeHWC, candidate: rgb))
}

// MARK: - Бюджет видеопамяти layered-пути

/// Дополнительные буферы на кадр поверх `convVAEBytesPerSample`:
/// таргет 8n + градиент 8n + композит 3n. Без них расчёт пачки завышал бы
/// доступный batch и повторял историю с OOM на 8 ГБ (docs/AUDIT.md §C2).
public func layeredActivationBytes(w: Int, h: Int) -> Int {
    (2 * LayeredScene.layerChannels + 3) * w * h * 4
}
