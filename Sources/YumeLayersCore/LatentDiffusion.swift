// YumeLayers — Фаза 3: диффузия в латентном пространстве VAE.
//
// Чистая математика DDPM (линейное расписание шума) и датасет временных окон
// из манифеста клипов. Здесь нет GPU: обучение денойзинга поверх этих функций
// идёт отдельно (Metal-ядра поверх `GpuSession`).
import Foundation

/// Линейное расписание шума DDPM:
/// betas линейно от betaStart до betaEnd, alphas = 1 − betas,
/// alphaBars = cumprod(alphas). Индекс t ∈ [0, timesteps).
public struct DiffusionSchedule: Equatable, Sendable {
    public let timesteps: Int
    public let betas: [Double]
    public let alphas: [Double]
    public let alphaBars: [Double]

    public init(timesteps: Int = 1000,
                betaStart: Double = 1e-4,
                betaEnd: Double = 0.02) {
        precondition(timesteps > 0)
        self.timesteps = timesteps
        var b = [Double]()
        b.reserveCapacity(timesteps)
        for i in 0..<timesteps {
            let f = Double(i) / Double(max(1, timesteps - 1))
            b.append(betaStart + f * (betaEnd - betaStart))
        }
        var a = [Double]()
        var ab = [Double]()
        var cum: Double = 1
        for beta in b {
            let alpha = 1 - beta
            a.append(alpha)
            cum *= alpha
            ab.append(cum)
        }
        self.betas = b
        self.alphas = a
        self.alphaBars = ab
    }
}

/// Результат прямого зашумления: x_t = √(ᾱ_t)·x0 + √(1−ᾱ_t)·ε.
public struct DiffusedLatent: Equatable, Sendable {
    public let xT: [Float]
    public let eps: [Float]
    public let sqrtAlphaBar: Double
    public let sqrtOneMinus: Double
}

/// Прямое зашумление x0 → x_t по расписанию.
/// - Parameters:
///   - t: шаг шума (0 … timesteps−1).
///   - noise: тот же размер, что и x0 (обычно N(0,1)).
public func forwardDiffuse(x0: [Float], t: Int,
                           schedule: DiffusionSchedule,
                           noise: [Float]) -> DiffusedLatent {
    precondition(x0.count == noise.count, "x0 и noise одного размера")
    precondition(t >= 0 && t < schedule.timesteps, "t вне [0, timesteps)")
    let alphaBar = schedule.alphaBars[t]
    let sqrtAB = sqrt(max(alphaBar, 0))
    let sqrtOne = sqrt(max(1 - alphaBar, 0))
    var out = [Float](repeating: 0, count: x0.count)
    for i in 0..<x0.count {
        out[i] = Float(sqrtAB) * x0[i] + Float(sqrtOne) * noise[i]
    }
    return DiffusedLatent(xT: out, eps: noise,
                          sqrtAlphaBar: sqrtAB, sqrtOneMinus: sqrtOne)
}

/// Разворачивает клипы движения в плоский список ссылок на кадры
/// (порядок: клип за клипом, кадр за кадром).
public func flattenWindowRefs(_ windows: [MotionClip]) -> [ExtractedFrameRef] {
    var refs: [ExtractedFrameRef] = []
    for w in windows {
        for idx in w.frameIndices {
            refs.append(ExtractedFrameRef(source: w.source, index: idx))
        }
    }
    return refs
}

/// Делит плоский список ссылок кадров на батчи по `framesPerBatch`.
/// Порядок окон не нарушается; последний батч может быть меньше.
public func latentBatchRefs(windows: [MotionClip],
                            framesPerBatch: Int) -> [[ExtractedFrameRef]] {
    let batch = max(1, framesPerBatch)
    let refs = flattenWindowRefs(windows)
    guard !refs.isEmpty else { return [] }
    var out: [[ExtractedFrameRef]] = []
    var i = 0
    while i < refs.count {
        let end = min(i + batch, refs.count)
        out.append(Array(refs[i..<end]))
        i = end
    }
    return out
}

/// Детерминированная выборка не более `maxWindows` окон из списка клипов.
/// Использует `SplitMix64` как единственный источник случайности.
func subsampleWindows(_ clips: [MotionClip], maxWindows: Int,
                      rng: inout SplitMix64) -> [MotionClip] {
    guard maxWindows > 0, maxWindows < clips.count else { return clips }
    var pool = clips
    var chosen: [MotionClip] = []
    while chosen.count < maxWindows, !pool.isEmpty {
        let idx = Int.random(in: 0..<pool.count, using: &rng)
        chosen.append(pool.remove(at: idx))
    }
    return chosen
}
