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

/// Один шаг обратного сэмплирования (DDIM, каноническая форма):
///   x̂0 = (x_t − √(1−ᾱ_t)·εθ)/√ᾱ_t
///   x_{t-1} = √ᾱ_{t-1}·x̂0 + √(1−ᾱ_{t-1} − σ_t²)·εθ + σ_t·z,
/// где σ_t = η·√((1−ᾱ_{t-1})/(1−ᾱ_t))·√(1−ᾱ_t/ᾱ_{t-1}); η=0 — детерминизм.
public func ddpmDenoiseStep(xT: [Float], t: Int, schedule: DiffusionSchedule,
                            predEps: [Float], eta: Double = 0,
                            z: [Float]? = nil) -> [Float] {
    precondition(xT.count == predEps.count, "размеры совпадают")
    precondition(t >= 0 && t < schedule.timesteps, "t вне [0, timesteps)")
    let abar = schedule.alphaBars[t]
    let abarPrev = t > 0 ? schedule.alphaBars[t - 1] : 1.0
    let sqrtAbar = sqrt(max(abar, 0))
    let sqrtOneAbar = sqrt(max(1 - abar, 0))
    let sqrtAbarPrev = sqrt(max(abarPrev, 0))
    let sigma = eta * sqrt(max((1 - abarPrev) / max(1 - abar, 1e-12), 0))
        * sqrt(max(1 - abar / max(abarPrev, 1e-12), 0))
    let sqrtOneAbarPrevMinusSigma = sqrt(max(1 - abarPrev - sigma * sigma, 0))
    var out = [Float](repeating: 0, count: xT.count)
    let zz = z ?? [Float](repeating: 0, count: xT.count)
    for i in 0..<xT.count {
        let x0hat = (Double(xT[i]) - sqrtOneAbar * Double(predEps[i])) / sqrtAbar
        let base = sqrtAbarPrev * x0hat + sqrtOneAbarPrevMinusSigma * Double(predEps[i])
        out[i] = Float(base + sigma * Double(zz[i]))
    }
    return out
}

/// Полная цепочка денисинга: x_T → x_0 (η=0 — детерминированный DDIM).
public func ddpmSampleChain(xT: [Float], schedule: DiffusionSchedule,
                            predEps: (Int, [Float]) -> [Float],
                            eta: Double = 0) -> [Float] {
    var x = xT
    var t = schedule.timesteps - 1
    while t >= 0 {
        let eps = predEps(t, x)
        x = ddpmDenoiseStep(xT: x, t: t, schedule: schedule,
                            predEps: eps, eta: eta)
        t -= 1
    }
    return x
}

/// Кодирует окно кадров в латенты VAE и возвращает плоские массивы.
/// Чтения GPU→CPU выполняются ТОЛЬКО после `session.end()`.
public func encodeWindowLatents(vae: ConvVAEGPU, session: GpuSession,
                                frames: [GPUTensor], w: Int, h: Int,
                                deterministic: Bool = true,
                                seed: UInt32 = 0) -> [[Float]] {
    guard !frames.isEmpty else { return [] }
    session.begin()
    var zTensors: [GPUTensor] = []
    for f in frames {
        let e = vae.encode(session, image: f, w: w, h: h,
                           deterministic: deterministic, seed: seed)
        zTensors.append(e.z)
    }
    _ = session.end()
    return zTensors.map { $0.toArray() }
}

/// Детерминированный «шум» из seed (SplitMix64: Float.random в [-1, 1]).
public func noiseFromSeed(seed: UInt64, count: Int) -> [Float] {
    var rng = SplitMix64(seed: seed)
    return (0..<count).map { _ in Float.random(in: -1...1, using: &rng) }
}

/// Сид-продолжение: из латента последнего кадра z_last генерирует следующий
/// латент z' через DDIM-цепочку (зашумление по seed → денисинг).
/// Один и тот же seed → детерминированный результат.
public func continuationFromSeed(lastZ: [Float], schedule: DiffusionSchedule,
                                 predEps: (Int, [Float]) -> [Float],
                                 seed: UInt64) -> [Float] {
    let noise = noiseFromSeed(seed: seed, count: lastZ.count)
    let xT = forwardDiffuse(x0: lastZ, t: schedule.timesteps - 1,
                            schedule: schedule, noise: noise)
    return ddpmSampleChain(xT: xT.xT, schedule: schedule, predEps: predEps)
}

/// Декод плоского массива латента (C×plane) в кадр [0,1] (sigmoid).
/// nil при несовпадении формы. Чтения — после end().
public func decodeLatentArray(engine: MetalEngine, vae: ConvVAEGPU,
                              z: [Float], w: Int, h: Int) -> [Float]? {
    let C = vae.latentChannels
    let plane = (w / 8) * (h / 8)
    guard z.count == C * plane else { return nil }
    let session = GpuSession(engine)
    session.begin()
    let zT = GPUTensor(engine: engine, data: z, rows: C)
    let d = vae.decode(session, z: zT, w: w, h: h)
    _ = session.end()
    return d.recon.toArray()
}

/// Предиктор шума денойзера для CPU-цепочки: каждая итерация — один GPU-форвард.
/// (net, session) захватываются; сессия используется последовательно.
func makeDenoiserPredictor(session: GpuSession, net: DenoiserMLPGPU,
                           dimEmb: Int) -> (Int, [Float]) -> [Float] {
    { t, x in
        session.begin()
        let xT = GPUTensor(engine: session.engine, data: x, rows: 1)
        let emb = timeEmbedding(t: t, dim: dimEmb)
        let pred = net.forward(session, xT: xT, emb: emb)
        _ = session.end()
        return pred.toArray()
    }
}
