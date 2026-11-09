// YumeLayers — тесты Фазы 3: обратное сэмплирование и интеграция
// денойзера с латентами VAE.
import Foundation

func testDDIMDenoiseStep() {
    section("Фаза 3: DDIM-шаг денисинга (формула)")
    let schedule = DiffusionSchedule(timesteps: 20, betaStart: 1e-4, betaEnd: 0.02)
    var rng = SplitMix64(seed: 77)
    let x0 = (0..<8).map { _ in Float.random(in: -1...1, using: &rng) }
    let noise = (0..<8).map { _ in Float.random(in: -1...1, using: &rng) }
    let t = 10
    let xt = forwardDiffuse(x0: x0, t: t, schedule: schedule, noise: noise)
    let xPrev = ddpmDenoiseStep(xT: xt.xT, t: t, schedule: schedule,
                                predEps: noise)

    // Алгебраическое тождество DDIM-шага:
    // x̂0 = (x_t − √(1−ᾱ_t)·ε)/√ᾱ_t;  x_{t-1} = √ᾱ_{t-1}·x̂0 + √(1−ᾱ_{t-1})·ε.
    let abar = schedule.alphaBars[t]
    let abarPrev = schedule.alphaBars[t - 1]
    let sqrtAbar = sqrt(max(abar, 0))
    let sqrtOneAbar = sqrt(max(1 - abar, 0))
    let sqrtAbarPrev = sqrt(max(abarPrev, 0))
    let sqrtOneAbarPrev = sqrt(max(1 - abarPrev, 0))
    for i in 0..<8 {
        let x0hat = (Double(xt.xT[i]) - sqrtOneAbar * Double(noise[i])) / sqrtAbar
        let expected = sqrtAbarPrev * x0hat + sqrtOneAbarPrev * Double(noise[i])
        expectClose(Double(xPrev[i]), expected, 1e-3,
                    "x_{t-1} удовлетворяет тождеству DDIM")
    }
    // Устойчивость: при pred=0 шаг остаётся конечным.
    let zero = ddpmDenoiseStep(xT: xt.xT, t: t, schedule: schedule, predEps: [Float](repeating: 0, count: 8))
    expect(zero.allSatisfy { $0.isFinite }, "pred=0: конечные значения")
}

func testDDIMChainPerfectPredictor() {
    section("Фаза 3: цепочка DDIM — свойства (T=1 восстановление, детерминизм)")
    // T=1: цепочка из одного шага с точным шумом восстанавливает x0
    // (допуск ~ примеси на t=0, как в forwardDiffuse).
    let schedule1 = DiffusionSchedule(timesteps: 1, betaStart: 1e-4, betaEnd: 1e-4)
    var rng = SplitMix64(seed: 5)
    let x0 = (0..<6).map { _ in Float.random(in: -1...1, using: &rng) }
    let noise = (0..<6).map { _ in Float.random(in: -1...1, using: &rng) }
    let xT1 = forwardDiffuse(x0: x0, t: 0, schedule: schedule1, noise: noise)
    let xHat1 = ddpmSampleChain(xT: xT1.xT, schedule: schedule1,
                                predEps: { _, _ in noise })
    for i in 0..<6 {
        expectClose(Double(xHat1[i]), Double(x0[i]), 0.02,
                    "T=1 с точным ε: x0 восстановлен")
    }

    // Полная цепочка (T=16): детерминизм и конечность при произвольном предикторе.
    let schedule = DiffusionSchedule(timesteps: 16, betaStart: 1e-4, betaEnd: 0.02)
    let xT = forwardDiffuse(x0: x0, t: 15, schedule: schedule, noise: noise)
    let pred: (Int, [Float]) -> [Float] = { _, x in
        x.map { $0 * 0.5 }
    }
    let xHat = ddpmSampleChain(xT: xT.xT, schedule: schedule, predEps: pred)
    expect(xHat.allSatisfy { $0.isFinite }, "цепочка: конечные значения")
    let again = ddpmSampleChain(xT: xT.xT, schedule: schedule, predEps: pred)
    expectEqual(again, xHat, "детерминизм цепочки")
}

func testDenoiserOnVAELatents() {
    section("Фаза 3: обучение денойзера на латентах VAE (окно кадров)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let vae = ConvVAEGPU(engine: engine, seed: 0xABCD_0000_1111_2222)
    let nFrames = 4
    var frames: [[Float]] = []
    for f in 0..<nFrames {
        var rng = SplitMix64(seed: UInt64(5000 + f))
        frames.append(makeSyntheticScene(w: w, h: h, rng: &rng))
    }
    let xTensors = frames.map { GPUTensor(engine: engine, data: hwcToChw($0), rows: 3) }
    let latents = encodeWindowLatents(vae: vae, session: session,
                                      frames: xTensors, w: w, h: h)
    expectEqual(latents.count, nFrames, "латент на кадр")
    let dim = latents[0].count
    expect(dim > 0, "латент непустой")

    let schedule = DiffusionSchedule(timesteps: 20, betaStart: 1e-4, betaEnd: 0.02)
    let t = 5
    var dataset: [(x: [Float], eps: [Float], emb: [Float])] = []
    for (k, z) in latents.enumerated() {
        var rng = SplitMix64(seed: UInt64(9000 + k))
        let eps = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        let xt = forwardDiffuse(x0: z, t: t, schedule: schedule, noise: eps)
        dataset.append((x: xt.xT, eps: eps, emb: timeEmbedding(t: t, dim: 8)))
    }
    let net = DenoiserMLPGPU(engine: engine, dimIn: dim, dimEmb: 8, hidden: 32)
    var firstLoss = Double.infinity
    var lastLoss = Double.infinity
    for step in 1...400 {
        var sumLoss = 0.0
        for sample in dataset {
            session.begin()
            let xTg = GPUTensor(engine: engine, data: sample.x, rows: 1)
            let target = GPUTensor(engine: engine, data: sample.eps, rows: 1)
            let (pred, cache) = net.forwardCached(session, xT: xTg, emb: sample.emb)
            let grads = net.backward(session, cache, target: target)
            sanitizeGrads(grads, maxNorm: 10.0, preScale: 1.0 / Float(dataset.count))
            net.applyAdam(session, grads, t: step, lr: 0.001)
            _ = session.end()
            let pa = pred.toArray()
            var s = 0.0
            for i in 0..<dim {
                let d = Double(pa[i] - sample.eps[i])
                s += d * d
            }
            sumLoss += s / Double(dim)
        }
        let loss = sumLoss / Double(dataset.count)
        if step == 1 { firstLoss = loss }
        if step == 400 { lastLoss = loss }
    }
    expect(firstLoss.isFinite && lastLoss.isFinite, "loss конечна")
    expect(lastLoss < firstLoss * 0.7,
           "денойзер обучается на латентах VAE (\(String(format: "%.4f", firstLoss))→\(String(format: "%.4f", lastLoss)))")
}
