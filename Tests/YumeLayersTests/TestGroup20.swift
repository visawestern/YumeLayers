// YumeLayers — тесты Фазы 3/5: полный синтетический цикл диффузии
// (кадры → латенты → шум → денисинг → декод → кадр).
import Foundation

func testSyntheticDiffusionCycle() {
    section("Фаза 3: полный цикл диффузии на синтетике")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let vae = ConvVAEGPU(engine: engine, seed: 0x7777_0000_0000_0001)

    // Окно кадров → латенты.
    let frames = (0..<4).map { f -> [Float] in
        var rng = SplitMix64(seed: UInt64(12000 + f))
        return makeSyntheticScene(w: w, h: h, rng: &rng)
    }
    let tensors = frames.map { GPUTensor(engine: engine, data: hwcToChw($0), rows: 3) }
    let latents = encodeWindowLatents(vae: vae, session: session,
                                      frames: tensors, w: w, h: h)
    expectEqual(latents.count, 4, "4 латента")
    let dim = latents[0].count

    // Быстрая докрутка денойзера на 3 латентах.
    let schedule = DiffusionSchedule(timesteps: 20, betaStart: 1e-4, betaEnd: 0.02)
    let t = 4
    var dataset: [(x: [Float], eps: [Float], emb: [Float])] = []
    for (k, z) in latents.prefix(3).enumerated() {
        var rng = SplitMix64(seed: UInt64(15000 + k))
        let eps = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        let xt = forwardDiffuse(x0: z, t: t, schedule: schedule, noise: eps)
        dataset.append((x: xt.xT, eps: eps, emb: timeEmbedding(t: t, dim: 8)))
    }
    let net = DenoiserMLPGPU(engine: engine, dimIn: dim, dimEmb: 8, hidden: 32)
    for step in 1...120 {
        for sample in dataset {
            session.begin()
            let xTg = GPUTensor(engine: engine, data: sample.x, rows: 1)
            let target = GPUTensor(engine: engine, data: sample.eps, rows: 1)
            let (_, cache) = net.forwardCached(session, xT: xTg, emb: sample.emb)
            let grads = net.backward(session, cache, target: target)
            sanitizeGrads(grads, maxNorm: 10.0, preScale: 1.0 / 3.0)
            net.applyAdam(session, grads, t: step, lr: 0.002)
            _ = session.end()
        }
    }

    // Сид-продолжение с денойзером как предиктором.
    let predictor = makeDenoiserPredictor(session: session, net: net, dimEmb: 8)
    let lastZ = latents[3]
    let zNext = continuationFromSeed(lastZ: lastZ, schedule: schedule,
                                     predEps: predictor, seed: 4242)
    expect(zNext.allSatisfy { $0.isFinite }, "сгенерированный латент конечен")

    // Декод латента → кадр.
    guard let frame = decodeLatentArray(engine: engine, vae: vae, z: zNext,
                                        w: w, h: h) else {
        expect(false, "декод латента"); return
    }
    expectEqual(frame.count, w * h * 3, "кадр w×h×3")
    expect(frame.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "кадр в [0,1]")
    let mean = Double(frame.reduce(0) { $0 + Double($1) }) / Double(frame.count)
    let variance = Double(frame.reduce(0) { $0 + Double($1) * Double($1) })
        / Double(frame.count) - mean * mean
    expect(variance > 1e-6, "кадр не вырожден (var=\(variance))")
}
