// YumeLayers — тесты архитектуры v3 (GroupNorm): обучение и градиенты.
// Выделены из TestGroup11 для соблюдения лимита 600 строк/файл.
import Foundation

func testV3GroupNormNoResGrads() {
    section("Архитектура v3(no-res): GroupNorm-градиенты (сильные сигналы)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let w = 32, h = 24
    var rng = SplitMix64(seed: 11)
    let img = makeSyntheticScene(w: w, h: h, rng: &rng)
    let x = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)

    // Отдельная численная проверка v3 БЕЗ residual: только GroupNorm-интеграция.
    let v3n = ConvVAEGPU(engine: engine, groupNorm: true)
    func fwdN(_ ws: [Matrix]) -> Double {
        let v = ConvVAEGPU(engine: engine, groupNorm: true, seed: 0x5EED_1234_ABCD_0001)
        try? v.load(from: ws)
        let x2 = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
        let s2 = GpuSession(engine)
        s2.begin()
        let c2 = v.forward(s2, image: x2, w: w, h: h, deterministic: true)
        _ = s2.end()
        return vaeLossGeneric(cache: c2, target: hwcToChw(img), beta: 0.01, kind: .mse)
    }
    let matsN = v3n.matrices
    let sN = GpuSession(engine)
    sN.begin()
    let cN = v3n.forward(sN, image: x, w: w, h: h, deterministic: true)
    let (_, gN) = v3n.backward(sN, cN, target: x, beta: 0.01)
    _ = sN.end()
    var strongChecksN = 0
    var strongFailsN = 0
    for (L, wi) in [(0, 17), (2, 40), (8, 5), (10, 90)] {
        let m = matsN[L]
        let orig = m.data[wi]
        let eps = 1e-4
        var mp = matsN; mp[L].data[wi] = orig + Float(eps)
        var mm = matsN; mm[L].data[wi] = orig - Float(eps)
        let lp = fwdN(mp)
        let lm = fwdN(mm)
        let numeric = (lp - lm) / (2 * Double(eps))
        let analytic = Double(gN[L].toArray()[wi])
        guard abs(analytic) > 5e-5 && abs(numeric) > 5e-5 else { continue }
        strongChecksN += 1
        let sameSign = (analytic > 0) == (numeric > 0)
        let ratio = abs(analytic) / max(1e-12, abs(numeric))
        let inOrder = ratio > 0.5 && ratio < 2.0
        let ok = sameSign && inOrder
        if !ok { strongFailsN += 1 }
        expect(ok,
               String(format: "v3(no-res): dL/dW[%d][%d] аналит %.4e числ %.4e (знак %@, отношение %.2f)",
                      L, wi, analytic, numeric, sameSign ? "✓" : "✗", ratio))
    }
    expect(strongChecksN >= 2, "v3(no-res): достаточно сильных сигналов (\(strongChecksN))")
    expect(strongFailsN == 0, "v3(no-res): все сильные сигналы сходятся (\(strongFailsN)/\(strongChecksN))")
}

func testTrainV3GroupNorm() {
    section("Архитектура v3: обучение с GroupNorm (loss снижается)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let w = 32, h = 24

    // Обучение v3: loss снижается.
    let session = GpuSession(engine)
    let nFrames = 8
    var frames = [[Float]]()
    for f in 0..<nFrames {
        var r2 = SplitMix64(seed: UInt64(2000 + f))
        frames.append(makeSyntheticScene(w: w, h: h, rng: &r2))
    }
    let vae = ConvVAEGPU(engine: engine, residual: true, groupNorm: true)
    var firstLoss = Double.infinity
    var lastLoss = Double.infinity
    for step in 1...300 {
        let img2 = frames[(step - 1) % nFrames]
        let chw = hwcToChw(img2)
        let x2 = GPUTensor(engine: engine, data: chw, rows: 3)
        session.begin()
        let c2 = vae.forward(session, image: x2, w: w, h: h, deterministic: false, seed: 7)
        let (_, g2) = vae.backward(session, c2, target: x2, beta: 0.01)
        vae.applyAdam(session, grads: g2, t: step, lr: 0.004)
        _ = session.end()
        if step == 1 { firstLoss = vaeLossGeneric(cache: c2, target: chw, beta: 0.01, kind: .mse) }
        if step == 300 { lastLoss = vaeLossGeneric(cache: c2, target: chw, beta: 0.01, kind: .mse) }
    }
    expect(firstLoss.isFinite && lastLoss.isFinite, "v3: loss конечна")
    expect(lastLoss < firstLoss,
           "v3: loss снижается (\(String(format: "%.4f", firstLoss))→\(String(format: "%.4f", lastLoss)))")
}
