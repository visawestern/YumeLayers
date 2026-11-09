// YumeLayers — тесты Фазы 3: GPU-обучение денойзер-сети (MLP) на латентах.
import Foundation

func testTimeEmbedding() {
    section("Фаза 3: временное эмбеддинг (синусоидальное)")
    let e0 = timeEmbedding(t: 0, dim: 8)
    expectEqual(e0.count, 8, "размерность 8")
    // Формула: [sin(t/10000^(2i/d)), cos(t/10000^(2i/d))] парами.
    expectClose(Double(e0[1]), 1.0, 1e-9, "t=0: cos(0)=1 на нечётной позиции")
    expectClose(Double(e0[0]), 0.0, 1e-9, "t=0: sin(0)=0 на чётной позиции")
    let e1 = timeEmbedding(t: 1, dim: 8)
    expect(e1[0] > 0, "t=1: sin(1/10000)>0")
    expect(e1 != e0, "разные t → разные эмбеддинги")
    expectEqual(timeEmbedding(t: 7, dim: 8).count, 8, "t=7: размерность 8")

    // Детерминизм.
    expectEqual(timeEmbedding(t: 42, dim: 8), timeEmbedding(t: 42, dim: 8),
                "детерминизм")
}

func testDenoiserMLPShape() {
    section("Фаза 3: форма денойзер-MLP и градиентов")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let dim = 64
    let net = DenoiserMLPGPU(engine: engine, dimIn: dim, dimEmb: 8, hidden: 32)
    var rng = SplitMix64(seed: 3)
    let xArr = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    let epsArr = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    let emb = timeEmbedding(t: 5, dim: 8)

    session.begin()
    let xT = GPUTensor(engine: engine, data: xArr, rows: 1)
    let target = GPUTensor(engine: engine, data: epsArr, rows: 1)
    let (pred, cache) = net.forwardCached(session, xT: xT, emb: emb)
    let grads = net.backward(session, cache, target: target)
    _ = session.end()

    expectEqual(pred.toArray().count, dim, "pred: dim значений")
    expectEqual(grads.count, 6, "6 тензоров градиентов (3 слоя × W,b)")
    expect(grads.allSatisfy { $0.n > 0 }, "все градиенты непустые")
}

func testDenoiserNumericGrads() {
    section("Фаза 3: численная сверка градиентов денойзер-MLP")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let dim = 32
    var rng = SplitMix64(seed: 21)
    let xArr = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    let epsArr = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    let emb = timeEmbedding(t: 3, dim: 8)
    let seed: UInt64 = 0x1234_5678_9ABC_DEF0

    let net = DenoiserMLPGPU(engine: engine, dimIn: dim, dimEmb: 8, hidden: 24,
                             seed: seed)
    session.begin()
    let xT = GPUTensor(engine: engine, data: xArr, rows: 1)
    let target = GPUTensor(engine: engine, data: epsArr, rows: 1)
    let (_, cache) = net.forwardCached(session, xT: xT, emb: emb)
    let grads = net.backward(session, cache, target: target)
    _ = session.end()

    func fwd(_ mats: [Matrix]) -> Double {
        let v = DenoiserMLPGPU(engine: engine, dimIn: dim, dimEmb: 8, hidden: 24,
                               seed: seed)
        try? v.load(from: mats)
        let s2 = GpuSession(engine)
        s2.begin()
        let x2 = GPUTensor(engine: engine, data: xArr, rows: 1)
        let t2 = GPUTensor(engine: engine, data: epsArr, rows: 1)
        let (p, _) = v.forwardCached(s2, xT: x2, emb: emb)
        _ = s2.end()
        let pa = p.toArray()
        var sum = 0.0
        for i in 0..<dim {
            let d = Double(pa[i] - epsArr[i])
            sum += d * d
        }
        return sum / Double(dim)
    }

    let mats = net.matrices
    var strong = 0
    var fails = 0
    for (idx, wi) in [(0, 7), (2, 13), (4, 5)] {
        let analytic = Double(grads[idx].toArray()[wi])
        guard abs(analytic) > 1e-5 else { continue }
        strong += 1
        let eps = 1e-4
        var mp = mats; mp[idx].data[wi] += Float(eps)
        var mm = mats; mm[idx].data[wi] -= Float(eps)
        let numeric = (fwd(mp) - fwd(mm)) / (2 * Double(eps))
        let sameSign = (analytic > 0) == (numeric > 0)
        let ratio = abs(analytic) / max(1e-12, abs(numeric))
        let ok = sameSign && ratio > 0.5 && ratio < 2.0
        if !ok { fails += 1 }
        expect(ok,
               String(format: "градиент[%d][%d]: аналит %.4e числ %.4e (знак %@, отношение %.2f)",
                      idx, wi, analytic, numeric, sameSign ? "✓" : "✗", ratio))
    }
    expect(strong >= 3, "достаточно сильных сигналов (\(strong))")
    expect(fails == 0, "все сильные сигналы сходятся (0/\(strong))")
}

func testDenoiserTrainOverfit() {
    section("Фаза 3: обучение денойзер-MLP снижает loss")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let dim = 48
    // Фиксированный датасет: 8 пар (x_t, eps) с разными t.
    var dataset: [(x: [Float], eps: [Float], emb: [Float])] = []
    for k in 0..<8 {
        var rng = SplitMix64(seed: UInt64(1000 + k))
        dataset.append((
            x: (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) },
            eps: (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) },
            emb: timeEmbedding(t: k * 7, dim: 8)
        ))
    }
    let net = DenoiserMLPGPU(engine: engine, dimIn: dim, dimEmb: 8, hidden: 32)
    var firstLoss = Double.infinity
    var lastLoss = Double.infinity
    for step in 1...500 {
        var sumLoss = 0.0
        for sample in dataset {
            session.begin()
            let xT = GPUTensor(engine: engine, data: sample.x, rows: 1)
            let target = GPUTensor(engine: engine, data: sample.eps, rows: 1)
            let (pred, cache) = net.forwardCached(session, xT: xT, emb: sample.emb)
            let grads = net.backward(session, cache, target: target)
            sanitizeGrads(grads, maxNorm: 10.0, preScale: 1.0 / Float(dataset.count))
            net.applyAdam(session, grads, t: step, lr: 0.001)
            _ = session.end()
            let diff = (0..<dim).map { Double(pred.toArray()[$0] - sample.eps[$0]) }
            sumLoss += diff.reduce(0) { $0 + $1 * $1 } / Double(dim)
        }
        let loss = sumLoss / Double(dataset.count)
        if step == 1 { firstLoss = loss }
        if step == 500 { lastLoss = loss }
    }
    expect(firstLoss.isFinite && lastLoss.isFinite, "loss конечна")
    expect(lastLoss < firstLoss * 0.7,
           "обучение снижает loss (\(String(format: "%.4f", firstLoss))→\(String(format: "%.4f", lastLoss)))")
}
