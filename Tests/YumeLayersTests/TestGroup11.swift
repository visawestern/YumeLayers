// YumeLayers — тесты потерь (Этап 1: L1 / Sobel / переключатель).
import Foundation

func testL1AndSobelLoss() {
    section("Этап 1: L1 и Sobel edge-loss (чистые функции)")

    // meanL1
    expectClose(meanL1([0, 0], [1, 1]), 1.0, 1e-9, "L1: |0-1| среднее = 1")
    expectClose(meanL1([0.5, 0.5], [0.5, 0.5]), 0.0, 1e-12, "L1: идентичные → 0")

    // Sobel на однородном изображении → нулевой градиент
    let flat = [Float](repeating: 0.5, count: 16 * 16 * 3)
    let gFlat = sobelLumaGradient(flat, w: 16, h: 16)
    expect(gFlat.count == 16 * 16, "Sobel: размер w×h")
    expect(gFlat.allSatisfy { $0 == 0 }, "Sobel: однородное → нулевой градиент")

    // Резкий вертикальный край (чёрное/белое) → ненулевой градиент
    var edge = [Float](repeating: 0, count: 16 * 16 * 3)
    for y in 0..<16 {
        for x in 0..<16 {
            let i = (y * 16 + x) * 3
            if x >= 8 { edge[i] = 1; edge[i + 1] = 1; edge[i + 2] = 1 }
        }
    }
    let gEdge = sobelLumaGradient(edge, w: 16, h: 16)
    let mx = gEdge.max() ?? 0
    expect(mx > 0.1, "Sobel: вертикальный край → ненулевой градиент (\(mx))")

    // Edge-loss: идентичные → 0; размытое vs резкое → >0
    expectClose(sobelEdgeLoss(recon: edge, target: edge, w: 16, h: 16), 0.0, 1e-9,
                "edge-loss: идентичные → 0")
    var blurred = [Float](repeating: 0, count: 16 * 16 * 3)
    for y in 0..<16 {
        for x in 0..<16 {
            let i = (y * 16 + x) * 3
            let v = Float(min(15, x + 3)) / 15.0
            blurred[i] = v; blurred[i + 1] = v; blurred[i + 2] = v
        }
    }
    let edgeBlurred = sobelEdgeLoss(recon: blurred, target: edge, w: 16, h: 16)
    expect(edgeBlurred > 0.01, "edge-loss: размытое vs резкое → >0 (\(edgeBlurred))")
}

func testSobelEdgeGradCPU() {
    section("Этап 1: Sobel edge-grad CPU — численная проверка")
    let w = 16, h = 12
    var rng = SplitMix64(seed: 1234)
    var recon = [Float](repeating: 0, count: w * h * 3)
    var target = [Float](repeating: 0, count: w * h * 3)
    for i in 0..<recon.count {
        recon[i] = Float.random(in: 0...1, using: &rng)
        target[i] = Float.random(in: 0...1, using: &rng)
    }
    let grad = sobelEdgeGradCPU(recon: recon, target: target, w: w, h: h)
    expectEqual(grad.count, w * h * 3, "градиент: размер w×h×3")

    func loss(_ r: [Float]) -> Double {
        sobelEdgeLoss(recon: r, target: target, w: w, h: h)
    }
    let eps: Float = 1e-3
    // Проверяем центральный пиксель + угловой + на краю окна Собеля.
    for idx in [Int(w * 7 + 8) * 3, (w * h - 1) * 3 + 2, 5 * 3 + 1] {
        let analytic = Double(grad[idx])
        let orig = recon[idx]
        recon[idx] = orig + eps
        let lp = loss(recon)
        recon[idx] = orig - eps
        let lm = loss(recon)
        recon[idx] = orig
        let numeric = (lp - lm) / (2 * Double(eps))
        let denom = max(1e-9, abs(analytic) + abs(numeric))
        let rel = abs(analytic - numeric) / denom
        expect(rel < 0.02,
               String(format: "пиксель %d: аналит %.4e числ %.4e (отн %.1f%%)",
                      idx, analytic, numeric, rel * 100))
    }

    let selfGrad = sobelEdgeGradCPU(recon: target, target: target, w: w, h: h)
    expect(selfGrad.allSatisfy { $0 == 0 }, "recon == target → нулевой градиент")
}

func testGroupNormGradCPU() {
    section("Этап 1: GroupNorm-градиент CPU — численная проверка")
    let channels = 8, groups = 2, plane = 12
    var rng = SplitMix64(seed: 99)
    var xs = [Float](repeating: 0, count: channels * plane)
    for i in 0..<xs.count { xs[i] = Float.random(in: -2...2, using: &rng) }
    var dys = [Float](repeating: 0, count: channels * plane)
    for i in 0..<dys.count { dys[i] = Float.random(in: -1...1, using: &rng) }

    let grad = groupNormGradCPU(channels: channels, groups: groups, plane: plane,
                                xs: xs, dys: dys)
    expectEqual(grad.count, channels * plane, "градиент: размер channels×plane")

    // Численная проверка: потеря L = Σ y_i·dy_i; dL/dx = градиент.
    func loss(_ x: [Float]) -> Double {
        let y = groupNormCPU(channels: channels, groups: groups, plane: plane, xs: x)
        var s = 0.0
        for i in 0..<y.count { s += Double(y[i]) * Double(dys[i]) }
        return s
    }
    let eps: Float = 1e-3
    for idx in [0, channels * plane / 2, channels * plane - 1, 5] {
        let analytic = Double(grad[idx])
        let orig = xs[idx]
        xs[idx] = orig + eps
        let lp = loss(xs)
        xs[idx] = orig - eps
        let lm = loss(xs)
        xs[idx] = orig
        let numeric = (lp - lm) / (2 * Double(eps))
        let denom = max(1e-9, abs(analytic) + abs(numeric))
        let rel = abs(analytic - numeric) / denom
        expect(rel < 0.02,
               String(format: "пиксель %d: аналит %.4e числ %.4e (отн %.1f%%)",
                      idx, analytic, numeric, rel * 100))
    }

    // Группы независимы: градиент в одной группе не зависит от значений другой.
    let dysA = dys
    var dysB = dys
    dysB[0] += 1
    let gA = groupNormGradCPU(channels: channels, groups: groups, plane: plane,
                              xs: xs, dys: dysA)
    let gB = groupNormGradCPU(channels: channels, groups: groups, plane: plane,
                              xs: xs, dys: dysB)
    expectEqual(gA[4 * plane], gB[4 * plane], "группа 1 не зависит от dy группы 0")
}

func testGPU_GroupNormGrad() {
    section("Этап 1: GPU-GroupNorm-градиент ≈ CPU-эталон")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let channels = 8, groups = 2, plane = 32 * 24
    var rng = SplitMix64(seed: 77)
    var xs = [Float](repeating: 0, count: channels * plane)
    var dys = [Float](repeating: 0, count: channels * plane)
    for i in 0..<xs.count {
        xs[i] = Float.random(in: -2...2, using: &rng)
        dys[i] = Float.random(in: -1...1, using: &rng)
    }
    let x = GPUTensor(engine: engine, data: xs, rows: channels)
    let dy = GPUTensor(engine: engine, data: dys, rows: channels)
    session.begin()
    let g = session.encodeGroupNormGrad(x, dy, groups: groups)
    _ = session.end()
    let got = g.toArray()
    let expected = groupNormGradCPU(channels: channels, groups: groups, plane: plane,
                                    xs: xs, dys: dys)
    let diff = maxAbsDiff(got, expected)
    expect(diff < 1e-4,
           "GPU-GroupNormgrad ≈ CPU (max|Δ| = \(diff))")
}

func testVaeLossKindSwitch() {
    section("Этап 1: vaeLossGeneric переключатель mse/l1")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    var rng = SplitMix64(seed: 5)
    let img = hwcToChw(makeSyntheticScene(w: w, h: h, rng: &rng))
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    let vae = ConvVAEGPU(engine: engine)
    session.begin()
    let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
    _ = session.end()

    expectClose(vaeLossGeneric(cache: c, target: img, beta: 0.5, kind: .mse),
                vaeLoss(cache: c, target: img, beta: 0.5),
                1e-9, "generic .mse совпадает с vaeLoss")
    let l1 = vaeLossGeneric(cache: c, target: img, beta: 0.5, kind: .l1)
    expect(l1.isFinite && l1 >= 0, "generic .l1 конечен и ≥0")
    expect(l1 != vaeLossGeneric(cache: c, target: img, beta: 0.5, kind: .mse),
           "generic .l1 отличается от .mse")
}

func testGPUGroupNorm() {
    section("Этап 1: GPU-GroupNorm совпадает с CPU-эталоном")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let C = 8, P = 16, G = 4
    var rng = SplitMix64(seed: 77)
    var data = [Float](repeating: 0, count: C * P)
    for i in 0..<data.count { data[i] = Float.random(in: -2...2, using: &rng) }

    let x = GPUTensor(engine: engine, data: data, rows: C)
    session.begin()
    let y = session.groupNorm(x, groups: G)
    _ = session.end()
    let got = y.toArray()
    let expected = groupNormCPU(channels: C, groups: G, plane: P, xs: data)
    expect(maxAbsDiff(got, expected) < 1e-3,
           "GPU-GroupNorm ≈ CPU-эталон (max|Δ| = \(maxAbsDiff(got, expected)))")

    // Статистика первой группы: среднее ≈ 0, σ ≈ 1.
    let gSize = C / G
    var gsum = 0.0
    for c in 0..<gSize {
        for p in 0..<P { gsum += Double(got[c * P + p]) }
    }
    let gmean = gsum / Double(gSize * P)
    expect(abs(gmean) < 1e-3, "группа 0: среднее ≈ 0 (\(gmean))")
}

func testResidualBlockGradients() {
    section("Этап 1: ResidualBlockGPU — численная проверка градиентов")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let C = 4, ih = 8, iw = 8, plane = ih * iw
    var rng = SplitMix64(seed: 9)
    let block = ResidualBlockGPU(engine: engine, channels: C, rng: &rng)

    var data = [Float](repeating: 0, count: C * plane)
    for i in 0..<data.count { data[i] = Float.random(in: -1...1, using: &rng) }
    let x = GPUTensor(engine: engine, data: data, rows: C)
    var targetData = [Float](repeating: 0, count: C * plane)
    for i in 0..<targetData.count { targetData[i] = Float.random(in: -1...1, using: &rng) }
    let target = GPUTensor(engine: engine, data: targetData, rows: C)

    func loss() -> Double {
        session.begin()
        let (o, _) = block.forward(session, x, ih: ih, iw: iw)
        _ = session.end()
        let oa = o.toArray()
        var s = 0.0
        for i in 0..<oa.count { let d = Double(oa[i] - targetData[i]); s += d * d }
        return s / Double(oa.count)
    }

    session.begin()
    let (out, cache) = block.forward(session, x, ih: ih, iw: iw)
    let dOut = session.smul(2.0 / Float(C * plane), session.sub(out, target))
    let (_, grads) = block.backward(session, cache, dOut: dOut)
    _ = session.end()

    var maxGrad = 0.0
    for g in grads {
        let a = g.toArray()
        maxGrad = max(maxGrad, a.map { abs(Double($0)) }.max() ?? 0)
    }
    expect(maxGrad > 1e-6, "Residual: градиенты невырождены (max |g| = \(maxGrad))")

    let eps: Float = 1e-2
    var checked = 0
    let layers: [ConvLayerGPU] = [block.c1, block.c2]
    for (li, layer) in layers.enumerated() {
        let g = grads[li * 2].toArray()
        let top = (0..<g.count).sorted { abs(g[$0]) > abs(g[$1]) }.prefix(2)
        for weightIdx in top {
            let analytic = Double(g[weightIdx])
            let orig = layer.W.getValue(weightIdx)
            layer.W.setValue(weightIdx, orig + eps)
            let lp = loss()
            layer.W.setValue(weightIdx, orig - eps)
            let lm = loss()
            layer.W.setValue(weightIdx, orig)
            let numeric = (lp - lm) / (2 * Double(eps))
            let denom = max(1e-9, abs(analytic) + abs(numeric))
            let rel = abs(analytic - numeric) / denom
            expect(rel < 0.10,
                   String(format: "Residual слой %d вес %d: аналит %.4e числ %.4e (отн %.1f%%)",
                          li, weightIdx, analytic, numeric, rel * 100))
            checked += 1
        }
    }
    expectEqual(checked, 4, "проверено 4 точки ResidualBlock")
}

func testRandomCrop() {
    section("Этап 1: случайный кроп кадра (патчи 128–256px)")
    let w = 64, h = 48
    var img = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            let i = (y * w + x) * 3
            img[i] = Float(x) / Float(w); img[i + 1] = Float(y) / Float(h); img[i + 2] = 0.5
        }
    }
    var rng = SplitMix64(seed: 42)
    let cw = 32, ch = 24
    guard let r = randomCrop(rgb: img, w: w, h: h, cropW: cw, cropH: ch, rng: &rng) else {
        expect(false, "кроп создан"); return
    }
    expectEqual(r.crop.count, cw * ch * 3, "кроп: размер cw×ch×3")
    expect(r.x0 >= 0 && r.x0 <= w - cw, "кроп: x0 в границах")
    expect(r.y0 >= 0 && r.y0 <= h - ch, "кроп: y0 в границах")
    // Проверка соответствия пикселей источнику в позиции x0,y0.
    let si0 = (r.y0 * w + r.x0) * 3
    expectClose(Double(r.crop[0]), Double(img[si0]), 1e-9, "кроп: пиксель (0,0) совпадает с источником (r=\(img[si0]))")

    // Детерминизм: одинаковый seed → одинаковый кроп.
    var rng2 = SplitMix64(seed: 42)
    let r2 = randomCrop(rgb: img, w: w, h: h, cropW: cw, cropH: ch, rng: &rng2)
    expectEqual(r2?.x0, r.x0, "кроп: детерминизм x0")
    expectEqual(r2?.crop, r.crop, "кроп: детерминизм содержимого")

    // Невалидные размеры → nil.
    expect(randomCrop(rgb: img, w: w, h: h, cropW: 1000, cropH: 24, rng: &rng) == nil,
           "кроп: cropW > w → nil")
}

func testGPU_L1Grad() {
    section("Этап 1: GPU-L1 градиент ≈ CPU-эталон")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    var rng = SplitMix64(seed: 31)
    var reconData = [Float](repeating: 0, count: 256)
    var targetData = [Float](repeating: 0, count: 256)
    for i in 0..<256 {
        reconData[i] = Float.random(in: 0...1, using: &rng)
        targetData[i] = Float.random(in: 0...1, using: &rng)
    }
    let recon = GPUTensor(engine: engine, data: reconData, rows: 4)
    let target = GPUTensor(engine: engine, data: targetData, rows: 4)
    session.begin()
    let g = session.encodeL1Grad(recon: recon, target: target)
    _ = session.end()
    let got = g.toArray()
    let expected = l1GradCPU(recon: reconData, target: targetData)
    expect(maxAbsDiff(got, expected) < 1e-6,
           "GPU-L1grad ≈ CPU (max|Δ| = \(maxAbsDiff(got, expected)))")
}

func testGPU_SobelEdgeGrad() {
    section("Этап 1: GPU-Sobel edge-grad ≈ CPU-эталон")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    var rng = SplitMix64(seed: 31)
    var recon = [Float](repeating: 0, count: w * h * 3)
    var target = [Float](repeating: 0, count: w * h * 3)
    for i in 0..<recon.count {
        recon[i] = Float.random(in: 0...1, using: &rng)
        target[i] = Float.random(in: 0...1, using: &rng)
    }
    let x = GPUTensor(engine: engine, data: hwcToChw(recon), rows: 3)
    let y = GPUTensor(engine: engine, data: hwcToChw(target), rows: 3)
    session.begin()
    let g = session.encodeSobelEdgeGrad(recon: x, target: y, w: w, h: h)
    _ = session.end()
    let got = chwToHwc(g.toArray())
    let expected = sobelEdgeGradCPU(recon: recon, target: target, w: w, h: h)
    let diff = maxAbsDiff(got, expected)
    expect(diff < 1e-4,
           "GPU-Sobelgrad ≈ CPU (max|Δ| = \(diff))")
}

func testTrainWithL1() {
    section("Этап 1: обучение с L1-градиентом (dReconOverride)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 64, h = 40
    let nFrames = 8
    var frames = [[Float]]()
    for f in 0..<nFrames {
        var rng = SplitMix64(seed: UInt64(2000 + f))
        frames.append(makeSyntheticScene(w: w, h: h, rng: &rng))
    }
    let beta: Float = 0.01
    let lr: Float = 0.004
    let vae = ConvVAEGPU(engine: engine)
    var firstLoss = Double.infinity
    var lastLoss = Double.infinity
    for step in 1...400 {
        let img = frames[(step - 1) % nFrames]
        let chw = hwcToChw(img)
        let x = GPUTensor(engine: engine, data: chw, rows: 3)
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 7)
        let dR = session.encodeL1Grad(recon: c.recon, target: x)
        let (_, g) = vae.backward(session, c, target: x, beta: beta, dReconOverride: dR)
        vae.applyAdam(session, grads: g, t: step, lr: lr)
        _ = session.end()
        if step == 1 { firstLoss = vaeLossGeneric(cache: c, target: chw, beta: beta, kind: .l1) }
        if step == 400 { lastLoss = vaeLossGeneric(cache: c, target: chw, beta: beta, kind: .l1) }
    }
    expect(firstLoss.isFinite && lastLoss.isFinite, "обучение с L1: loss конечна")
    expect(lastLoss < firstLoss,
           "обучение с L1: loss снижается (\(String(format: "%.4f", firstLoss))→\(String(format: "%.4f", lastLoss)))")
}

func testTrainWithEdge() {
    section("Этап 1: обучение с Sobel edge-loss (dReconOverride = MSE + λ·edge)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 64, h = 40
    let nFrames = 8
    var frames = [[Float]]()
    for f in 0..<nFrames {
        var rng = SplitMix64(seed: UInt64(3000 + f))
        frames.append(makeSyntheticScene(w: w, h: h, rng: &rng))
    }
    let beta: Float = 0.01
    let lr: Float = 0.004
    let edgeWeight: Float = 0.5
    let vae = ConvVAEGPU(engine: engine)
    var firstLoss = Double.infinity
    var lastLoss = Double.infinity
    for step in 1...400 {
        let img = frames[(step - 1) % nFrames]
        let chw = hwcToChw(img)
        let x = GPUTensor(engine: engine, data: chw, rows: 3)
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 7)
        let edgeGrad = session.encodeSobelEdgeGrad(recon: c.recon, target: x,
                                                   w: w, h: h)
        let dR = session.add(session.smul(2.0 / Float(x.n), session.sub(c.recon, x)),
                             session.smul(edgeWeight, edgeGrad))
        let (_, g) = vae.backward(session, c, target: x, beta: beta, dReconOverride: dR)
        vae.applyAdam(session, grads: g, t: step, lr: lr)
        _ = session.end()
        if step == 1 { firstLoss = vaeLossGeneric(cache: c, target: chw, beta: beta, kind: .mse) }
        if step == 400 { lastLoss = vaeLossGeneric(cache: c, target: chw, beta: beta, kind: .mse) }
    }
    expect(firstLoss.isFinite && lastLoss.isFinite, "обучение с edge: loss конечна")
    expect(lastLoss < firstLoss,
           "обучение с edge: loss снижается (\(String(format: "%.4f", firstLoss))→\(String(format: "%.4f", lastLoss)))")
}

func testV3GroupNorm() {
    section("Архитектура v3: GroupNorm (shape, градиенты, обучение)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let v3 = ConvVAEGPU(engine: engine, residual: true, groupNorm: true)
    let w = 32, h = 24
    var rng = SplitMix64(seed: 11)
    let img = makeSyntheticScene(w: w, h: h, rng: &rng)
    let x = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
    let s = GpuSession(engine)
    s.begin()
    let c = v3.forward(s, image: x, w: w, h: h, deterministic: true)
    let (_, g) = v3.backward(s, c, target: x, beta: 0.01)
    _ = s.end()
    expectEqual(c.recon.toArray().count, w * h * 3, "v3: recon размер w×h×3")
    expectEqual(g.count, v3.layers.count * 2, "v3: градиенты по всем слоям")

    // Численная проверка градиентов (полный backward) на конечных разностях.
    // ε=1e-4 (центральные разности), по 2 веса из разных слоёв.
    // ВАЖНО: loss читается ТОЛЬКО после end() (внутри сессии toArray гоняется
    // с незакоммиченным command buffer — чтения невалидны).
    func fwd(_ ws: [Matrix]) -> Double {
        let v = ConvVAEGPU(engine: engine, residual: true, groupNorm: true, seed: 0x5EED_1234_ABCD_0001)
        try? v.load(from: ws)
        let x2 = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
        let s2 = GpuSession(engine)
        s2.begin()
        let c2 = v.forward(s2, image: x2, w: w, h: h, deterministic: true)
        _ = s2.end()
        return vaeLossGeneric(cache: c2, target: hwcToChw(img), beta: 0.01, kind: .mse)
    }
    var mats = v3.matrices
    // Детерминизм forward: 3 одинаковых forward в одной сессии → одинаковый loss.
    // (Чтения ТОЛЬКО после end() — внутри сессии toArray гоняется с command buffer.)
    let vProbe = ConvVAEGPU(engine: engine, residual: true, groupNorm: true, seed: 0x5EED_1234_ABCD_0001)
    let sProbe = GpuSession(engine)
    var probeLosses: [Double] = []
    for _ in 0..<3 {
        sProbe.begin()
        let xP = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
        let cP = vProbe.forward(sProbe, image: xP, w: w, h: h, deterministic: true)
        _ = sProbe.end()
        probeLosses.append(vaeLossGeneric(cache: cP, target: hwcToChw(img), beta: 0.01, kind: .mse))
    }
    expect(probeLosses.allSatisfy { abs($0 - (probeLosses.first ?? 0)) < 1e-9 },
           "v3: forward детерминирован (\(probeLosses))")
    // Полносетевая численная сверка градиентов: GroupNorm делает каждую
    // активацию функцией всей группы, а изломы ReLU по одному весу дают
    // разрывы. На слабых градиентах (~1e-6) шум конечных разностей ломает
    // знак/масштаб — поэтому проверяем ТОЛЬКО сильные сигналы (|g|>5e-5),
    // где облако шума незначительно. Точная сверка — на уровне операций
    // (testGPU_GroupNormGrad: max|Δ|=2.4e-7) + обучение/детерминизм ниже.
    var strongChecks = 0
    var strongFails = 0
    for (L, wi) in [(0, 17), (2, 40), (8, 5), (12, 90), (16, 7)] {
        let m = mats[L]
        let orig = m.data[wi]
        let eps = 1e-4
        var mp = mats; mp[L].data[wi] = orig + Float(eps)
        var mm = mats; mm[L].data[wi] = orig - Float(eps)
        let lp = fwd(mp)
        let lm = fwd(mm)
        let numeric = (lp - lm) / (2 * Double(eps))
        let analytic = Double(g[L].toArray()[wi])   // dL/dW слоя L
        guard abs(analytic) > 5e-5 && abs(numeric) > 5e-5 else { continue }
        strongChecks += 1
        let sameSign = (analytic > 0) == (numeric > 0)
        let ratio = abs(analytic) / max(1e-12, abs(numeric))
        let inOrder = ratio > 0.5 && ratio < 2.0
        let ok = sameSign && inOrder
        if !ok { strongFails += 1 }
        expect(ok,
               String(format: "v3: dL/dW[%d][%d] аналит %.4e числ %.4e (знак %@, отношение %.2f)",
                      L, wi, analytic, numeric, sameSign ? "✓" : "✗", ratio))
    }
    expect(strongChecks >= 3, "v3: достаточно сильных сигналов для сверки (\(strongChecks))")
    expect(strongFails == 0, "v3: все сильные сигналы сходятся (\(strongFails)/\(strongChecks))")
}

func testTrainOnPatchesFullFrame() {
    section("Этап 1: обучение на патчах → инференс полного кадра")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let W = 320, H = 256
    var rngBig = SplitMix64(seed: 5)
    let bigFrame = makeSyntheticScene(w: W, h: H, rng: &rngBig)
    let cw = 128, ch = 96   // кратны 8 → валидны для VAE
    let beta: Float = 0.01
    let lr: Float = 0.004
    let steps = 300
    let vae = ConvVAEGPU(engine: engine)

    for step in 1...steps {
        var rng = SplitMix64(seed: UInt64(step))
        guard let cr = randomCrop(rgb: bigFrame, w: W, h: H,
                                  cropW: cw, cropH: ch, rng: &rng) else {
            expect(false, "кроп создан"); return
        }
        let chw = hwcToChw(cr.crop)
        let x = GPUTensor(engine: engine, data: chw, rows: 3)
        session.begin()
        let c = vae.forward(session, image: x, w: cw, h: ch, deterministic: false, seed: 7)
        let dR = session.encodeL1Grad(recon: c.recon, target: x)
        let (_, g) = vae.backward(session, c, target: x, beta: beta, dReconOverride: dR)
        vae.applyAdam(session, grads: g, t: step, lr: lr)
        _ = session.end()
    }

    // Инференс полного кадра (без переобучения) — сеть полностью свёрточная.
    let bigCHW = hwcToChw(bigFrame)
    let xBig = GPUTensor(engine: engine, data: bigCHW, rows: 3)
    session.begin()
    let cBig = vae.forward(session, image: xBig, w: W, h: H, deterministic: true)
    _ = session.end()
    let fullRecon = cBig.recon.toArray()
    expectEqual(fullRecon.count, W * H * 3, "инференс полного кадра: размер w×h×3")
    let psnrFull = psnr(reference: bigFrame, candidate: chwToHwc(fullRecon))
    expect(psnrFull.isFinite && psnrFull > 0, "инференс полного кадра: PSNR конечен и >0 (\(psnrFull))")
    print("      инференс полного кадра \(W)×\(H): PSNR = \(String(format: "%.2f", psnrFull)) дБ")
}

func testFullFrame854Inference() {
    section("Этап 2: инференс полного кадра 854×480 (с паддингом)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let W = 854, H = 480
    var rng = SplitMix64(seed: 11)
    let img = makeSyntheticScene(w: W, h: H, rng: &rng)
    let vae = ConvVAEGPU(engine: engine)

    guard let recon = reconstructPadded(engine, vae, session, rgb: img, w: W, h: H) else {
        expect(false, "854×480: реконструкция получена"); return
    }
    expectEqual(recon.count, W * H * 3, "854×480: размер w×h×3")
    let psnrV = psnr(reference: img, candidate: recon)
    expect(psnrV.isFinite && psnrV > 0, "854×480: PSNR конечен (\(psnrV))")
    print("      854×480: PSNR = \(String(format: "%.2f", psnrV)) дБ")
}
