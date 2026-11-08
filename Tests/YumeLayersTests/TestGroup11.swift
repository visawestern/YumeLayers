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
