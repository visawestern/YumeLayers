// YumeLayers — тесты редукции размерности и Float64/CPU-сети движения.
// Вынесены из TestGroup22.swift для соблюдения лимита 600 строк на файл.
import Foundation

func testMotionPoolMath() {
    section("Motion: редукция размерности (average-pool / upsample)")
    let d1 = latentPoolDims(h3: 60, w3: 107, block: 8)
    expect(d1.0 == 8 && d1.1 == 14, "60×107 → 8×14")
    let d2 = latentPoolDims(h3: 7, w3: 10, block: 1)
    expect(d2.0 == 7 && d2.1 == 10, "block=1 не меняет форму")
    let d3 = latentPoolDims(h3: 60, w3: 107, block: 1)
    expect(d3.0 == 60 && d3.1 == 107, "60×107 → 60×107")
    let z: [Float] = [1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12,
                      13, 14, 15, 16]
    let p = averagePoolLatent(z, channels: 1, h3: 4, w3: 4, block: 2)
    expectEqual(p, [3.5, 5.5, 11.5, 13.5], "пул 2×2: средние блоков")
    // Неполные блоки по краям: 4 элемента при блоке 3 → [1,2,3] и [4].
    let odd = averagePoolLatent([1, 2, 3, 4], channels: 1, h3: 1, w3: 4, block: 3)
    expectEqual(odd, [2.0, 4.0], "неполные блоки: [1,2,3]→2, [4]→4")
    let u = upsampleLatentNearest(p, channels: 1, ph: 2, pw: 2,
                                  h3: 4, w3: 4, block: 2)
    let expected: [Float] = [3.5, 3.5, 5.5, 5.5,
                             3.5, 3.5, 5.5, 5.5,
                             11.5, 11.5, 13.5, 13.5,
                             11.5, 11.5, 13.5, 13.5]
    expectEqual(u, expected, "upsample nearest повторяет значения пула")
}

/// E2E 480p: обучение сети движения с пулингом (block=8) поверх conv480.
func testMotionTrainingImprovesPrediction480() {
    section("Motion E2E 480p: пулинг + conv480")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let fm = FileManager.default
    let framesDir = "/tmp/yume_frames"
    guard let files = try? fm.contentsOfDirectory(atPath: framesDir) else {
        skip("нет /tmp/yume_frames"); return
    }
    let jpgs = files.filter { $0.hasSuffix(".jpg") }.sorted()
    guard jpgs.count >= 60 else { skip("мало кадров (\(jpgs.count))"); return }
    let ckpt = ProcessInfo.processInfo.environment["YUME_VAE_CKPT"]
        ?? (fm.homeDirectoryForCurrentUser.path + "/YumeLayers/checkpoints/conv.safetensors")
    guard fm.fileExists(atPath: ckpt),
          let st = try? loadStateWithMeta(path: ckpt) else { skip("нет VAE: \(ckpt)"); return }
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine, latentChannels: st.latentChannels,
                         width: st.modelWidth, residual: st.archVersion >= 2,
                         groupNorm: st.archVersion >= 3)
    guard (try? vae.load(from: st.mats)) != nil else { skip("чекпоинт не грузится"); return }
    let pw = 856, ph = 480
    let count = min(jpgs.count, 64)
    var fullLatents: [Int: [Float]] = [:]
    for i in 0..<count {
        let path = (framesDir as NSString).appendingPathComponent(jpgs[i])
        guard let rgb = loadImageRGB(path: path, w: pw, h: ph),
              let z = try? encodeLatent(vae: vae, image: rgb, w: pw, h: ph) else { continue }
        fullLatents[i] = z
    }
    guard fullLatents.count >= 56 else { skip("не удалось закодировать"); return }
    let h3 = pw / 8, w3 = ph / 8
    let channels = fullLatents.values.first!.count / (h3 * w3)
    let block = 8
    var latents: [Int: [Float]] = [:]
    for (idx, z) in fullLatents {
        latents[idx] = averagePoolLatent(z, channels: channels, h3: h3,
                                         w3: w3, block: block)
    }
    let trainFrameCount = 56
    let trainPairs = (motionBuildPairs(0, trainFrameCount, 48, 24)
        + motionBuildPairs(0, trainFrameCount, 56, 56))
        .filter { latents[$0.a] != nil && latents[$0.b] != nil && latents[$0.k] != nil }
    guard !trainPairs.isEmpty else { skip("нет обучающих пар"); return }
    let net = MotionMLPGPU(engine: engine, latentDim: latents.values.first!.count,
                           dimEmb: 8, hidden: 96)
    net.latentH3 = h3
    net.latentW3 = w3
    net.poolBlock = block
    motionNormalizeNet(net, latents: latents, range: 0..<trainFrameCount)
    let session = GpuSession(engine)
    let mseBuf = engine.device.makeBuffer(length: 4, options: .storageModeShared)
    var rng = SplitMix64(seed: 0x480D_0000_0000_0001)
    var nanLoss = 0
    for step in 1...4000 {
        let p = trainPairs[Int(rng.next() % UInt64(trainPairs.count))]
        let loss = runMotionStep(net: net, session: session, pair: p,
                                 latents: latents, mseBuf: mseBuf,
                                 step: step, lr: 0.0005)
        if !loss.isFinite { nanLoss += 1 }
    }
    expect(nanLoss == 0, "обучение без NaN (\(nanLoss) шагов с NaN)")
    let main = motionEvalWindow(net: net, latents: latents, a: 0,
                                b: trainFrameCount - 1)
    print(String(format: "      Motion E2E 480p окно (0,%d): lerp MSE %.5f · сеть MSE %.5f (улучшение %.1f%%)",
                 trainFrameCount - 1, main.lerp, main.net,
                 (1 - main.net / max(main.lerp, 1e-12)) * 100))
    expect(main.net < main.lerp, "сеть точнее lerp на главном окне (480p)")
    if main.lerp > 0.0005 {
        expect(main.net < main.lerp * 0.90,
               "улучшение ≥ 10% при существенном движении")
    }
}

/// Численная проверка градиентов CPU-Double сети движения.
func testMotionMLPDoubleNumericGrads() {
    section("MotionMLP Double: численная сверка градиентов")
    let latentDim = 16
    let dimEmb = 4
    let hidden = 8
    let seed: UInt64 = 0x5151_0000_0000_0001
    let net = MotionMLPDouble(latentDim: latentDim, dimEmb: dimEmb, hidden: hidden,
                              seed: seed)
    var seededMats = net.matrices
    for mIdx in [0, 2, 4] {
        var i = 0
        while i < seededMats[mIdx].data.count {
            seededMats[mIdx].data[i] += 0.2
            i += 37
        }
    }
    try? net.load(from: seededMats)
    var rng = SplitMix64(seed: 99)
    let zA = (0..<latentDim).map { _ in Double.random(in: -1...1, using: &rng) }
    let zB = (0..<latentDim).map { _ in Double.random(in: -1...1, using: &rng) }
    let zK = (0..<latentDim).map { _ in Double.random(in: -1...1, using: &rng) }
    let base = (0..<latentDim).map { zA[$0] + (zB[$0] - zA[$0]) * 0.4 }
    let emb = (0..<dimEmb).map { _ in Double.random(in: -1...1, using: &rng) }
    let xT = zA + zB
    let (pred, cache) = net.forward(xT: xT, emb: emb, base: base)
    var dOut = [Double](repeating: 0, count: latentDim)
    for k in 0..<latentDim { dOut[k] = 2 * (pred[k] - zK[k]) / Double(latentDim) }
    let g = net.backward(cache, dOut: dOut)

    func fwd(_ mats: [Matrix]) -> Double {
        let n = MotionMLPDouble(latentDim: latentDim, dimEmb: dimEmb, hidden: hidden,
                                seed: seed)
        try? n.load(from: mats)
        let (p, _) = n.forward(xT: xT, emb: emb, base: base)
        var s = 0.0
        for k in 0..<latentDim {
            let d = p[k] - zK[k]
            s += d * d
        }
        return s / Double(latentDim)
    }

    let mats = net.matrices
    var strong = 0
    var fails = 0
    let targets: [(Int, Int)] = [(0, 3), (2, 1), (4, 5)]
    let gArrays: [[Double]] = [g.gW1, g.gW2, g.gW3]
    for (mi, (idx, wi)) in targets.enumerated() {
        let stride = mi == 0 ? 2 * latentDim + dimEmb : hidden
        let analytic = gArrays[mi][idx * stride + wi]
        guard abs(analytic) > 1e-8 else { continue }
        strong += 1
        let eps = 1e-3
        var mp = mats
        var mm = mats
        mp[idx].data[idx * stride + wi] += Float(eps)
        mm[idx].data[idx * stride + wi] -= Float(eps)
        let numeric = (fwd(mp) - fwd(mm)) / (2 * eps)
        let ok = abs(analytic - numeric) <= max(1e-4, abs(analytic) * 0.5)
        if !ok { fails += 1 }
        expect(ok, String(format: "grad[%d][%d]: аналит %.6e числ %.6e", mi, wi, analytic, numeric))
    }
    expect(strong >= 2, "достаточно сильных сигналов (\(strong))")
    expect(fails == 0, "все сильные сигналы сходятся")
}

/// Overfit CPU-Double сети: loss должен падать.
func testMotionMLPDoubleOverfit() {
    section("MotionMLP Double: обучение снижает loss")
    let latentDim = 24
    let dimEmb = 4
    var dataset: [(a: [Double], b: [Double], k: [Double], t: Double)] = []
    for s in 0..<16 {
        var rng = SplitMix64(seed: UInt64(7000 + s))
        let a = (0..<latentDim).map { _ in Double.random(in: -1...1, using: &rng) }
        let b = (0..<latentDim).map { _ in Double.random(in: -1...1, using: &rng) }
        let t = Double(s) / 15.0
        let kk = (0..<latentDim).map { i in
            a[i] + (b[i] - a[i]) * t + 0.2 * sin(Double(i) * t * 3)
        }
        dataset.append((a: a, b: b, k: kk, t: t))
    }
    let net = MotionMLPDouble(latentDim: latentDim, dimEmb: dimEmb, hidden: 32)
    var firstLoss = Double.infinity
    var lastLoss = Double.infinity
    for step in 1...2000 {
        var sumLoss = 0.0
        for sample in dataset {
            let base = (0..<latentDim).map {
                sample.a[$0] + (sample.b[$0] - sample.a[$0]) * sample.t
            }
            let xT = sample.a + sample.b
            let emb = (0..<dimEmb).map { _ in 0.0 }
            let (pred, cache) = net.forward(xT: xT, emb: emb, base: base)
            var mse = 0.0
            for k in 0..<latentDim {
                let d = pred[k] - sample.k[k]
                mse += d * d
            }
            mse /= Double(latentDim)
            sumLoss += mse
            var dOut = [Double](repeating: 0, count: latentDim)
            for k in 0..<latentDim {
                dOut[k] = 2 * (pred[k] - sample.k[k]) / Double(latentDim)
            }
            let g = net.backward(cache, dOut: dOut)
            net.applyAdam(g, t: step, lr: 0.002)
        }
        let avg = sumLoss / Double(dataset.count)
        if step == 1 { firstLoss = avg }
        lastLoss = avg
    }
    print(String(format: "      MotionMLP Double overfit: loss %.6f → %.6f", firstLoss, lastLoss))
    expect(firstLoss.isFinite && lastLoss.isFinite, "loss конечен (без NaN)")
    expect(lastLoss < firstLoss, "обучение снижает loss")
    expect(lastLoss < firstLoss * 0.2, "значимое снижение")
}
