// YumeLayers — Фаза 4: планирование движения (TDD, чистые функции).
import Foundation

func testResampleSegmentIndices() {
    section("Motion: ресемплинг сегмента видео")
    // Концы всегда включены.
    let idx = resampleSegmentIndices(total: 241, targetCount: 120)
    expect(idx.count == 120, "выборка содержит ровно 120 индексов (получено \(idx.count))")
    expect(idx.first == 0, "первый индекс = 0")
    expect(idx.last == 240, "последний индекс = 240")
    // Строгое возрастание.
    var increasing = true
    for i in 1..<idx.count where idx[i] <= idx[i - 1] { increasing = false }
    expect(increasing, "индексы строго возрастают")
    // Равномерность: шаги равны ~2 (241 кадр → 120 позиций).
    let gaps = (1..<idx.count).map { idx[$0] - idx[$0 - 1] }
    let avgStep = Double(gaps.reduce(0, +)) / Double(gaps.count)
    expect(abs(avgStep - 2.0) < 0.5,
           "средний шаг ≈ 2 (получено \(String(format: "%.2f", avgStep)))")
    var maxGap = 0
    for g in gaps { maxGap = max(maxGap, g) }
    expect(maxGap <= 3, "нет резких скачков: максимальный шаг \(maxGap) ≤ 3")

    // Маленький сегмент: ничего не добавляем.
    let small = resampleSegmentIndices(total: 10, targetCount: 5)
    expectEqual(small.count, 5, "10 кадров → 5 позиций")
    expect(small.first == 0 && small.last == 9, "концы включены (10 кадров)")
    // Отбрасывание подряд не происходит: 5 позиций из 10 равномерны.
    let smallGaps = (1..<small.count).map { small[$0] - small[$0 - 1] }
    expect(smallGaps.max()! <= 3, "максимальный шаг ≤ 3 (равномерно)")

    // targetCount >= total → все индексы.
    let all = resampleSegmentIndices(total: 5, targetCount: 9)
    expectEqual(all, [0, 1, 2, 3, 4], "targetCount ≥ total → все индексы")

    // Граничные случаи.
    expectEqual(resampleSegmentIndices(total: 0, targetCount: 5), [], "пустой сегмент")
    expectEqual(resampleSegmentIndices(total: 1, targetCount: 5), [0], "один кадр")
    expectEqual(resampleSegmentIndices(total: 100, targetCount: 1), [0], "targetCount < 2")
}

func testMotionPairs() {
    section("Motion: обучающие пары окон")
    let pairs = motionPairs(frameCount: 16, window: 8, stride: 4)
    expect(!pairs.isEmpty, "пары есть")
    for p in pairs {
        expect(p.a < p.k && p.k < p.b, "a < k < b для пары (\(p.a), \(p.b), \(p.k))")
        expect(p.b - p.a == 7, "длина окна 8 (b−a=7)")
    }
    // Окна: [0,7], [4,11], [8,15] → 6+6+6 = 18 пар.
    expectEqual(pairs.count, 18, "3 окна × 6 внутренних кадров = 18 пар (получено \(pairs.count))")
    // t ∈ (0,1).
    for p in pairs {
        expect(p.t > 0 && p.t < 1, "t строго между 0 и 1 (\(p.t))")
    }
    // Слишком короткая серия → пусто.
    expectEqual(motionPairs(frameCount: 3, window: 8, stride: 1), [], "короткая серия → пусто")
    // Минимальное окно.
    let minPairs = motionPairs(frameCount: 4, window: 3, stride: 1)
    expectEqual(minPairs.count, 2, "окно 3: пары (0,2,1) и (1,3,2)")
    expectEqual(minPairs.first, MotionPair(a: 0, b: 2, k: 1), "первая пара (0,2,1)")
    expectEqual(minPairs.last, MotionPair(a: 1, b: 3, k: 2), "последняя пара (1,3,2)")
}

func testMotionMLPShape() {
    section("MotionMLP: форма и градиенты")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let latentDim = 32
    let net = MotionMLPGPU(engine: engine, latentDim: latentDim, dimEmb: 8, hidden: 24)
    var rng = SplitMix64(seed: 3)
    let zA = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
    let zB = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
    let zK = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
    let base = interpolateScalars(zA, zB, t: 0.5)
    let emb = motionTimeEmbedding(0.5, dim: 8)

    session.begin()
    let xT = GPUTensor(engine: engine, data: zA + zB, rows: 1)
    let baseT = GPUTensor(engine: engine, data: base, rows: 1)
    let target = GPUTensor(engine: engine, data: zK, rows: 1)
    let (pred, cache) = net.forwardCached(session, xT: xT, emb: emb, base: baseT)
    let grads = net.backward(session, cache, target: target)
    _ = session.end()

    expectEqual(pred.toArray().count, latentDim, "pred: latentDim значений")
    expectEqual(grads.count, 6, "6 тензоров градиентов")
    expect(grads.allSatisfy { $0.n > 0 }, "все градиенты непустые")
}

func testMotionMLPNumericGrads() {
    section("MotionMLP: численная сверка градиентов")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let latentDim = 24
    let seed: UInt64 = 0x1234_5678_9ABC_DEF1
    var rng = SplitMix64(seed: 21)
    let zA = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
    let zB = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
    let zK = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
    let base = interpolateScalars(zA, zB, t: 0.7)
    let emb = motionTimeEmbedding(0.7, dim: 8)
    let xArr = zA + zB

    let net = MotionMLPGPU(engine: engine, latentDim: latentDim, dimEmb: 8,
                           hidden: 24, seed: seed)
    let session = GpuSession(engine)
    session.begin()
    let xT = GPUTensor(engine: engine, data: xArr, rows: 1)
    let baseT = GPUTensor(engine: engine, data: base, rows: 1)
    let target = GPUTensor(engine: engine, data: zK, rows: 1)
    let (_, cache) = net.forwardCached(session, xT: xT, emb: emb, base: baseT)
    let grads = net.backward(session, cache, target: target)
    _ = session.end()

    func fwd(_ mats: [Matrix]) -> Double {
        let v = MotionMLPGPU(engine: engine, latentDim: latentDim, dimEmb: 8,
                             hidden: 24, seed: seed)
        try? v.load(from: mats)
        let s2 = GpuSession(engine)
        s2.begin()
        let x2 = GPUTensor(engine: engine, data: xArr, rows: 1)
        let b2 = GPUTensor(engine: engine, data: base, rows: 1)
        let (p, _) = v.forwardCached(s2, xT: x2, emb: emb, base: b2)
        _ = s2.end()
        let pa = p.toArray()
        var sum = 0.0
        for i in 0..<latentDim {
            let d = Double(pa[i] - zK[i])
            sum += d * d
        }
        return sum / Double(latentDim)
    }

    // Выходной слой l3 инициализирован нулями (residual-старт): его градиент
    // на старте ~0. Поэтому сначала проверяем l1/l2, а для l3 — с ненулевыми
    // весами (модифицируем матрицу и повторяем forward/backward).
    func checkGrad(_ gradArr: [Float], _ mats: [Matrix], _ idx: Int, _ wi: Int,
                   _ label: String) -> Bool {
        let analytic = Double(gradArr[wi])
        guard abs(analytic) > 1e-6 else { return true }
        let eps = 1e-4
        var mp = mats; mp[idx].data[wi] += Float(eps)
        var mm = mats; mm[idx].data[wi] -= Float(eps)
        let numeric = (fwd(mp) - fwd(mm)) / (2 * Double(eps))
        let sameSign = (analytic > 0) == (numeric > 0)
        let ratio = abs(analytic) / max(1e-12, abs(numeric))
        let ok = sameSign && ratio > 0.5 && ratio < 2.0
        expect(ok,
               String(format: "градиент[%d][%d] %@: аналит %.4e числ %.4e (знак %@, отношение %.2f)",
                      idx, wi, label, analytic, numeric, sameSign ? "✓" : "✗", ratio))
        return ok
    }

    let mats = net.matrices
    var fails = 0
    if !checkGrad(grads[0].toArray(), mats, 0, 5, "Wx") { fails += 1 }
    if !checkGrad(grads[2].toArray(), mats, 2, 9, "W2") { fails += 1 }
    // l3: устанавливаем ненулевые веса и повторяем forward/backward.
    var mats3 = mats
    for j in 0..<mats3[4].data.count where j % 47 == 0 {
        mats3[4].data[j] += 0.3
    }
    let net3 = MotionMLPGPU(engine: engine, latentDim: latentDim, dimEmb: 8,
                            hidden: 24, seed: seed)
    try? net3.load(from: mats3)
    net3.latentMean = net.latentMean
    net3.latentScale = net.latentScale
    let s3 = GpuSession(engine)
    s3.begin()
    let x3 = GPUTensor(engine: engine, data: xArr, rows: 1)
    let b3 = GPUTensor(engine: engine, data: base, rows: 1)
    let t3 = GPUTensor(engine: engine, data: zK, rows: 1)
    let (_, c3) = net3.forwardCached(s3, xT: x3, emb: emb, base: b3)
    let g3 = net3.backward(s3, c3, target: t3)
    _ = s3.end()
    if !checkGrad(g3[4].toArray(), mats3, 4, 5, "W3") { fails += 1 }
    expect(fails == 0, "все сильные сигналы сходятся")
}

func testMotionMLPTrainOverfit() {
    section("MotionMLP: обучение снижает loss")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let latentDim = 32
    var dataset: [(a: [Float], b: [Float], k: [Float], t: Double)] = []
    for k in 0..<16 {
        var rng = SplitMix64(seed: UInt64(3000 + k))
        let a = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
        let b = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
        let t = Double(k) / 15.0
        // Цель нелинейно зависит от t — не под силу чистому lerp.
        let kk = (0..<latentDim).map { i in
            a[i] + (b[i] - a[i]) * Float(t) + 0.3 * Float(sin(Double(i) * t * 3))
        }
        dataset.append((a: a, b: b, k: kk, t: t))
    }
    let net = MotionMLPGPU(engine: engine, latentDim: latentDim, dimEmb: 8, hidden: 48)
    var firstLoss = Double.infinity
    var lastLoss = Double.infinity
    for step in 1...800 {
        var sumLoss = 0.0
        for sample in dataset {
            let base = interpolateScalars(sample.a, sample.b, t: sample.t)
            let emb = motionTimeEmbedding(sample.t, dim: 8)
            session.begin()
            let xT = GPUTensor(engine: engine, data: sample.a + sample.b, rows: 1)
            let baseT = GPUTensor(engine: engine, data: base, rows: 1)
            let target = GPUTensor(engine: engine, data: sample.k, rows: 1)
            let (pred, cache) = net.forwardCached(session, xT: xT, emb: emb,
                                                   base: baseT)
            let grads = net.backward(session, cache, target: target)
            sanitizeGrads(grads, maxNorm: 10.0, preScale: 1.0 / Float(dataset.count))
            net.applyAdam(session, grads, t: step, lr: 0.002)
            _ = session.end()
            let pa = pred.toArray()
            var mse = 0.0
            for i in 0..<latentDim {
                let d = Double(pa[i] - sample.k[i])
                mse += d * d
            }
            sumLoss += mse / Double(latentDim)
        }
        let avg = sumLoss / Double(dataset.count)
        if step == 1 { firstLoss = avg }
        lastLoss = avg
    }
    print(String(format: "      MotionMLP overfit: loss %.5f → %.5f", firstLoss, lastLoss))
    expect(lastLoss < firstLoss, "обучение снижает loss")
    expect(lastLoss < firstLoss * 0.3, "значимое снижение (до 30% старта)")
}

func testMotionCheckpointRoundTrip() {
    section("MotionMLP: чекпоинт round-trip")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let net = MotionMLPGPU(engine: engine, latentDim: 40, dimEmb: 8, hidden: 32)
    let path = NSTemporaryDirectory() + "/yume_motion_test.safetensors"
    do {
        try saveMotionCheckpoint(net: net, to: path)
    } catch {
        expect(false, "сохранение чекпоинта: \(error)")
        return
    }
    guard let loaded = try? loadMotionCheckpoint(from: path) else {
        expect(false, "чтение чекпоинта")
        return
    }
    expectEqual(loaded.meta.latentDim, 40, "latentDim в метаданных")
    expectEqual(loaded.meta.hidden, 32, "hidden в метаданных")
    expectEqual(loaded.meta.latentScale, 1, "latentScale по умолчанию 1")
    expectEqual(loaded.mats.count, 6, "6 матриц")
    expectEqual(loaded.mats.map { $0.rows }, net.matrices.map { $0.rows },
                "формы матриц совпадают")
    // Загрузка в новую сеть даёт тот же forward.
    var rng = SplitMix64(seed: 9)
    let zA = (0..<40).map { _ in Float.random(in: -1...1, using: &rng) }
    let zB = (0..<40).map { _ in Float.random(in: -1...1, using: &rng) }
    let base = interpolateScalars(zA, zB, t: 0.25)
    let emb = motionTimeEmbedding(0.25, dim: 8)
    let net2 = MotionMLPGPU(engine: engine, latentDim: 40, dimEmb: 8, hidden: 32)
    try? net2.load(from: loaded.mats)
    let session = GpuSession(engine)
    session.begin()
    let xT1 = GPUTensor(engine: engine, data: zA + zB, rows: 1)
    let b1 = GPUTensor(engine: engine, data: base, rows: 1)
    let (p1, _) = net.forwardCached(session, xT: xT1, emb: emb, base: b1)
    let xT2 = GPUTensor(engine: engine, data: zA + zB, rows: 1)
    let b2 = GPUTensor(engine: engine, data: base, rows: 1)
    let (p2, _) = net2.forwardCached(session, xT: xT2, emb: emb, base: b2)
    _ = session.end()
    let a1 = p1.toArray(), a2 = p2.toArray()
    var same = true
    for i in 0..<a1.count where abs(a1[i] - a2[i]) > 1e-4 { same = false }
    expect(same, "forward после round-trip совпадает")
    try? FileManager.default.removeItem(atPath: path)
}

/// E2E: обучение сети движения на реальных кадрах улучшает предсказание
/// промежуточных латентов против чистого lerp на главном окне A→B.
func testMotionTrainingImprovesPrediction() {
    section("Motion E2E: обучение улучшает предсказание против lerp (hold-out)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let fm = FileManager.default
    let framesDir = "/tmp/yume_frames"
    guard let files = try? fm.contentsOfDirectory(atPath: framesDir) else {
        skip("нет /tmp/yume_frames — сначала извлеките кадры реального видео"); return
    }
    let jpgs = files.filter { $0.hasSuffix(".jpg") }.sorted()
    guard jpgs.count >= 48 else {
        skip("мало реальных кадров (\(jpgs.count))"); return
    }
    let ckpt = fm.homeDirectoryForCurrentUser.path + "/YumeLayers/checkpoints/conv.safetensors"
    guard fm.fileExists(atPath: ckpt),
          let st = try? loadStateWithMeta(path: ckpt) else {
        skip("нет VAE-чекпоинта / не читается"); return
    }
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine, latentChannels: st.latentChannels,
                         width: st.modelWidth, residual: st.archVersion >= 2,
                         groupNorm: st.archVersion >= 3)
    guard (try? vae.load(from: st.mats)) != nil else {
        skip("чекпоинт не грузится"); return
    }
    let pw = 80, ph = 56
    let count = min(jpgs.count, 64)
    var latents: [Int: [Float]] = [:]
    for i in 0..<count {
        let path = (framesDir as NSString).appendingPathComponent(jpgs[i])
        guard let rgb = loadImageRGB(path: path, w: pw, h: ph),
              let z = try? encodeLatent(vae: vae, image: rgb, w: pw, h: ph) else { continue }
        latents[i] = z
    }
    guard latents.count >= 48 else { skip("не удалось закодировать кадры"); return }
    let trainFrameCount = 56
    let holdSize = count - trainFrameCount
    guard holdSize >= 6 else { skip("мало hold-out кадров"); return }

    // Обучаем на крупных окнах (близких к целевой длине интервала A→B):
    // сеть должна выучить динамику именно этого масштаба.
    let trainPairs = (motionBuildPairs(0, trainFrameCount, 24, 12)
        + motionBuildPairs(0, trainFrameCount, 48, 24)
        + motionBuildPairs(0, trainFrameCount, 56, 56))
        .filter { latents[$0.a] != nil && latents[$0.b] != nil && latents[$0.k] != nil }
    guard !trainPairs.isEmpty else { skip("нет обучающих пар"); return }

    let net = MotionMLPGPU(engine: engine, latentDim: latents.values.first!.count,
                           dimEmb: 8, hidden: 96)
    motionNormalizeNet(net, latents: latents, range: 0..<trainFrameCount)
    let session = GpuSession(engine)
    let mseBuf = engine.device.makeBuffer(length: 4, options: .storageModeShared)
    motionTrainLoop(net: net, session: session, pairs: trainPairs,
                    latents: latents, mseBuf: mseBuf,
                    steps: 4000, lr: 0.0005, seed: 0xABCD_0000_0000_5555)

    // Контроль: обучалась ли сеть (метрика на самих train-парах).
    var trainLerp = 0.0
    var trainNet = 0.0
    for pair in trainPairs {
        let zA = latents[pair.a]!, zB = latents[pair.b]!, zK = latents[pair.k]!
        trainLerp += motionMSE(interpolateScalars(zA, zB, t: pair.t), zK)
        trainNet += motionMSE(predictMotionLatent(net: net, zA: zA, zB: zB,
                                                  t: pair.t), zK)
    }
    trainLerp /= Double(trainPairs.count)
    trainNet /= Double(trainPairs.count)
    print(String(format: "      Motion E2E train: lerp MSE %.5f · сеть MSE %.5f (улучшение %.1f%%)",
                 trainLerp, trainNet, (1 - trainNet / max(trainLerp, 1e-12)) * 100))

    // Главная метрика: окно (0, 55) — все промежуточные кадры между опорами.
    let main = motionEvalWindow(net: net, latents: latents, a: 0,
                                b: trainFrameCount - 1)
    let casc = motionCascadeWindow(net: net, latents: latents, a: 0,
                                   b: trainFrameCount - 1)
    print(String(format: "      Motion E2E окно (0,%d): lerp MSE %.5f · сеть MSE %.5f (улучшение %.1f%%)",
                 trainFrameCount - 1, main.lerp, main.net,
                 (1 - main.net / max(main.lerp, 1e-12)) * 100))
    print(String(format: "      Motion E2E каскад (0,%d): сеть MSE %.5f (улучшение %.1f%%)",
                 trainFrameCount - 1, casc,
                 (1 - casc / max(main.lerp, 1e-12)) * 100))
    expect(main.net < main.lerp, "сеть точнее lerp на главном окне")
    expect(casc < main.lerp, "каскад точнее lerp на главном окне")
    if main.lerp > 0.0005 {
        expect(min(main.net, casc) < main.lerp * 0.85,
               "улучшение ≥ 15% при существенном движении")
    }

    // Информационная проверка обобщения на коротких hold-out окнах (56..<64).
    let holdPairs = motionBuildPairs(trainFrameCount, holdSize, 8, 2)
        .filter { latents[$0.a] != nil && latents[$0.b] != nil && latents[$0.k] != nil }
    if !holdPairs.isEmpty {
        var l = 0.0
        var n = 0.0
        for pair in holdPairs {
            let zA = latents[pair.a]!, zB = latents[pair.b]!, zK = latents[pair.k]!
            l += motionMSE(interpolateScalars(zA, zB, t: pair.t), zK)
            n += motionMSE(predictMotionLatent(net: net, zA: zA, zB: zB,
                                               t: pair.t), zK)
        }
        l /= Double(holdPairs.count)
        n /= Double(holdPairs.count)
        print(String(format: "      Motion E2E hold-out (короткие окна, вне масштаба): lerp MSE %.5f · сеть MSE %.5f",
                     l, n))
    }
}

/// Пиксельный E2E: насколько последовательность, сгенерированная сетью
/// движения, близка к реальному видео — относительно потолка, заданного
/// качеством реконструкции VAE (recon(gt) vs gt).
func testMotionPixelQualityVsCeiling() {
    section("Motion пиксели: движение vs потолок реконструкции VAE")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let fm = FileManager.default
    let framesDir = "/tmp/yume_frames"
    guard let files = try? fm.contentsOfDirectory(atPath: framesDir) else {
        skip("нет /tmp/yume_frames"); return
    }
    let jpgs = files.filter { $0.hasSuffix(".jpg") }.sorted()
    guard jpgs.count >= 56 else { skip("мало кадров"); return }
    let ckpt = fm.homeDirectoryForCurrentUser.path + "/YumeLayers/checkpoints/conv.safetensors"
    guard fm.fileExists(atPath: ckpt),
          let st = try? loadStateWithMeta(path: ckpt) else { skip("нет VAE"); return }
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine, latentChannels: st.latentChannels,
                         width: st.modelWidth, residual: st.archVersion >= 2,
                         groupNorm: st.archVersion >= 3)
    guard (try? vae.load(from: st.mats)) != nil else { skip("чекпоинт не грузится"); return }
    let pw = 80, ph = 56
    var latents: [Int: [Float]] = [:]
    var frames: [Int: [Float]] = [:]
    for i in 0..<56 {
        guard let rgb = loadImageRGB(path: (framesDir as NSString).appendingPathComponent(jpgs[i]),
                                     w: pw, h: ph),
              let z = try? encodeLatent(vae: vae, image: rgb, w: pw, h: ph) else { continue }
        latents[i] = z
        frames[i] = rgb
    }
    guard latents.count >= 56 else { skip("не удалось закодировать"); return }
    let net = MotionMLPGPU(engine: engine, latentDim: latents.values.first!.count,
                           dimEmb: 8, hidden: 96)
    var sum = 0.0, sumSq = 0.0, nElem = 0
    for i in 0..<56 {
        for v in latents[i]! { sum += Double(v); sumSq += Double(v) * Double(v); nElem += 1 }
    }
    let mean = Float(sum / Double(nElem))
    net.latentMean = mean
    net.latentScale = max(Float(sqrt(max((sumSq / Double(nElem)) - Double(mean) * Double(mean), 1e-6))), 1e-3)
    let pairs = (motionPairs(frameCount: 56, window: 56, stride: 56)
        + motionPairs(frameCount: 56, window: 48, stride: 24)
        + motionPairs(frameCount: 56, window: 24, stride: 12))
        .filter { latents[$0.k] != nil && frames[$0.k] != nil }
    let session = GpuSession(engine)
    let mseBuf = engine.device.makeBuffer(length: 4, options: .storageModeShared)
    var rng = SplitMix64(seed: 0x7777_0000_0000_1111)
    for step in 1...4000 {
        let p = pairs[Int(rng.next() % UInt64(pairs.count))]
        _ = runMotionStep(net: net, session: session, pair: p,
                          latents: latents, mseBuf: mseBuf,
                          step: step, lr: 0.0005)
    }
    // Метрики на окне (0, 55).
    var ceilSum = 0.0
    var motionSum = 0.0
    var lerpSum = 0.0
    var count = 0
    for k in 1..<55 {
        guard let zK = latents[k], let gt = frames[k] else { continue }
        // Потолок: реконструкция реального кадра.
        guard let recon = try? decodeLatent(vae: vae, latent: zK, w: pw, h: ph) else { continue }
        ceilSum += Double(psnr(reference: gt, candidate: recon))
        // Сеть движения.
        let pred = predictMotionLatent(net: net, zA: latents[0]!, zB: latents[55]!,
                                       t: Double(k) / 55.0)
        if let predFrame = try? decodeLatent(vae: vae, latent: pred, w: pw, h: ph) {
            motionSum += Double(psnr(reference: gt, candidate: predFrame))
        }
        // Lerp baseline.
        let lerpZ = interpolateScalars(latents[0]!, latents[55]!, t: Double(k) / 55.0)
        if let lerpFrame = try? decodeLatent(vae: vae, latent: lerpZ, w: pw, h: ph) {
            lerpSum += Double(psnr(reference: gt, candidate: lerpFrame))
        }
        count += 1
    }
    let ceiling = ceilSum / Double(max(count, 1))
    let motion = motionSum / Double(max(count, 1))
    let lerp = lerpSum / Double(max(count, 1))
    print(String(format: "      Motion пиксели: потолок VAE %.2f дБ · lerp %.2f дБ · сеть движения %.2f дБ (разрыв к потолку %.2f дБ)",
                 ceiling, lerp, motion, ceiling - motion))
    expect(count >= 40, "оценено достаточно кадров (\(count))")
    expect(motion >= lerp - 0.3, "сеть не хуже lerp в пикселях")
    expect(lerp < ceiling - 3.0, "в сегменте есть движение, заметное для lerp")
}

/// Замер потолка реконструкции VAE на нескольких разрешениях: показывает,
/// какой выигрыш даст подъём рабочего разрешения перед тюнингом движения.
func testMotionCeilingAcrossResolutions() {
    section("Motion: потолок реконструкции VAE по разрешениям")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let fm = FileManager.default
    let framesDir = "/tmp/yume_frames"
    guard let files = try? fm.contentsOfDirectory(atPath: framesDir) else {
        skip("нет /tmp/yume_frames"); return
    }
    let jpgs = files.filter { $0.hasSuffix(".jpg") }.sorted()
    guard jpgs.count >= 8 else { skip("мало кадров"); return }
    let ckpt = fm.homeDirectoryForCurrentUser.path + "/YumeLayers/checkpoints/conv.safetensors"
    guard fm.fileExists(atPath: ckpt),
          let st = try? loadStateWithMeta(path: ckpt) else { skip("нет VAE"); return }
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine, latentChannels: st.latentChannels,
                         width: st.modelWidth, residual: st.archVersion >= 2,
                         groupNorm: st.archVersion >= 3)
    guard (try? vae.load(from: st.mats)) != nil else { skip("чекпоинт не грузится"); return }
    func mseRegion(_ a: [Float], _ b: [Float], width: Int, height: Int,
                   cropW: Int) -> Double {
        var s = 0.0
        for y in 0..<height {
            for x in 0..<cropW {
                let base = (y * width + x) * 3
                for c in 0..<3 {
                    let d = Double(a[base + c] - b[base + c])
                    s += d * d
                }
            }
        }
        return s / Double(cropW * height * 3)
    }
    let sizes: [(Int, Int)] = [(80, 56), (128, 72), (256, 144), (856, 480)]
    for (w, h) in sizes {
        var sum = 0.0
        var count = 0
        var skipReason = ""
        let cropW = w == 856 ? 854 : w
        for i in 0..<min(6, jpgs.count) {
            let path = (framesDir as NSString).appendingPathComponent(jpgs[i])
            guard let rgb = loadImageRGB(path: path, w: w, h: h) else { skipReason = "load"; continue }
            guard let z = try? encodeLatent(vae: vae, image: rgb, w: w, h: h) else { skipReason = "encode"; continue }
            guard let recon = try? decodeLatent(vae: vae, latent: z, w: w, h: h) else { skipReason = "decode"; continue }
            let mse = mseRegion(rgb, recon, width: w, height: h, cropW: cropW)
            sum += -10.0 * log10(max(mse, 1e-12))
            count += 1
        }
        if count > 0 {
            print(String(format: "      потолок VAE %d×%d (crop %d): %.2f дБ (n=%d)", w, h, cropW, sum / Double(count), count))
        } else {
            print("      потолок VAE \(w)×\(h): пропущено (последний сбой: \(skipReason))")
        }
    }
}

/// Диагностика: пошаговое выполнение пиксельного шага (находит падающий этап).
/// Изолированная проверка backwardFromGrad (внешний градиент) — маленькие размеры.
func testBackwardFromGradDirect() {
    section("MotionMLP: backwardFromGrad (внешний dOut)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let latentDim = 2240
    let net = MotionMLPGPU(engine: engine, latentDim: latentDim, dimEmb: 8, hidden: 96)
    var rng = SplitMix64(seed: 11)
    let zA = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
    let zB = (0..<latentDim).map { _ in Float.random(in: -1...1, using: &rng) }
    let base = interpolateScalars(zA, zB, t: 0.4)
    let emb = motionTimeEmbedding(0.4, dim: 8)
    let xT = zA + zB
    let s1 = GpuSession(engine)
    s1.begin()
    let x1 = GPUTensor(engine: engine, data: xT, rows: 1)
    let b1 = GPUTensor(engine: engine, data: base, rows: 1)
    let (_, cache) = net.forwardCached(s1, xT: x1, emb: emb, base: b1)
    _ = s1.end()
    let s2 = GpuSession(engine)
    s2.begin()
    let dOut = GPUTensor(engine: engine,
                         data: (0..<latentDim).map { _ in Float.random(in: -0.01...0.01, using: &rng) },
                         rows: 1)
    let grads = net.backwardFromGrad(s2, cache, dOut: dOut)
    let ms = s2.end()
    expect(grads.count == 6, "6 градиентов")
    expect(grads.allSatisfy { $0.n > 0 }, "градиенты непустые")
    expect(ms >= 0, "сессия завершилась")
    print("      backwardFromGrad: OK (\(ms) мс)")
}

func testMotionTimeEmbedding() {
    section("Motion: временное эмбеддинг")
    let emb0 = motionTimeEmbedding(0, dim: 8)
    expectEqual(emb0.count, 8, "размерность 8")
    // t=0: sin(0)=0, cos(0)=1 на чётных/нечётных позициях.
    expectClose(Double(emb0[0]), 0, 1e-6, "emb[0] = sin(0) = 0")
    expectClose(Double(emb0[1]), 1, 1e-6, "emb[1] = cos(0) = 1")
    let emb1 = motionTimeEmbedding(1, dim: 8)
    expect(emb1[0] != emb0[0] || emb1[1] != emb0[1], "t=1 отличается от t=0")
    // Ограничение t ∈ [0,1].
    let embClamp = motionTimeEmbedding(5, dim: 8)
    expectEqual(embClamp, emb1, "t>1 ограничивается до 1")
}
