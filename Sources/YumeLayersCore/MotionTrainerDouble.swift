// YumeLayers — обучение сети движения в двойной точности (CPU, Float64).
//
// Полные латенты 854×480 (205k измерений) на GPU-Float32 дают NaN; на CPU
// в Double обучение стабильно. VAE encode/decode остаётся на GPU.
import Foundation

// ---------- Чекпоинт Double-сети ----------

/// Сохранение чекпоинта Double-сети (матрицы конвертируются в Float32,
/// как в safetensors-формате; метка backend="cpu64").
public func saveMotionCheckpointDouble(net: MotionMLPDouble, to path: String) throws {
    let mats = net.matrices
    var dict: [String: Any] = [:]
    var offset = 0
    var order: [String] = []
    for (i, m) in mats.enumerated() {
        guard m.data.allSatisfy({ $0.isFinite }) else {
            throw NSError(domain: "YumeMotionDouble", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "NaN/Inf в весах Double-сети движения p\(i)"])
        }
        let name = "p\(i)"
        order.append(name)
        let bytes = m.data.count * 4
        dict[name] = ["dtype": "F32",
                      "shape": [m.rows, m.cols],
                      "data_offsets": [offset, offset + bytes]]
        offset += bytes
    }
    dict["format"] = "YumeLayers Motion"
    dict["order"] = order
    dict["latentDim"] = net.latentDim
    dict["dimEmb"] = net.dimEmb
    dict["hidden"] = net.hidden
    dict["archVersion"] = 5
    dict["backend"] = "cpu64"
    dict["latentMean"] = Float(net.latentMean)
    dict["latentScale"] = Float(net.latentScale)
    dict["latentH3"] = 0
    dict["latentW3"] = 0
    dict["poolBlock"] = 1
    let header = try JSONSerialization.data(withJSONObject: dict)
    var headerLen = UInt64(header.count).littleEndian
    var file = Data()
    file.append(Data(bytes: &headerLen, count: 8))
    file.append(header)
    for m in mats {
        let bytes = m.data.map { $0.bitPattern }
        var block = Data()
        for b in bytes {
            var v = b.littleEndian
            block.append(Data(bytes: &v, count: 4))
        }
        file.append(block)
    }
    try file.write(to: URL(fileURLWithPath: path))
}

/// Чтение чекпоинта движения (GPU или Double): длина заголовка по offset 0.
public func loadMotionCheckpointAny(from path: String)
    throws -> (mats: [Matrix], meta: MotionCheckpointMeta)? {
    let url = URL(fileURLWithPath: path)
    guard let file = try? Data(contentsOf: url), file.count >= 8 else { return nil }
    let headerLen = file.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self) }
    let headerSize = Int(headerLen.littleEndian)
    guard headerSize > 0, 8 + headerSize <= file.count else { return nil }
    let headerData = file.subdata(in: 8..<(8 + headerSize))
    guard let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any],
          (obj["format"] as? String) == "YumeLayers Motion",
          let order = obj["order"] as? [String] else { return nil }
    var mats: [Matrix] = []
    for name in order {
        guard let info = obj[name] as? [String: Any],
              let shape = info["shape"] as? [Int], shape.count == 2,
              let offs = info["data_offsets"] as? [Int], offs.count == 2 else { return nil }
        let rows = shape[0], cols = shape[1]
        let start = 8 + headerSize + offs[0]
        let end = 8 + headerSize + offs[1]
        guard end <= file.count, end > start else { return nil }
        let raw = file.subdata(in: start..<end)
        let count = rows * cols
        guard raw.count == count * 4 else { return nil }
        var data = [Float](repeating: 0, count: count)
        _ = raw.withUnsafeBytes { src in
            data.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: src)
            }
        }
        mats.append(Matrix(data, rows: rows))
    }
    guard let latentDim = obj["latentDim"] as? Int,
          let dimEmb = obj["dimEmb"] as? Int,
          let hidden = obj["hidden"] as? Int else { return nil }
    let v = (obj["archVersion"] as? Int) ?? 5
    let mean = (obj["latentMean"] as? Float) ?? 0
    let scale = (obj["latentScale"] as? Float) ?? 1
    let h3 = (obj["latentH3"] as? Int) ?? 0
    let w3 = (obj["latentW3"] as? Int) ?? 0
    let block = (obj["poolBlock"] as? Int) ?? 1
    let backend = (obj["backend"] as? String) ?? "gpu"
    let meta = MotionCheckpointMeta(latentDim: latentDim, dimEmb: dimEmb,
                                    hidden: hidden, archVersion: v,
                                    latentMean: mean, latentScale: scale,
                                    latentH3: h3, latentW3: w3, poolBlock: block,
                                    backend: backend)
    return (mats, meta)
}

// ---------- Обучение ----------

/// Один шаг обучения Double-сети на паре (A, B, K).
func runMotionStepDouble(net: MotionMLPDouble, pair: MotionPair,
                         latents: [Int: [Double]], t: Int, lr: Double) -> Double {
    let zA = latents[pair.a]!
    let zB = latents[pair.b]!
    let zK = latents[pair.k]!
    let tt = pair.t
    let mean = net.latentMean, scale = net.latentScale
    let na = zA.map { ($0 - mean) / scale }
    let nb = zB.map { ($0 - mean) / scale }
    let nk = zK.map { ($0 - mean) / scale }
    let base = (0..<na.count).map { na[$0] + (nb[$0] - na[$0]) * tt }
    let emb = motionTimeEmbedding(tt, dim: net.dimEmb).map { Double($0) }
    let xT = na + nb
    let (pred, cache) = net.forward(xT: xT, emb: emb, base: base)
    var mse = 0.0
    for k in 0..<pred.count {
        let d = pred[k] - nk[k]
        mse += d * d
    }
    mse /= Double(max(pred.count, 1))
    var dOut = [Double](repeating: 0, count: pred.count)
    for k in 0..<pred.count {
        dOut[k] = 2 * (pred[k] - nk[k]) / Double(max(pred.count, 1))
    }
    let g = net.backward(cache, dOut: dOut)
    net.applyAdam(g, t: t, lr: lr)
    return mse
}

/// Обучение Double-сети движения на полных латентах (без пулинга).
public func trainMotionMLPDouble(memoryFrames: [MemoryFrame],
                                 vae: ConvVAEGPU,
                                 pw: Int, ph: Int,
                                 steps: Int = 600,
                                 lr: Double = 0.0005,
                                 window: Int = 24,
                                 maxEncodedFrames: Int = 512,
                                 checkpointPath: String,
                                 onProgress: ((Int, Int) -> Void)? = nil) throws {
    guard !memoryFrames.isEmpty else {
        throw NSError(domain: "YumeMotionDouble", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "нет кадров для обучения движения"])
    }
    guard steps > 0 else { return }
    let runs = motionFrameRuns(memoryFrames)
    let total = memoryFrames.count
    let targetN = min(total, maxEncodedFrames)
    let selected = resampleSegmentIndices(total: total, targetCount: targetN)
    let full = encodeMotionLatents(frames: memoryFrames, selected: selected,
                                   vae: vae, pw: pw, ph: ph)
    guard !full.isEmpty else {
        throw NSError(domain: "YumeMotionDouble", code: 3,
                      userInfo: [NSLocalizedDescriptionKey: "не удалось закодировать ни одного кадра"])
    }
    var latents: [Int: [Double]] = [:]
    for (idx, z) in full { latents[idx] = z.map { Double($0) } }
    let pairs = buildMotionTrainingPairs(runs: runs, latents: full,
                                         window: window)
    guard !pairs.isEmpty else {
        throw NSError(domain: "YumeMotionDouble", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "недостаточно кадров одного источника для окон движения"])
    }
    let latentDim = latents.values.first!.count
    let net = MotionMLPDouble(latentDim: latentDim)
    var sum = 0.0
    var sumSq = 0.0
    var count = 0
    for z in latents.values {
        for v in z {
            sum += v
            sumSq += v * v
            count += 1
        }
    }
    if count > 0 {
        let mean = sum / Double(count)
        net.latentMean = mean
        net.latentScale = max(sqrt(max((sumSq / Double(count)) - mean * mean, 1e-12)), 1e-6)
    }
    var rng = SplitMix64(seed: 0x900D_0000_0000_4444)
    var lossSum = 0.0
    for step in 1...steps {
        if TrainingControl.shared.isCancelled { throw CancellationError() }
        let pair = pairs[Int(rng.next() % UInt64(pairs.count))]
        let loss = runMotionStepDouble(net: net, pair: pair, latents: latents,
                                       t: step, lr: lr)
        lossSum += loss
        if step % 50 == 0 {
            yumeLog(String(format: "Движение (Float64): шаг %d/%d, loss %.5f (MSE латента)",
                           step, steps, lossSum / Double(step)))
        }
        onProgress?(step, steps)
    }
    if !lossSum.isFinite {
        throw NSError(domain: "YumeMotionDouble", code: 5,
                      userInfo: [NSLocalizedDescriptionKey: "training дал NaN"])
    }
    try saveMotionCheckpointDouble(net: net, to: checkpointPath)
    yumeLog(String(format: "✅ Сеть движения (Float64, CPU) обучена: %d шагов, средний loss %.5f → чекпоинт %@",
                   steps, lossSum / Double(steps), checkpointPath))
}

/// Загружает VAE из чекпоинта и обучает Double-сеть движения.
public func trainMotionMLPDoubleFromCheckpoint(memoryFrames: [MemoryFrame],
                                               vaeCheckpointPath: String,
                                               pw: Int, ph: Int,
                                               steps: Int = 600,
                                               lr: Double = 0.0005,
                                               window: Int = 24,
                                               maxEncodedFrames: Int = 512,
                                               objectivePath: String,
                                               onProgress: ((Int, Int) -> Void)? = nil) throws {
    let st = try loadStateWithMeta(path: vaeCheckpointPath)
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine, latentChannels: st.latentChannels,
                         width: st.modelWidth, residual: st.archVersion >= 2,
                         groupNorm: st.archVersion >= 3)
    try vae.load(from: st.mats)
    try trainMotionMLPDouble(memoryFrames: memoryFrames, vae: vae,
                             pw: pw, ph: ph, steps: steps, lr: lr,
                             window: window, maxEncodedFrames: maxEncodedFrames,
                             checkpointPath: objectivePath, onProgress: onProgress)
}

// ---------- Инференс ----------

/// Предсказывает латент Double-сетью: lerp + residual, на CPU в Double.
public func predictMotionLatentDouble(net: MotionMLPDouble,
                                      zA: [Float], zB: [Float],
                                      t: Double) -> [Float] {
    let a = zA.map { Double($0) }
    let b = zB.map { Double($0) }
    let mean = net.latentMean, scale = net.latentScale
    let na = a.map { ($0 - mean) / scale }
    let nb = b.map { ($0 - mean) / scale }
    let base = (0..<na.count).map { na[$0] + (nb[$0] - na[$0]) * t }
    let emb = motionTimeEmbedding(t, dim: net.dimEmb).map { Double($0) }
    let xT = na + nb
    let (pred, _) = net.forward(xT: xT, emb: emb, base: base)
    guard pred.allSatisfy({ $0.isFinite }) else {
        return zA.enumerated().map { $0.element + (zB[$0.offset] - $0.element) * Float(t) }
    }
    return pred.map { Float($0 * scale + mean) }
}
