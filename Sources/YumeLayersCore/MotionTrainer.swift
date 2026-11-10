// YumeLayers — Фаза 4: обучение сети движения на парах реального видео.
//
// Итоговое обучение «движения»: после VAE модель учится воспроизводить
// промежуточные кадры между двумя опорными, используя реальные кадры
// видео как целевые (в латентном пространстве VAE). Чекпоинт — в формате
// safetensors (как у диффузии), метаданные содержат архитектуру сети.
import Foundation
import Metal

// ---------- Чекпоинт сети движения ----------

/// Метаданные чекпоинта сети движения.
public struct MotionCheckpointMeta: Equatable {
    public let latentDim: Int
    public let dimEmb: Int
    public let hidden: Int
    public let archVersion: Int
    /// Нормализация латентов: z' = (z − mean) / scale (скаляры, посчитаны
    /// по обучающему набору). Без неё значения mu VAE (порядка десятков)
    /// переполняют MLP и дают NaN.
    public let latentMean: Float
    public let latentScale: Float

    public init(latentDim: Int, dimEmb: Int, hidden: Int, archVersion: Int = 5,
                latentMean: Float = 0, latentScale: Float = 1) {
        self.latentDim = latentDim
        self.dimEmb = dimEmb
        self.hidden = hidden
        self.archVersion = archVersion
        self.latentMean = latentMean
        self.latentScale = latentScale > 0 ? latentScale : 1
    }
}

/// Сохранение чекпоинта сети движения (матрицы + метаданные).
public func saveMotionCheckpoint(net: MotionMLPGPU, to path: String) throws {
    let mats = net.matrices
    var dict: [String: Any] = [:]
    var offset = 0
    var order: [String] = []
    for (i, m) in mats.enumerated() {
        guard m.data.allSatisfy({ $0.isFinite }) else {
            throw NSError(domain: "YumeMotion", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "NaN/Inf в весах сети движения p\(i)"])
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
    dict["latentMean"] = net.latentMean
    dict["latentScale"] = net.latentScale
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

/// Результат чтения чекпоинта сети движения.
public struct MotionCheckpoint {
    public let mats: [Matrix]
    public let meta: MotionCheckpointMeta
}

/// Чтение чекпоинта сети движения. nil при повреждённых данных.
public func loadMotionCheckpoint(from path: String) throws -> MotionCheckpoint? {
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
    return MotionCheckpoint(mats: mats,
                            meta: MotionCheckpointMeta(latentDim: latentDim,
                                                       dimEmb: dimEmb,
                                                       hidden: hidden,
                                                       archVersion: v,
                                                       latentMean: mean,
                                                       latentScale: scale))
}

// ---------- Подготовка данных ----------

/// Непрерывные серии кадров одного источника (строгий порядок извлечения).
public func motionFrameRuns(_ frames: [MemoryFrame]) -> [[Int]] {
    var runs: [[Int]] = []
    var current: [Int] = []
    var currentSource = ""
    for (i, f) in frames.enumerated() {
        if f.source != currentSource, !current.isEmpty {
            runs.append(current)
            current = []
        }
        currentSource = f.source
        current.append(i)
    }
    if !current.isEmpty { runs.append(current) }
    return runs
}

/// Кодирует выбранные кадры в латенты VAE. Пропускает нечитаемые JPEG.
func encodeMotionLatents(frames: [MemoryFrame], selected: [Int],
                         vae: ConvVAEGPU, pw: Int, ph: Int) -> [Int: [Float]] {
    var latents: [Int: [Float]] = [:]
    for i in selected {
        guard i >= 0, i < frames.count,
              let rgb = loadImageRGB(data: frames[i].jpeg, w: pw, h: ph),
              let z = try? encodeLatent(vae: vae, image: rgb, w: pw, h: ph) else {
            continue
        }
        latents[i] = z
    }
    return latents
}

/// Строит обучающие пары по сериям: все окна × все внутренние кадры,
/// с фильтром по доступности латентов (Some кадры могли не закодироваться).
func buildMotionTrainingPairs(runs: [[Int]], latents: [Int: [Float]],
                              window: Int) -> [MotionPair] {
    var pairs: [MotionPair] = []
    for run in runs {
        guard run.count >= 3 else { continue }
        let w = min(max(3, window), run.count)
        let stride = max(1, w / 2)
        for p in motionPairs(frameCount: run.count, window: w, stride: stride) {
            let a = run[p.a], b = run[p.b], k = run[p.k]
            guard latents[a] != nil, latents[b] != nil, latents[k] != nil else { continue }
            pairs.append(MotionPair(a: a, b: b, k: k))
        }
    }
    return pairs
}

// ---------- Пиксельное обучение (PSNR-driven) ----------

/// Один шаг пиксельного обучения: предсказанный латент декодируется VAE,
/// MSE с целевым кадром считается в пикселях, градиент dL/dz идёт через
/// декодер (gradToLatent) обратно в сеть движения.
func runMotionStepPixel(net: MotionMLPGPU, vae: ConvVAEGPU,
                        session: GpuSession,
                        pair: MotionPair,
                        latents: [Int: [Float]],
                        frames: [Int: [Float]],
                        pw: Int, ph: Int,
                        mseBuf: MTLBuffer?,
                        step: Int, lr: Float) -> Double {
    let zA = latents[pair.a]!
    let zB = latents[pair.b]!
    let frameK = frames[pair.k]!
    let t = pair.t
    let emb = motionTimeEmbedding(t, dim: net.dimEmb)
    let na = normalizeLatent(zA, mean: net.latentMean, scale: net.latentScale)
    let nb = normalizeLatent(zB, mean: net.latentMean, scale: net.latentScale)
    let base = interpolateScalars(na, nb, t: t)
    let xT = na + nb
    let latentDim = zA.count
    let engine = session.engine
    let xTensor = GPUTensor(engine: engine, rows: 1, cols: xT.count)
    let baseTensor = GPUTensor(engine: engine, rows: 1, cols: base.count)
    let targetTensor = GPUTensor(engine: engine, rows: 3, cols: pw * ph)
    let meanTensor = GPUTensor(engine: engine,
                               data: [Float](repeating: net.latentMean, count: latentDim),
                               rows: 1)
    session.begin()
    session.upload(xTensor, from: xT)
    session.upload(baseTensor, from: base)
    session.upload(targetTensor, from: hwcToChw(frameK))
    let (pred, cache) = net.forwardCached(session, xT: xTensor, emb: emb,
                                         base: baseTensor)
    let zAbs = session.add(session.smul(net.latentScale, pred), meanTensor)
    let dec = vae.decode(session, z: zAbs, w: pw, h: ph)
    let mse = session.encodeMSE(recon: dec.recon, target: targetTensor, into: mseBuf)
    let dzAbs = vae.gradToLatent(session, dec, target: targetTensor, w: pw, h: ph)
    let dOut = session.smul(net.latentScale, dzAbs)
    let grads = net.backwardFromGrad(session, cache, dOut: dOut)
    let gpuMs = session.end()
    _ = gpuMs
    let loss = Double(mse.contents().assumingMemoryBound(to: Float.self)[0])
        / Double(max(pw * ph * 3, 1))
    sanitizeGrads(grads, maxNorm: 10.0)
    session.begin()
    net.applyAdam(session, grads, t: step, lr: lr)
    session.end()
    return loss
}

/// Обучение сети движения с пиксельной потерей (PSNR-driven). Кадры
/// декодируются в RGB один раз; на каждом шаге предсказание декодируется
/// VAE и сравнивается с реальным кадром в пикселях.
public func trainMotionMLPPixel(memoryFrames: [MemoryFrame],
                                vae: ConvVAEGPU,
                                pw: Int, ph: Int,
                                steps: Int = 400,
                                lr: Float = 0.001,
                                window: Int = 24,
                                maxEncodedFrames: Int = 128,
                                checkpointPath: String,
                                onProgress: ((Int, Int) -> Void)? = nil) throws {
    guard !memoryFrames.isEmpty else {
        throw NSError(domain: "YumeMotion", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "нет кадров для обучения движения"])
    }
    guard steps > 0 else { return }
    let runs = motionFrameRuns(memoryFrames)
    let total = memoryFrames.count
    let targetN = min(total, maxEncodedFrames)
    let selected = resampleSegmentIndices(total: total, targetCount: targetN)
    var latents: [Int: [Float]] = [:]
    var frames: [Int: [Float]] = [:]
    for i in selected {
        guard i >= 0, i < memoryFrames.count,
              let rgb = loadImageRGB(data: memoryFrames[i].jpeg, w: pw, h: ph) else { continue }
        frames[i] = rgb
        if let z = try? encodeLatent(vae: vae, image: rgb, w: pw, h: ph) {
            latents[i] = z
        }
    }
    guard !latents.isEmpty else {
        throw NSError(domain: "YumeMotion", code: 3,
                      userInfo: [NSLocalizedDescriptionKey: "не удалось закодировать ни одного кадра"])
    }
    let pairs = buildMotionTrainingPairs(runs: runs, latents: latents,
                                         window: window)
        .filter { frames[$0.k] != nil }
    guard !pairs.isEmpty else {
        throw NSError(domain: "YumeMotion", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "недостаточно кадров одного источника для окон движения"])
    }
    let latentDim = latents.values.first!.count
    let net = MotionMLPGPU(engine: MetalEngine.shared, latentDim: latentDim)
    var sum: Double = 0
    var sumSq: Double = 0
    var count = 0
    for z in latents.values {
        for v in z {
            sum += Double(v)
            sumSq += Double(v) * Double(v)
            count += 1
        }
    }
    if count > 0 {
        let mean = Float(sum / Double(count))
        let variance = max((sumSq / Double(count)) - Double(mean) * Double(mean), 1e-6)
        net.latentMean = mean
        net.latentScale = max(Float(sqrt(variance)), 1e-3)
    }
    let session = GpuSession(MetalEngine.shared)
    let mseBuf = MetalEngine.shared.device.makeBuffer(length: 4,
                                                      options: .storageModeShared)
    var rng = SplitMix64(seed: 0x900D_0000_0000_3333)
    var lossSum: Double = 0
    for step in 1...steps {
        if TrainingControl.shared.isCancelled {
            throw CancellationError()
        }
        let pair = pairs[Int(rng.next() % UInt64(pairs.count))]
        let loss = runMotionStepPixel(net: net, vae: vae, session: session,
                                      pair: pair, latents: latents,
                                      frames: frames, pw: pw, ph: ph,
                                      mseBuf: mseBuf,
                                      step: step, lr: lr)
        lossSum += loss
        if step % 50 == 0 {
            yumeLog(String(format: "Движение (пиксели): шаг %d/%d, MSE кадра %.4f (PSNR %.2f дБ)",
                           step, steps, lossSum / Double(step),
                           -10.0 * log10(max(lossSum / Double(step), 1e-12))))
        }
        onProgress?(step, steps)
    }
    try saveMotionCheckpoint(net: net, to: checkpointPath)
    yumeLog(String(format: "✅ Сеть движения (пиксели) обучена: %d шагов, средний MSE кадра %.4f → чекпоинт %@",
                   steps, lossSum / Double(steps), checkpointPath))
}

// ---------- Публичная точка входа (из UI) ----------

/// Загружает VAE из чекпоинта и обучает сеть движения на реальных парах.
/// Используется итоговым обучением (после VAE) и /api/train-motion.
public func trainMotionMLPFromCheckpoint(memoryFrames: [MemoryFrame],
                                         vaeCheckpointPath: String,
                                         pw: Int, ph: Int,
                                         steps: Int = 300,
                                         lr: Float = 0.001,
                                         window: Int = 24,
                                         maxEncodedFrames: Int = 512,
                                         objectivePath: String) throws {
    let st = try loadStateWithMeta(path: vaeCheckpointPath)
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine, latentChannels: st.latentChannels,
                         width: st.modelWidth, residual: st.archVersion >= 2,
                         groupNorm: st.archVersion >= 3)
    try vae.load(from: st.mats)
    try trainMotionMLP(memoryFrames: memoryFrames, vae: vae, pw: pw, ph: ph,
                       steps: steps, lr: lr, window: window,
                       maxEncodedFrames: maxEncodedFrames,
                       checkpointPath: objectivePath)
}

/// Загружает VAE из чекпоинта и обучает сеть движения с ПИКСЕЛЬНОЙ потерей.
public func trainMotionMLPPixelFromCheckpoint(memoryFrames: [MemoryFrame],
                                              vaeCheckpointPath: String,
                                              pw: Int, ph: Int,
                                              steps: Int = 300,
                                              lr: Float = 0.001,
                                              window: Int = 24,
                                              maxEncodedFrames: Int = 128,
                                              objectivePath: String) throws {
    let st = try loadStateWithMeta(path: vaeCheckpointPath)
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine, latentChannels: st.latentChannels,
                         width: st.modelWidth, residual: st.archVersion >= 2,
                         groupNorm: st.archVersion >= 3)
    try vae.load(from: st.mats)
    try trainMotionMLPPixel(memoryFrames: memoryFrames, vae: vae,
                            pw: pw, ph: ph, steps: steps, lr: lr,
                            window: window, maxEncodedFrames: maxEncodedFrames,
                            checkpointPath: objectivePath)
}

// ---------- Инференс ----------

/// Предсказывает латент сети движения: lerp(zA,zB,t) + поправка сети.
/// Входы нормализуются параметрами чекпоинта (mean/scale), выход — обратно.
/// Возвращает абсолютный латент (1×latentDim). Чтение — строго после end().
public func predictMotionLatent(net: MotionMLPGPU, zA: [Float], zB: [Float],
                                t: Double) -> [Float] {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let a = normalizeLatent(zA, mean: net.latentMean, scale: net.latentScale)
    let b = normalizeLatent(zB, mean: net.latentMean, scale: net.latentScale)
    let base = interpolateScalars(a, b, t: t)
    let emb = motionTimeEmbedding(t, dim: net.dimEmb)
    let xT = a + b
    let xTensor = GPUTensor(engine: engine, rows: 1, cols: xT.count)
    let baseTensor = GPUTensor(engine: engine, rows: 1, cols: base.count)
    session.begin()
    session.upload(xTensor, from: xT)
    session.upload(baseTensor, from: base)
    let pred = net.forward(session, xT: xTensor, emb: emb, base: baseTensor)
    let gpuMs = session.end()
    _ = gpuMs
    let norm = pred.toArray()
    // Численная защита: если сеть дала не-конечный результат на этой паре,
    // откатываемся на чистый lerp (сеть не должна ломать генерацию).
    guard norm.allSatisfy({ $0.isFinite }) else {
        return interpolateScalars(zA, zB, t: t)
    }
    let s = net.latentScale > 0 ? net.latentScale : 1
    return norm.map { $0 * s + net.latentMean }
}

// ---------- Обучение ----------

/// Нормализация латента к нулевому среднему и единичному разбросу.
func normalizeLatent(_ z: [Float], mean: Float, scale: Float) -> [Float] {
    let s = scale > 0 ? scale : 1
    return z.map { ($0 - mean) / s }
}

/// Один шаг обучения сети движения (forward/backward/Adam + loss).
/// Обучение идёт в НОРМАЛИЗОВАННОМ пространстве латентов (z'=(z−mean)/scale):
/// значения mu VAE малы и не переполняют MLP.
func runMotionStep(net: MotionMLPGPU, session: GpuSession,
                   pair: MotionPair, latents: [Int: [Float]],
                   mseBuf: MTLBuffer? = nil,
                   step: Int, lr: Float) -> Double {
    let zA = normalizeLatent(latents[pair.a]!, mean: net.latentMean, scale: net.latentScale)
    let zB = normalizeLatent(latents[pair.b]!, mean: net.latentMean, scale: net.latentScale)
    let zK = normalizeLatent(latents[pair.k]!, mean: net.latentMean, scale: net.latentScale)
    let t = pair.t
    let emb = motionTimeEmbedding(t, dim: net.dimEmb)
    let base = interpolateScalars(zA, zB, t: t)
    let xT = zA + zB
    let xTensor = GPUTensor(engine: session.engine, rows: 1, cols: xT.count)
    let baseTensor = GPUTensor(engine: session.engine, rows: 1, cols: base.count)
    let targetTensor = GPUTensor(engine: session.engine, rows: 1, cols: zK.count)
    // 1) Forward/backward + MSE в одной сессии.
    session.begin()
    session.upload(xTensor, from: xT)
    session.upload(baseTensor, from: base)
    session.upload(targetTensor, from: zK)
    let (pred, cache) = net.forwardCached(session, xT: xTensor, emb: emb,
                                         base: baseTensor)
    let grads = net.backward(session, cache, target: targetTensor)
    let mse = session.encodeMSE(recon: pred, target: targetTensor, into: mseBuf)
    let gpuMs = session.end()
    _ = gpuMs
    let loss = Double(mse.contents().assumingMemoryBound(to: Float.self)[0])
        / Double(max(net.latentDim, 1))
    // 2) Санация градиентов ТОЛЬКО после фиксации вычислений (иначе
    //    чтение невычисленных буферов даёт NaN/мусор).
    sanitizeGrads(grads, maxNorm: 10.0)
    // 3) Adam — отдельной сессией.
    session.begin()
    net.applyAdam(session, grads, t: step, lr: lr)
    session.end()
    return loss
}

/// Обучение сети движения на парах реального видео.
///
/// - Parameters:
///   - memoryFrames: кадры в памяти (порядок извлечения, source — имя видео);
///   - vae: обученный VAE (кодирование кадров в латенты);
///   - pw, ph: рабочий размер кадра модели (кратен 8);
///   - steps: число шагов обучения;
///   - lr: скорость обучения;
///   - window: длина окна движения (кадров);
///   - maxEncodedFrames: потолок числа кодируемых кадров (при нехватке —
///     равномерная выборка `resampleSegmentIndices`);
///   - checkpointPath: куда сохранить сеть;
///   - onProgress: (готово, всего).
public func trainMotionMLP(memoryFrames: [MemoryFrame],
                           vae: ConvVAEGPU,
                           pw: Int, ph: Int,
                           steps: Int = 600,
                           lr: Float = 0.0005,
                           window: Int = 24,
                           maxEncodedFrames: Int = 512,
                           checkpointPath: String,
                           onProgress: ((Int, Int) -> Void)? = nil) throws {
    guard !memoryFrames.isEmpty else {
        throw NSError(domain: "YumeMotion", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "нет кадров для обучения движения"])
    }
    guard steps > 0 else { return }
    let runs = motionFrameRuns(memoryFrames)
    let total = memoryFrames.count
    let targetN = min(total, maxEncodedFrames)
    let selected = resampleSegmentIndices(total: total, targetCount: targetN)
    let latents = encodeMotionLatents(frames: memoryFrames, selected: selected,
                                      vae: vae, pw: pw, ph: ph)
    guard !latents.isEmpty else {
        throw NSError(domain: "YumeMotion", code: 3,
                      userInfo: [NSLocalizedDescriptionKey: "не удалось закодировать ни одного кадра"])
    }
    let pairs = buildMotionTrainingPairs(runs: runs, latents: latents,
                                         window: window)
    guard !pairs.isEmpty else {
        throw NSError(domain: "YumeMotion", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "недостаточно кадров одного источника для окон движения"])
    }
    let latentDim = latents.values.first!.count
    let net = MotionMLPGPU(engine: MetalEngine.shared, latentDim: latentDim)
    // Нормализация латентов по обучающему набору (mean/scale — скаляры).
    var sum: Double = 0
    var sumSq: Double = 0
    var count = 0
    for z in latents.values {
        for v in z {
            sum += Double(v)
            sumSq += Double(v) * Double(v)
            count += 1
        }
    }
    if count > 0 {
        let mean = Float(sum / Double(count))
        let variance = max((sumSq / Double(count)) - Double(mean) * Double(mean), 1e-6)
        net.latentMean = mean
        net.latentScale = max(Float(sqrt(variance)), 1e-3)
    }
    let session = GpuSession(MetalEngine.shared)
    let mseBuf = MetalEngine.shared.device.makeBuffer(length: 4,
                                                      options: .storageModeShared)
    var rng = SplitMix64(seed: 0x900D_0000_0000_2222)
    var lossSum: Double = 0
    for step in 1...steps {
        if TrainingControl.shared.isCancelled {
            throw CancellationError()
        }
        let pair = pairs[Int(rng.next() % UInt64(pairs.count))]
        let loss = runMotionStep(net: net, session: session, pair: pair,
                                 latents: latents, mseBuf: mseBuf,
                                 step: step, lr: lr)
        lossSum += loss
        if step % 50 == 0 {
            yumeLog(String(format: "Движение: шаг %d/%d, loss %.4f (MSE латента)",
                           step, steps, lossSum / Double(step)))
        }
        onProgress?(step, steps)
    }
    try saveMotionCheckpoint(net: net, to: checkpointPath)
    let avg = lossSum / Double(steps)
    yumeLog(String(format: "✅ Сеть движения обучена: %d шагов, средний loss %.4f → чекпоинт %@",
                   steps, avg, checkpointPath))
}
