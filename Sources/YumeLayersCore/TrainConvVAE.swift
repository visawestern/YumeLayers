// YumeLayers — ядро обучения свёрточного VAE (trainConvVAE).
//
// Выделено из ConvTrainer.swift и декомпозировано на контекст-объект
// (TrainContext) + методы фаз: сборка пачки, прямой/обратный проход,
// потеря, Adam, термо-контроль, чекпоинты. Логика перенесена дословно.

import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import Metal

/// Накопители одной эпохи (потеря, PSNR, число шагов, превью).
private struct EpochState {
    var lossSum = 0.0
    var psnrSum = 0.0
    var stepsDone = 0
    var previewTarget: [Float]? = nil
    var previewRecon: [Float]? = nil
}

/// Контекст прогона: неизменяемые зависимости + мутируемое состояние цикла.
private struct TrainContext {
    // Зависимости (константы прогона).
    let p: Params
    let engine: MetalEngine
    let session: GpuSession
    let vae: ConvVAEGPU
    var source: DataSource
    let gpuPool: GPUFramePool?
    let datasetFrames: Int
    let pw: Int
    let ph: Int
    let effBatch: Int
    let epochSteps: Int
    let inputPool: [GPUTensor]
    let mseOutBufs: [MTLBuffer]
    let previewDir: String
    let currentFingerprint: String
    let resumeWithoutOptimizer: Bool
    let pauseMs: Double
    let onStep: ((TrainStepReport) -> Void)?
    let onProgress: TrainProgress?
    let gpuSensor = GpuSensor()

    // Мутируемое состояние.
    var t: Int
    var stepsThisRun: Int
    var start: Date
    var stepTimes: [Double] = []
    var stableRun = true
    var stepBaselineMs = 0.0
    var lastHeatLevel: HeatLevel = .nominal
    var pendingSleepMs = 0.0
    var gpuMsThisStep = 0.0
    var heatLogAt = Date.distantPast
    var heatDebouncer = HeatDebouncer(steps: 3)
    var lastTempC: Double? = nil
    var power: PowerController
    var currentDuty: Double
    var lastSensorAt = Date.distantPast
    var lastSensorLog = Date.distantPast
    var lastDutyCutAt = Date.distantPast
    var lastGpuClockWarnAt = Date.distantPast

    init(p: Params, engine: MetalEngine, session: GpuSession, vae: ConvVAEGPU,
         source: DataSource, gpuPool: GPUFramePool?, datasetFrames: Int,
         pw: Int, ph: Int, effBatch: Int, epochSteps: Int,
         inputPool: [GPUTensor], mseOutBufs: [MTLBuffer],
         previewDir: String, currentFingerprint: String,
         resumeWithoutOptimizer: Bool, pauseMs: Double,
         onStep: ((TrainStepReport) -> Void)?, onProgress: TrainProgress?,
         t: Int, power: PowerController, currentDuty: Double) {
        self.p = p
        self.engine = engine
        self.session = session
        self.vae = vae
        self.source = source
        self.gpuPool = gpuPool
        self.datasetFrames = datasetFrames
        self.pw = pw
        self.ph = ph
        self.effBatch = effBatch
        self.epochSteps = epochSteps
        self.inputPool = inputPool
        self.mseOutBufs = mseOutBufs
        self.previewDir = previewDir
        self.currentFingerprint = currentFingerprint
        self.resumeWithoutOptimizer = resumeWithoutOptimizer
        self.pauseMs = pauseMs
        self.onStep = onStep
        self.onProgress = onProgress
        self.t = t
        self.stepsThisRun = 0
        self.start = Date()
        self.power = power
        self.currentDuty = currentDuty
    }
}

extension TrainContext {
    /// Термо-регулятор: решение на каждый шаг по фактическому времени GPU.
    mutating func governHeat(gpuMs: Double, wallMs: Double, duty: Double,
                             tempC: Double?) -> ThrottleDecision {
        let raw = throttleDecision(
            gpuMs: gpuMs,
            wallMs: wallMs,
            baselineMs: stepBaselineMs,
            thermalRaw: ProcessInfo.processInfo.thermalState.rawValue,
            floorMs: p.adaptiveThrottle ? 0 : pauseMs,
            maxPauseMs: Double(p.maxPauseMs),
            duty: duty,
            tempC: tempC)
        stepBaselineMs = raw.baselineMs

        let level = heatDebouncer.update(raw.level)

        if level != lastHeatLevel, Date().timeIntervalSince(heatLogAt) > 60 {
            heatLogAt = Date()
            yumeLog(String(format: "🌡️ тепло: %@ · GPU %.0f мс/шаг, база %.0f мс · пауза %.0f мс · нагрузка %.0f%%",
                           level.title, gpuMs, raw.baselineMs,
                           raw.sleepMs, duty * 100))
            if level >= .hot, let t = tempC {
                yumeLog(String(format: "⚠️ нагрев вырос до «%@» при %.0f °C — снижаю нагрузку",
                               level.title, t))
            }
        }
        lastHeatLevel = level
        return ThrottleDecision(level: level, sleepMs: raw.sleepMs, baselineMs: raw.baselineMs)
    }

    /// Собирает пачку кадров за шаг (из пула VRAM или из исходников).
    mutating func gatherBatch() -> (imgs: [[Float]], tensors: [GPUTensor]) {
        var batchImgs: [[Float]] = []
        var batchTensors: [GPUTensor] = []
        var poolFrames: [GPUTensor] = []
        if let pool = gpuPool {
            for _ in 0..<effBatch { poolFrames.append(pool.next()) }
        }
        session.begin()
        if poolFrames.isEmpty {
            for b in 0..<effBatch {
                let img = hwcToChw(source.sequentialSample())
                batchImgs.append(img)
                session.upload(inputPool[b], from: img)
                batchTensors.append(inputPool[b])
            }
        } else {
            for b in 0..<effBatch {
                session.copy(inputPool[b], from: poolFrames[b])
                batchTensors.append(inputPool[b])
            }
            for _ in 0..<effBatch { batchImgs.append([]) }
        }
        return (batchImgs, batchTensors)
    }

    /// Прямой + обратный проход для всей пачки, накопление градиентов.
    func forwardBackward(tensors: [GPUTensor]) -> (acc: [GPUTensor]?, recons: [GPUTensor],
                                                   mseBufs: [MTLBuffer],
                                                   firstCache: VAECache?,
                                                   firstGrads: [GPUTensor]?) {
        var firstGrads: [GPUTensor]? = nil
        var firstCache: VAECache? = nil
        var recons: [GPUTensor] = []
        var mseBufs: [MTLBuffer] = []
        var accGrads: [GPUTensor]? = nil
        for b in 0..<effBatch {
            let img = tensors[b]
            let noiseSeed = trainingNoiseSeed(step: t, batchIndex: b)
            let batchCache = vae.forward(session, image: img, w: pw, h: ph,
                                         deterministic: false, seed: noiseSeed)
            var dRecon: GPUTensor? = nil
            if p.lossKind == "l1" {
                dRecon = session.encodeL1Grad(recon: batchCache.recon, target: img)
            }
            if p.edgeWeight > 0 {
                let edgeGrad = session.encodeSobelEdgeGrad(recon: batchCache.recon,
                                                           target: img, w: pw, h: ph)
                let edgeTerm = session.smul(p.edgeWeight, edgeGrad)
                if let base = dRecon {
                    dRecon = session.add(base, edgeTerm)
                } else {
                    // dReconOverride заменяет дефолтный MSE-градиент; чтобы
                    // edge-loss был ДОПОЛНЕНИЕМ к MSE, включаем оба слагаемых.
                    dRecon = session.add(
                        session.smul(2.0 / Float(img.n), session.sub(batchCache.recon, img)),
                        edgeTerm)
                }
            }
            let (_, grads) = vae.backward(session, batchCache, target: img,
                                          beta: rgbKLBeta,
                                          dReconOverride: dRecon)
            recons.append(batchCache.recon)
            mseBufs.append(session.encodeMSE(recon: batchCache.recon, target: img,
                                             into: mseOutBufs[b]))
            if firstCache == nil { firstCache = batchCache }
            if firstGrads == nil { firstGrads = grads }
            if accGrads == nil {
                accGrads = grads.map { g in
                    let c = GPUTensor(engine: engine, rows: g.rows, cols: g.cols)
                    session.copy(c, from: g)
                    return c
                }
            } else {
                for i in 0..<grads.count { session.addInPlace(accGrads![i], grads[i]) }
            }
        }
        return (accGrads, recons, mseBufs, firstCache, firstGrads)
    }

    /// Читает MSE с GPU и (для первого шага эпохи) пиксели превью.
    func computeLossAndPreview(state: EpochState, mseBufs: [MTLBuffer],
                               recons: [GPUTensor],
                               tensors: [GPUTensor],
                               imgs: [[Float]]) -> (mse: Double, firstImg: [Float]?,
                                                    firstR: [Float]?) {
        var mseSum = 0.0
        var firstImg: [Float]? = nil
        var firstR: [Float]? = nil
        for b in 0..<effBatch {
            let mse = Double(mseBufs[b].contents().assumingMemoryBound(to: Float.self)[0])
                / Double(max(recons[b].n, 1))
            mseSum += mse
            if b == 0 && state.previewTarget == nil {
                if gpuPool != nil {
                    firstImg = tensors[0].toArray()
                } else {
                    firstImg = imgs[0]
                }
                firstR = recons[0].toArray()
            }
        }
        return (mseSum / Double(effBatch), firstImg, firstR)
    }

    /// Adam-шаг по накопленным градиентам + счётчики + разгон LR.
    mutating func applyAdam(grads: [GPUTensor]) {
        t += 1
        stepsThisRun += 1
        let lrScale = resumeWithoutOptimizer
            ? resumeWarmupFactor(step: stepsThisRun, warmupSteps: resumeWarmupSteps)
            : 1
        gpuMsThisStep += applyAdamStep(session: session, vae: vae,
                                      grads: grads, t: t,
                                      lr: p.lr * lrScale, effBatch: effBatch)
    }

    /// Живой отчёт по шагам + замер времени шага.
    mutating func liveReport(stepIdx: Int, ep: Int, stepStart: Date) {
        stepTimes.append(Date().timeIntervalSince(stepStart))
        emitLiveStepReport(onStep: onStep, stepsThisRun: stepsThisRun,
                           pEpochs: p.epochs, epochSteps: epochSteps,
                           stepIdx: stepIdx, effBatch: effBatch,
                           datasetFrames: datasetFrames, ep: ep,
                           stepTimes: stepTimes)
    }

    /// Термо-контроль шага: датчик, контур, пауза, аварийный срез.
    mutating func thermalControl() {
        if Date().timeIntervalSince(lastSensorAt) >= 2.0 {
            lastSensorAt = Date()
            let stats = gpuSensor.read()
            currentDuty = power.update(stats: stats)
            if let s = stats, s.hasTemperature {
                lastTempC = s.temperatureC
                if s.temperatureC >= emergencyTempC {
                    yumeLog(String(format: "🔥 GPU %.0f °C — принудительное остывание до 20 с",
                                   s.temperatureC))
                    let coolDeadline = Date().addingTimeInterval(20)
                    while Date() < coolDeadline, !TrainingControl.shared.isCancelled {
                        Thread.sleep(forTimeInterval: 0.25)
                    }
                    currentDuty = max(power.duty * 0.5, 0.05)
                }
                if Date().timeIntervalSince(lastSensorLog) > 60 {
                    lastSensorLog = Date()
                    yumeLog(String(format: "🌡️ GPU %.0f °C · %.0f Вт → нагрузка %.0f%% (цель %.0f °C)",
                                   s.temperatureC, s.powerW, currentDuty * 100,
                                   power.targetTempC))
                }
            } else {
                lastTempC = nil
            }
        }
        let wallMs = stepTimes.last.map { $0 * 1000 } ?? 0
        let gpuMs = boundedGpuReport(gpuMsThisStep: gpuMsThisStep, wallMs: wallMs,
                                     lastWarn: &lastGpuClockWarnAt)
        let decision = governHeat(gpuMs: gpuMs, wallMs: gpuMs,
                                  duty: currentDuty, tempC: lastTempC)
        pendingSleepMs = decision.sleepMs

        if lastTempC == nil, decision.level >= .hot,
           Date().timeIntervalSince(lastDutyCutAt) > 120 {
            lastDutyCutAt = Date()
            currentDuty = max(currentDuty * 0.90, 0.50)
            yumeLog(String(format: "⚠️ нет данных датчика, шаг замедлился (%@) — снижаю нагрузку до %.0f%%",
                           decision.level.title, currentDuty * 100))
        }
    }

    /// Промежуточный чекпоинт каждые 500 шагов + пауза срезами.
    func checkpointAndSleep(stepIdx: Int, ep: Int) {
        if stepIdx > 0 && stepIdx % 500 == 0 {
            saveIntermediateCheckpoint(vae: vae, convPath: p.convPath,
                                      step: t, epoch: ep, stepIdx: stepIdx,
                                      epochSteps: epochSteps,
                                      fingerprint: currentFingerprint)
        }
        sleepInSlices(totalMs: pendingSleepMs)
    }

    /// Один шаг обучения на пачку кадров.
    mutating func runStep(ep: Int, stepIdx: Int, state: inout EpochState) throws {
        if TrainingControl.shared.isCancelled { return }
        let stepStart = Date()
        let (batchImgs, batchTensors) = gatherBatch()
        let fwd = forwardBackward(tensors: batchTensors)
        gpuMsThisStep = session.end()

        let loss = computeLossAndPreview(state: state, mseBufs: fwd.mseBufs,
                                         recons: fwd.recons,
                                         tensors: batchTensors,
                                         imgs: batchImgs)
        let lastMse = loss.mse
        if !lastMse.isFinite {
            stableRun = false
            yumeLog("⚠️ loss = NaN на эпохе \(ep) — аварийная остановка обучения (чекпоинт не сохраняется)")
            throw ConvTrainingNaNError()
        }
        state.lossSum += lastMse
        state.psnrSum += -10.0 * log10(max(lastMse, 1e-12))
        state.stepsDone += 1

        if state.previewTarget == nil {
            state.previewTarget = loss.firstImg
            state.previewRecon = loss.firstR
        }
        if ep == 0 && stepIdx == 0 {
            traceFirstStepGrads(fwd.firstGrads!, fwd.firstCache!,
                                mse: lastMse, recon0: loss.firstR![0],
                                img0: loss.firstImg![0])
        }

        applyAdam(grads: fwd.acc!)
        liveReport(stepIdx: stepIdx, ep: ep, stepStart: stepStart)
        thermalControl()
        checkpointAndSleep(stepIdx: stepIdx, ep: ep)
    }

    /// Полный прогон по эпохам.
    ///
    /// Отмена проверяется и на границе эпох, и ПЕРЕД КАЖДЫМ шагом: «Стоп»,
    /// нажатый внутри длинной эпохи, прерывает обучение на ближайшем шаге,
    /// а не после досчитывания всей эпохи (порядок проверок = модель
    /// `runEpochLoop`, зафиксированная тестом `testPerStepCancellationModel`).
    mutating func runEpochs() throws {
        for ep in 0..<p.epochs {
            if TrainingControl.shared.isCancelled {
                yumeLog("Остановлено пользователем на эпохе \(ep)")
                break
            }
            var es = EpochState()
            var stopped = false
            for stepIdx in 0..<epochSteps {
                if TrainingControl.shared.isCancelled { stopped = true; break }
                try runStep(ep: ep, stepIdx: stepIdx, state: &es)
            }
            // Прерванную эпоху не отчитываем и не чекпоинтим: незавершённая
            // эпоха не должна записываться как выполненная (совпадает с моделью).
            if stopped {
                yumeLog("Остановлено пользователем на шаге \(es.stepsDone)/\(epochSteps) эпохи \(ep)")
                break
            }

            try emitEpochReport(ep: ep, lossSum: es.lossSum, psnrSum: es.psnrSum,
                                stepsDone: es.stepsDone, stepTimes: stepTimes,
                                epochSteps: epochSteps, start: start,
                                previewTarget: es.previewTarget, previewRecon: es.previewRecon,
                                previewDir: previewDir, pw: pw, ph: ph,
                                onProgress: onProgress)

            try saveEpochCheckpoints(vae: vae, convPath: p.convPath, step: t,
                                     nextEpoch: ep + 1, epochSteps: epochSteps,
                                     fingerprint: currentFingerprint,
                                     stableRun: stableRun)

            if epochCoolPause(gpuSensor: gpuSensor, pauseMs: pauseMs,
                              targetTempC: power.targetTempC,
                              maxTempC: power.maxTempC,
                              ep: ep, pEpochs: p.epochs) { break }
        }
    }
}

/// Обучение свёрточного VAE на Metal.
public func trainConvVAE(p: Params, onProgress: TrainProgress? = nil,
                         onStep: ((TrainStepReport) -> Void)? = nil) throws {
    TrainingControl.shared.beginRun()
    defer { TrainingControl.shared.endRun() }

    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let (realW, realH, pw, ph) = try validatedTrainingShape(p)

    var isPreparing = true
    let emitStatus: (String, TrainStage) -> Void = { note, stage in
        onStep?(TrainStepReport(done: 0, total: max(1, p.epochs * max(1, p.steps)),
                                epoch: 1, epochs: max(1, p.epochs),
                                step: 0, stepsPerEpoch: max(1, p.steps),
                                frame: 0, framesTotal: p.memoryFrames?.count ?? 0,
                                batch: 0, secondsPerStep: nil,
                                note: note, stage: stage))
    }
    emitStatus(trainPreparationNote(), .preparing)
    let prep = prepareDataSource(p: p, engine: engine, pw: pw, ph: ph,
                                 realW: realW, realH: realH,
                                 emitStatus: emitStatus, isPreparing: isPreparing)

    yumeLog("Свёрточный VAE: кадр \(realW)×\(realH) → внутренний \(pw)×\(ph), латент \(pw/8)×\(ph/8)×\(ConvVAEGPU.latentC) (сжатие \(Int(ConvVAEGPU.compressionRatio))×)")

    let vae = ConvVAEGPU(engine: engine,
                          latentChannels: max(p.latentChannels, 1),
                          width: max(p.modelWidth, 1),
                          residual: p.residual,
                          groupNorm: p.groupNorm)
    let currentFingerprint: String
    if let mem = p.memoryFrames, !mem.isEmpty {
        currentFingerprint = dataFingerprint(frames: mem)
    } else {
        currentFingerprint = "none"
    }
    let (adamStep, resumeWithoutOptimizer) = try loadResumeState(
        into: vae, convPath: p.convPath, resume: p.resume,
        currentFingerprint: currentFingerprint)

    let previewDir = (p.outPrefix as NSString).deletingLastPathComponent
    try FileManager.default.createDirectory(atPath: previewDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(atPath: (p.convPath as NSString).deletingLastPathComponent,
                                            withIntermediateDirectories: true)

    let batchCfg = computeEffectiveBatch(engine: engine, p: p, pw: pw, ph: ph)
    let epochSteps = computeEpochSteps(p: p, effBatch: batchCfg.effBatch)

    let power = PowerController(ceiling: effectiveDutyCeiling(powerPercent: p.gpuPowerPercent))
    yumeLog(String(format: "Нагрузка GPU: потолок %.0f%% (доля работы ≤ %.0f%%), цель %.0f °C",
                   Double(p.gpuPowerPercent), power.ceiling * 100, power.targetTempC))

    var ctx = TrainContext(
        p: p, engine: engine, session: session, vae: vae,
        source: prep.source, gpuPool: prep.gpuPool,
        datasetFrames: prep.datasetFrames,
        pw: pw, ph: ph,
        effBatch: batchCfg.effBatch, epochSteps: epochSteps,
        inputPool: batchCfg.inputPool, mseOutBufs: batchCfg.mseOutBufs,
        previewDir: previewDir, currentFingerprint: currentFingerprint,
        resumeWithoutOptimizer: resumeWithoutOptimizer,
        pauseMs: Double(max(p.pauseMs, 0)),
        onStep: onStep, onProgress: onProgress,
        t: adamStep, power: power, currentDuty: power.duty)

    try ctx.runEpochs()

    if TrainingControl.shared.isCancelled { throw CancellationError() }
    yumeLog("Готово. Чекпоинт: \(p.convPath)")
}
