// YumeLayers — вспомогательные под-шаги цикла обучения.
//
// Выделено из `ConvTrainer.swift` (trainConvVAE) для:
//   • атомарности функций (лимит ≤120 строк);
//   • уменьшения длины файла (лимит ≤600 строк).
// Чистые помощники без собственного состояния: весь контекст передаётся
// параметрами.
import Foundation

/// Версия архитектуры для чекпоинта: 1=plain, 2=residual, 3=groupNorm(+residual).
func archVersionOf(_ vae: ConvVAEGPU) -> Int {
    vae.groupNorm ? 3 : (vae.residual ? 2 : 1)
}

/// Epoch-aware возобновление: номер эпохи, с которой продолжаем цикл, чтобы не
/// переучивать завершённые эпохи после клина/стопа (не тратить время впустую).
/// Чистая, детерминированная функция → тестируется без GPU/данных.
///  - resume=off либо данные сменились (fingerprint не совпал) → 0 (полный прогон);
///  - иначе clamp(savedEpoch) в [0, requestedEpochs] — старт со следующей эпохи.
public func resumeStartEpoch(savedEpoch: Int, requestedEpochs: Int,
                             dataMatches: Bool, resume: Bool) -> Int {
    guard resume, dataMatches else { return 0 }
    return min(max(savedEpoch, 0), max(requestedEpochs, 0))
}

/// Детектор плато/дивергенции: по серии средних loss по эпохам решает, перестало ли
/// обучение улучшаться. Чистая функция — тестируется без GPU (`testPlateauDetection`).
/// Срабатывает, если за `window` эпох относительное улучшение меньше `relTolerance`
/// (включая рост/колебание). Требует минимум window+1 точек, иначе рано судить.
public func plateauDetected(losses: [Double], relTolerance: Double = 0.02, window: Int = 3) -> Bool {
    guard window >= 1, losses.count >= window + 1 else { return false }
    let recent = Array(losses.suffix(window + 1))
    guard let first = recent.first, let last = recent.last,
          first.isFinite, last.isFinite, first > 0 else { return false }
    let improvement = (first - last) / first
    return improvement < relTolerance
}

/// Сохранение промежуточного чекпоинта (каждые 500 шагов).
///
/// Сохраняется и состояние оптимизатора: без него продолжение обучения
/// стартует с нулевых моментов Adam и сносит модель первым же шагом.
func saveIntermediateCheckpoint(vae: ConvVAEGPU, convPath: String,
                                step: Int, epoch: Int, stepIdx: Int,
                                epochSteps: Int, fingerprint: String) {
    do {
        let opt = CheckpointOptimizerState(step: step, moments: vae.optimizerMoments())
        try saveState(vae.matrices, to: convPath,
                      epoch: epoch, step: stepIdx, totalSteps: epochSteps,
                      dataFingerprint: fingerprint, optimizer: opt,
                      modelWidth: vae.width,
                      latentChannels: vae.latentChannels,
                      archVersion: archVersionOf(vae))
        yumeLog("Промежуточный чекпоинт: эпоха \(epoch), шаг \(stepIdx)/\(epochSteps) — сохранён")
    } catch {
        yumeLog("⚠️ ошибка промежуточного чекпоинта: \(error.localizedDescription)")
    }
}

/// Сохранение рабочего и защищённого финального чекпоинтов на границе
/// эпохи. Финал пишется в отдельный файл и не перезаписывается
/// промежуточными сохранениями.
func saveEpochCheckpoints(vae: ConvVAEGPU, convPath: String,
                          step: Int, nextEpoch: Int, epochSteps: Int,
                          fingerprint: String, stableRun: Bool) throws {
    guard stableRun else { return }
    let opt = CheckpointOptimizerState(step: step, moments: vae.optimizerMoments())
    let av = archVersionOf(vae)
    try saveState(vae.matrices, to: convPath,
                  epoch: nextEpoch, step: 0, totalSteps: epochSteps,
                  dataFingerprint: fingerprint, optimizer: opt,
                  modelWidth: vae.width,
                  latentChannels: vae.latentChannels,
                  archVersion: av)
    let finalPath = finalCheckpointPath(for: convPath)
    try saveState(vae.matrices, to: finalPath,
                  epoch: nextEpoch, step: 0, totalSteps: epochSteps,
                  dataFingerprint: fingerprint, optimizer: opt,
                  modelWidth: vae.width,
                  latentChannels: vae.latentChannels,
                  archVersion: av)
}

/// Трассировка градиентов и кэша первого шага эпохи 0 (локализация NaN).
func traceFirstStepGrads(_ grads: [GPUTensor], _ cache: VAECache,
                         mse: Double, recon0: Float, img0: Float) {
    for (i, g) in grads.enumerated() {
        let a = g.toArray()
        if a.contains(where: { $0.isNaN }) {
            yumeLog("❌ NaN в градиенте [\(i)]")
        } else {
            yumeLog(String(format: "grad[%d] min=%.4e max=%.4e", i, a.min()!, a.max()!))
        }
    }
    func scanNaN(_ name: String, _ t: GPUTensor) {
        let a = t.toArray()
        if a.contains(where: { $0.isNaN }) {
            yumeLog("❌ NaN найден в \(name)")
        }
    }
    scanNaN("p1", cache.p1); scanNaN("a1", cache.a1)
    scanNaN("p2", cache.p2); scanNaN("a2", cache.a2)
    scanNaN("p3", cache.p3); scanNaN("mu", cache.mu)
    scanNaN("logvar", cache.logvar); scanNaN("z", cache.z)
    scanNaN("u1", cache.u1)
    scanNaN("p4", cache.p4); scanNaN("a4", cache.a4)
    scanNaN("u2", cache.u2)
    scanNaN("p5", cache.p5); scanNaN("a5", cache.a5)
    scanNaN("u3", cache.u3)
    scanNaN("p6", cache.p6)
    scanNaN("recon", cache.recon)
    yumeLog(String(format: "debug step0: mse=%.6f recon0=%.4f img0=%.4f",
                   mse, recon0, img0))
}

/// Живой отчёт о шаге (раз в 5 шагов и на последнем).
func emitLiveStepReport(onStep: ((TrainStepReport) -> Void)?,
                        stepsThisRun: Int, pEpochs: Int, epochSteps: Int,
                        stepIdx: Int, effBatch: Int, datasetFrames: Int,
                        ep: Int, stepTimes: [Double]) {
    guard let onStep = onStep else { return }
    let done = stepsThisRun
    let total = max(1, pEpochs * epochSteps)
    if done % 5 == 0 || done == total {
        let recent = stepTimes.suffix(50)
        let recentAvg = recent.reduce(0, +) / Double(max(recent.count, 1))
        let frame = currentFrameIndex(epochIndex: ep, stepIndex: stepIdx,
                                     stepsPerEpoch: epochSteps, batch: effBatch,
                                     datasetFrames: datasetFrames)
        onStep(TrainStepReport(done: done, total: total,
                               epoch: ep + 1, epochs: pEpochs,
                               step: stepIdx + 1, stepsPerEpoch: epochSteps,
                               frame: frame, framesTotal: datasetFrames,
                               batch: effBatch,
                               secondsPerStep: recentAvg))
    }
}

/// Итог эпохи: средний loss/PSNR, журнал, превью и onProgress.
func emitEpochReport(ep: Int, lossSum: Double, psnrSum: Double, stepsDone: Int,
                     stepTimes: [Double], epochSteps: Int, start: Date,
                     previewTarget: [Float]?, previewRecon: [Float]?,
                     previewDir: String, pw: Int, ph: Int,
                     onProgress: TrainProgress?) throws {
    let meanLoss = lossSum / Double(max(stepsDone, 1))
    let meanPsnr = psnrSum / Double(max(stepsDone, 1))
    let perStep = stepTimes.suffix(epochSteps).reduce(0, +) / Double(max(stepTimes.suffix(epochSteps).count, 1))
    let elapsed = Date().timeIntervalSince(start)
    yumeLog(String(format: "epoch %d  loss %.6f  psnr %.2f dB  %.2f s/шаг  всего %.0f c",
                   ep, meanLoss, meanPsnr, perStep, elapsed))

    if let pt = previewTarget, let pr = previewRecon {
        let outPath = "\(previewDir)/conv_ep\(String(format: "%03d", ep)).png"
        try savePairPNG(path: outPath, w: pw, h: ph,
                        target: chwToHwc(pt), recon: chwToHwc(pr))
        onProgress?(ep, meanLoss, meanPsnr, outPath)
    } else {
        onProgress?(ep, meanLoss, meanPsnr, nil)
    }
}

/// Пауза между эпохами по теплу (прерываемая). Возвращает true, если
/// обучение пора остановить (пользователь нажал «Стоп»).
func epochCoolPause(gpuSensor: GpuSensor, pauseMs: Double,
                    targetTempC: Double, maxTempC: Double,
                    ep: Int, pEpochs: Int) -> Bool {
    guard ep < pEpochs - 1 else { return false }
    if TrainingControl.shared.isCancelled { return true }
    let hotTemp = gpuSensor.read()?.temperatureC
    let coolMs = epochCoolPauseMs(tempC: hotTemp, pauseMs: pauseMs,
                                  targetTempC: targetTempC, maxTempC: maxTempC)
    if coolMs > 0 {
        yumeLog(String(format: "🌡️ пауза между эпохами: %.0f мс на остывание (GPU %.0f °C)",
                       coolMs, hotTemp ?? 0))
        let coolDeadline = Date().addingTimeInterval(coolMs / 1000)
        while Date() < coolDeadline, !TrainingControl.shared.isCancelled {
            Thread.sleep(forTimeInterval: min(0.25, coolMs / 1000))
        }
    }
    return TrainingControl.shared.isCancelled
}

/// Подготовка источника данных: VRAM-пул или CPU-кэш; размер набора.
struct DataPrep {
    var source: DataSource
    var datasetFrames: Int
    var gpuPool: GPUFramePool?
    var frameCache: FrameCache?
}

func prepareDataSource(p: Params, engine: MetalEngine, pw: Int, ph: Int,
                      realW: Int, realH: Int,
                      emitStatus: @escaping (String, TrainStage) -> Void,
                      isPreparing: Bool) -> DataPrep {
    var gpuPool: GPUFramePool? = nil
    var frameCache: FrameCache? = nil
    var source: DataSource
    if let mem = p.memoryFrames, !mem.isEmpty {
        gpuPool = GPUFramePool(engine: engine, frames: mem, w: pw, h: ph,
                               realW: realW, realH: realH) { loaded, total in
            emitStatus(trainVRAMLoadNote(loaded: loaded, total: total),
                       isPreparing ? .preparing : .training)
        }
        if gpuPool == nil {
            frameCache = FrameCache(frames: mem, w: pw, h: ph, realW: realW, realH: realH)
        }
        source = DataSource(files: [], memFrames: mem, cache: frameCache,
                            w: pw, h: ph, realW: realW, realH: realH)
        yumeLog("Данные: \(mem.count) кадров из памяти\(gpuPool != nil ? " (VRAM-пул)" : (frameCache != nil ? " (кэш RGB8)" : ""))")
    } else {
        let files = (p.dataDir.map { listImages(in: $0) } ?? []).sorted()
        source = DataSource(files: files, w: pw, h: ph, realW: realW, realH: realH)
        if files.isEmpty {
            yumeLog("Данные: синтетические сцены (реальные кадры не заданы: --data <папка>)")
        } else {
            yumeLog("Данные: \(files.count) файлов из \(p.dataDir!)")
        }
    }
    let datasetFrames: Int
    if let mem = p.memoryFrames, !mem.isEmpty {
        datasetFrames = mem.count
    } else if let cache = frameCache {
        datasetFrames = cache.count
    } else {
        datasetFrames = source.files.count
    }
    return DataPrep(source: source, datasetFrames: datasetFrames,
                    gpuPool: gpuPool, frameCache: frameCache)
}

/// Санация градиентов + шаг Adam в одном сеансе GPU.
func applyAdamStep(session: GpuSession, vae: ConvVAEGPU,
                   grads: [GPUTensor], t: Int, lr: Float, effBatch: Int) -> Double {
    sanitizeGrads(grads, maxNorm: 1.0, preScale: 1.0 / Float(effBatch))
    session.begin()
    vae.applyAdam(session, grads: grads, t: t, lr: lr)
    return session.end()
}

/// Пауза, нарезанная на короткие срезы (проверка «Стоп» между ними).
func sleepInSlices(totalMs: Double) {
    guard totalMs > 0 else { return }
    for slice in pauseSlices(totalMs: totalMs) {
        if TrainingControl.shared.isCancelled { break }
        Thread.sleep(forTimeInterval: slice / 1000)
    }
}

/// Ограничение показания времени GPU стенным временем шага + предупреждение.
func boundedGpuReport(gpuMsThisStep: Double, wallMs: Double,
                     lastWarn: inout Date) -> Double {
    let gpuMs = boundedGpuMs(reportedMs: gpuMsThisStep, wallMs: wallMs)
    if gpuMsThisStep > wallMs + 50, Date().timeIntervalSince(lastWarn) > 60 {
        lastWarn = Date()
        yumeLog(String(format: "⚠️ драйвер сообщил %.0f мс GPU при %.0f мс стенного времени — показание отброшено (пауза не вырастет)",
                       gpuMsThisStep, wallMs))
    }
    return gpuMs
}
