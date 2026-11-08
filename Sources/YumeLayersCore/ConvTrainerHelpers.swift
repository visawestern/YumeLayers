// YumeLayers — хелперы trainConvVAE (декомпозиция).
import Foundation
import Metal

/// Валидирует параметры и приводит размер кадра к кратному 8.
func validatedTrainingShape(_ p: Params) throws -> (realW: Int, realH: Int, pw: Int, ph: Int) {
    guard p.epochs >= 0 else {
        throw NSError(domain: "Yume", code: 12,
                      userInfo: [NSLocalizedDescriptionKey: "epochs < 0 (\(p.epochs))"])
    }
    let realW = p.width, realH = p.height
    let pw = ((realW + 7) / 8) * 8
    let ph = ((realH + 7) / 8) * 8
    guard pw >= 8, ph >= 8 else {
        throw NSError(domain: "Yume", code: 12,
                      userInfo: [NSLocalizedDescriptionKey:
                        "кадр слишком мал: \(p.width)×\(p.height) — минимум 8×8"])
    }
    return (realW, realH, pw, ph)
}

/// Считает эффективный батч (потолок стабильности/памяти) и тензоры пачки.
func computeEffectiveBatch(engine: MetalEngine, p: Params, pw: Int, ph: Int)
    -> (effBatch: Int, inputPool: [GPUTensor], mseOutBufs: [MTLBuffer]) {
    let vramBudgetFrames = effectiveBatchSize(
        requested: Int.max,
        w: pw, h: ph,
        budgetBytes: activationBudgetBytes(
            workingSet: engine.device.recommendedMaxWorkingSetSize))
    let maxSafeBatch = 16
    let effBatch = min(max(p.batch, 1), vramBudgetFrames, maxSafeBatch)
    if effBatch < max(p.batch, 1) {
        yumeLog("Батч урезан \(p.batch) → \(effBatch) (потолок стабильности \(maxSafeBatch); по памяти влезает \(vramBudgetFrames), \(convVAEBytesPerSample(w: pw, h: ph) / 1_048_576) МБ на кадр)")
    } else {
        yumeLog("Батч: \(effBatch) кадров за шаг (потолок стабильности \(maxSafeBatch); по памяти влезает \(vramBudgetFrames), \(convVAEBytesPerSample(w: pw, h: ph) / 1_048_576) МБ на кадр)")
    }
    let inputPool = (0..<effBatch).map { _ in GPUTensor(engine: engine, rows: 3, cols: pw * ph) }
    let mseOutBufs = (0..<effBatch).map { _ in
        engine.device.makeBuffer(length: 4, options: .storageModeShared)!
    }
    return (effBatch, inputPool, mseOutBufs)
}

/// Считает число шагов за эпоху (авто-подгонка под пачку).
func computeEpochSteps(p: Params, effBatch: Int) -> Int {
    var epochSteps = max(1, p.steps)
    if let mem = p.memoryFrames, !mem.isEmpty, p.steps == mem.count, effBatch > 1 {
        epochSteps = max(1, Int((Double(mem.count) / Double(effBatch)).rounded(.up)))
        yumeLog("Шагов за эпоху (авто под пачку): \(epochSteps) шагов × \(effBatch) кадров (набор \(mem.count), последний шаг неполный)")
    }
    return epochSteps
}

