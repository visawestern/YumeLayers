// YumeLayers — выгрузка реконструкций (выделено из ConvTrainer.swift).
//
// Отдельный модуль для соблюдения лимита 600 строк/файл.
import Foundation
import Metal

public func generateReconstructions(p: Params, onProgress: ((Int, Int) -> Void)? = nil) throws -> [String: Data] {
    let engine = MetalEngine.shared

    let pw = ((p.width + 7) / 8) * 8
    let ph = ((p.height + 7) / 8) * 8

    typealias FrameItem = (name: String, jpeg: Data)
    let items: [FrameItem]
    // Контрольная выборка (1 кадр/сек, итоговая модель) — приоритетна:
    // реконструируем только её, а не все кадры.
    if let ctrl = p.controlFrames, !ctrl.isEmpty {
        items = ctrl.map { ($0.name, $0.jpeg) }
        yumeLog("generateReconstructions: контрольная выборка \(items.count) кадров (1/сек) на итоговой модели")
    } else if let mem = p.memoryFrames, !mem.isEmpty {
        items = mem.map { ($0.name, $0.jpeg) }
        yumeLog("generateReconstructions: \(items.count) кадров из памяти (без диска)")
    } else {
        let files = (p.dataDir.map { listImages(in: $0) } ?? []).sorted()
        items = files.compactMap { f -> FrameItem? in
            guard let d = try? Data(contentsOf: URL(fileURLWithPath: f)) else { return nil }
            return ((f as NSString).lastPathComponent, d)
        }
        yumeLog("generateReconstructions: dataDir=\(p.dataDir ?? "nil"), files=\(items.count)")
    }
    guard !items.isEmpty else { return [:] }

    // Вторая GPU (встроенная Intel UHD) — только для достаточно больших наборов,
    // чтобы окупить инициализацию второго движка и компиляцию шейдеров.
    let allDevices = MTLCopyAllDevices()
    let secondaryDevice = allDevices.first { $0.registryID != engine.device.registryID }
    let useSecondary = secondaryDevice != nil && items.count >= 64

    if !useSecondary {
        return try runReconsOn(engine: engine, items: items, pw: pw, ph: ph,
                               realW: p.width, realH: p.height,
                               convPath: p.convPath, onProgress: onProgress)
    }

    let secondaryEngine = MetalEngine(device: secondaryDevice!)
    yumeLog("Реконструкции на двух GPU: Radeon Pro 5500M + \(secondaryEngine.deviceName)")
    let mid = items.count / 2
    let lock = NSLock()
    var done = 0
    var errors: [Error] = []
    var out: [String: Data] = [:]
    let group = DispatchGroup()

    func runChunk(_ eng: MetalEngine, _ chunk: ArraySlice<FrameItem>) {
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            defer { group.leave() }
            do {
                let part = try runReconsOn(engine: eng, items: Array(chunk), pw: pw, ph: ph,
                                           realW: p.width, realH: p.height,
                                           convPath: p.convPath, onProgress: nil)
                lock.lock()
                out.merge(part) { _, new in new }
                done += part.count
                let d = done
                lock.unlock()
                onProgress?(d, items.count)
            } catch {
                lock.lock(); errors.append(error); lock.unlock()
            }
        }
    }
    runChunk(engine, items[..<mid])
    runChunk(secondaryEngine, items[mid...])
    group.wait()

    if let e = errors.first { throw e }
    return out
}

private func runReconsOn(engine: MetalEngine, items: [(name: String, jpeg: Data)],
                         pw: Int, ph: Int, realW: Int? = nil, realH: Int? = nil,
                         convPath: String,
                         onProgress: ((Int, Int) -> Void)?) throws -> [String: Data] {
    // Канонический декод: истинный размер + паддинг (не растяжение).
    let decW = realW ?? pw, decH = realH ?? ph
    let session = GpuSession(engine)
    var vae = ConvVAEGPU(engine: engine)
    let sensor = GpuSensor()
    if FileManager.default.fileExists(atPath: convPath) {
        let st = try loadStateWithMeta(path: convPath)
        // Модель той же архитектуры (ширина + латент), что и обученная:
        // иначе формы слоёв не совпадают и загрузка падает.
        vae = ConvVAEGPU(engine: engine, latentChannels: st.latentChannels,
                          width: st.modelWidth)
        try vae.load(from: st.mats)
    } else {
        return [:]
    }

    let batchSize = max(1, min(16, effectiveBatchSize(
        requested: 16, w: pw, h: ph,
        budgetBytes: activationBudgetBytes(
            workingSet: engine.device.recommendedMaxWorkingSetSize))))
    var result: [String: Data] = [:]
    var done = 0
    var lastLoggedPercent = 0
    if !items.isEmpty { yumeLog("Реконструкции: старт, \(items.count) кадров") }
    for range in batchRanges(total: items.count, batch: batchSize) {
        if TrainingControl.shared.isCancelled {
            yumeLog("Реконструкции остановлены пользователем (\(done)/\(items.count))")
            break
        }
        thermalWaitForRecon()
        let chunkItems = Array(items[range])
        let imgs = decodeChunkImages(chunkItems, decW: decW, decH: decH)

        session.begin()
        let (_, targetToItem, recons) = encodeReconBatch(
            session, vae: vae, imgs: imgs, items: chunkItems, pw: pw, ph: ph)
        let reconGpuMs = session.end()

        let sleepMs = reconSleepFor(hot: sensor.read()?.temperatureC ?? 0,
                                    gpuMs: reconGpuMs)
        if sleepMs > 0 { Thread.sleep(forTimeInterval: sleepMs / 1000) }

        let reconByItem = Dictionary(uniqueKeysWithValues: zip(targetToItem, recons))
        done = emitReconResults(into: &result, items: chunkItems,
                                reconByItem: reconByItem, done: done,
                                pw: pw, ph: ph, onProgress: onProgress,
                                lastLoggedPercent: &lastLoggedPercent)
    }
    return result
}

/// Термо-контроль между батчами реконструкций.
private func thermalWaitForRecon() {
    switch ProcessInfo.processInfo.thermalState {
    case .critical:
        yumeLog("⚠️ реконструкции: критический перегрев — пауза до остывания (не дольше 5 мин)")
        let coolDeadline = Date().addingTimeInterval(300)
        while ProcessInfo.processInfo.thermalState == .critical,
              !TrainingControl.shared.isCancelled, Date() < coolDeadline {
            Thread.sleep(forTimeInterval: 0.25)
        }
    case .serious:
        Thread.sleep(forTimeInterval: 0.05)
    default:
        break
    }
}

/// Параллельный декод чанка в RGB8 (истинный размер + паддинг).
private func decodeChunkImages(_ items: [(name: String, jpeg: Data)],
                              decW: Int, decH: Int) -> [[Float]?] {
    var imgs = [[Float]?](repeating: nil, count: items.count)
    let lock = NSLock()
    DispatchQueue.concurrentPerform(iterations: items.count) { j in
        let image = loadImageRGB(data: items[j].jpeg, w: decW, h: decH)
            .map { padToMultipleOf8($0, w: decW, h: decH).0 }
        lock.lock(); imgs[j] = image; lock.unlock()
    }
    return imgs
}

/// Загрузка CHW-тензоров и forward пачки (по апостериорному среднему).
private func encodeReconBatch(_ session: GpuSession, vae: ConvVAEGPU,
                              imgs: [[Float]?],
                              items: [(name: String, jpeg: Data)],
                              pw: Int, ph: Int)
    -> (targets: [GPUTensor], map: [Int], recons: [GPUTensor]) {
    var targets: [GPUTensor] = []
    var map: [Int] = []
    var recons: [GPUTensor] = []
    for j in 0..<items.count {
        guard let img = imgs[j] else { continue }
        let t = GPUTensor(engine: session.engine, rows: 3, cols: img.count / 3)
        session.upload(t, from: hwcToChw(img))
        targets.append(t)
        map.append(j)
    }
    for t in targets {
        let cache = vae.forward(session, image: t, w: pw, h: ph, deterministic: true)
        recons.append(cache.recon)
    }
    return (targets, map, recons)
}

/// Пауза после батча по температуре (доля работы).
private func reconSleepFor(hot: Double, gpuMs: Double) -> Double {
    let duty: Double = hot >= defaultMaxTempC ? 0.60
        : hot >= defaultTargetTempC ? 0.85
        : safeDutyCap
    return dutySleepMs(gpuMs: gpuMs, duty: duty, maxPauseMs: 2000)
}

/// Эмит готовых реконструкций в результат с отчётом прогресса.
private func emitReconResults(into result: inout [String: Data],
                              items: [(name: String, jpeg: Data)],
                              reconByItem: [Int: GPUTensor],
                              done: Int, pw: Int, ph: Int,
                              onProgress: ((Int, Int) -> Void)?,
                              lastLoggedPercent: inout Int) -> Int {
    var d = done
    for (j, item) in items.enumerated() {
        if let t = reconByItem[j] {
            let r = t.toArray()
            if let png = encodePNG(width: pw, height: ph, rgb: chwToHwc(r)) {
                result[item.name] = png
            }
        }
        d += 1
        onProgress?(d, items.count)
        if shouldLogProgress(done: d, total: items.count,
                             lastLoggedPercent: lastLoggedPercent) {
            lastLoggedPercent = Int((Double(d) / Double(max(items.count, 1))) * 100)
            yumeLog("Реконструкции: \(d)/\(items.count) кадров (\(lastLoggedPercent)%)")
        }
    }
    return d
}
