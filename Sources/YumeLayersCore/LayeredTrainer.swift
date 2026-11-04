// YumeLayers — обучающий цикл для 8-канальной факторизации по экспертам.
//
// Отличие от `ConvTrainer.trainConvVAE`: вход не RGB-кадр, а 8 каналов
// ground-truth из `LayeredScene` (фон | объекты | маска | свет). Лосс —
// по слоям И по точной сборке кадра (LayeredVAELoss.swift). Это первый
// рабочий контур, где декодер учится не «рисовать картинку», а выдавать
// слои, из которых картинка собирается без потерь.
//
// ИТЕРАЦИЯ 6. Три вещи, которых раньше не было:
//   • обучение на ДАТАСЕТЕ сцен, а не на одной процедурной картинке;
//   • настраиваемая ёмкость латента (C=16 дал потолок 28.8 дБ);
//   • расписание lr и запись чекпоинта safetensors.
//
// ТЕПЛО. Каждый command buffer сопровождается `GpuDutyGuard`: GPU занята
// не более 90% времени, 10% остаётся WindowServer и на остывание. Раньше
// короткие прогоны (в том числе тесты) гнали шаги без пауз и держали
// очередь GPU целиком — именно с этого началась принудительная
// перезагрузка 4 ноября 2026. Аварийная температура добавляет паузу
// поверх скважности.
import Foundation

/// Косинусное расписание скорости обучения: `baseLR` на старте,
/// `floorFraction·baseLR` в конце, монотонно и без резкого обрыва.
///
/// Зачем: на фиксированном lr=0.01 потеря layered-VAE выходила на плато
/// 28.8 дБ и дальше только шумела — большие градиенты не давали сети
/// доуточнить маску и свет. Шаг за пределами расписания Не повышает lr:
/// это важно при resume, когда `step` продолжает расти.
public func cosineLR(step: Int, totalSteps: Int, baseLR: Float,
                     floorFraction: Float = 0.05) -> Float {
    let total = max(totalSteps, 1)
    let t = Double(min(max(step, 0), total)) / Double(total)
    let floorLR = Double(baseLR) * Double(min(max(floorFraction, 0), 1))
    let span = Double(baseLR) - floorLR
    return Float(floorLR + 0.5 * span * (1 + cos(Double.pi * t)))
}

/// Итог layered-прогона. Все поля — наблюдаемые, без «успеха по умолчанию»:
/// `acceptedByFloor` считается из реальных PSNR/SSIM собранного кадра.
public struct LayeredTrainReport {
    public let steps: Int
    public let firstLoss: Double
    public let lastLoss: Double
    public let psnr: Float
    public let ssim: Float
    public let acceptedByFloor: Bool
    public let compositeChannels: Int
    /// Фактический потолок скважности, с которым работал прогон.
    public let duty: Double
    /// Сколько РАЗНЫХ сцен участвовало в обучении (1 = прежний синтетический путь).
    public let sceneCount: Int
    /// Ёмкость латента, с которой реально собран VAE.
    public let latentChannels: Int
    /// Собранный кадр первой сцены после обучения (HWC) — для PNG-превью.
    public let reconstructionHWC: [Float]
    /// Путь чекпоинта, который удалось записать; nil, если не запрошен или
    /// запись не удалась. Отдельно от `checkpointSaved`, чтобы не выдавать
    /// «сохранено» за успех.
    public let checkpointPath: String?
    public let checkpointSaved: Bool
    /// Наблюдаемая тепловая телеметрия прогона (не оценка, а счётчики).
    public let totalGpuMs: Double
    public let totalPauseMs: Double
    public let sleepCalls: Int
    public let emergencyPauses: Int
    public let measuredDuty: Double

    public init(steps: Int, firstLoss: Double, lastLoss: Double,
                psnr: Float, ssim: Float, acceptedByFloor: Bool,
                compositeChannels: Int, duty: Double,
                sceneCount: Int = 1, latentChannels: Int = ConvVAEGPU.latentC,
                reconstructionHWC: [Float] = [],
                checkpointPath: String? = nil, checkpointSaved: Bool = false,
                totalGpuMs: Double = 0, totalPauseMs: Double = 0,
                sleepCalls: Int = 0, emergencyPauses: Int = 0,
                measuredDuty: Double = 0) {
        self.steps = steps
        self.firstLoss = firstLoss
        self.lastLoss = lastLoss
        self.psnr = psnr
        self.ssim = ssim
        self.acceptedByFloor = acceptedByFloor
        self.compositeChannels = compositeChannels
        self.duty = duty
        self.sceneCount = sceneCount
        self.latentChannels = latentChannels
        self.reconstructionHWC = reconstructionHWC
        self.checkpointPath = checkpointPath
        self.checkpointSaved = checkpointSaved
        self.totalGpuMs = totalGpuMs
        self.totalPauseMs = totalPauseMs
        self.sleepCalls = sleepCalls
        self.emergencyPauses = emergencyPauses
        self.measuredDuty = measuredDuty
    }
}

/// Отпечаток датасета для честного resume: если кадры поменялись, чекпоинт
/// не должен молча считаться «тем же самым». FNV-1a по разреженной выборке,
/// чтобы не читать все 8·w·h float'ов каждого кадра заново.
public func layeredFingerprint(_ targets: [LayeredTarget]) -> String {
    var hash: UInt64 = 0xcbf29ce484222325
    func mix(_ v: UInt64) {
        hash ^= v
        hash = hash &* 0x100000001b3
    }
    mix(UInt64(targets.count))
    for t in targets {
        mix(UInt64(t.w)); mix(UInt64(t.h))
        let stride = max(1, t.channelsCHW.count / 256)
        var i = 0
        while i < t.channelsCHW.count {
            mix(UInt64(bitPattern: Int64(Int32(bitPattern: t.channelsCHW[i].bitPattern))))
            i += stride
        }
    }
    return String(format: "%016llx", hash)
}

/// Обучить 8-канальный VAE.
///
/// Два пути в одном графе:
///   • `scenes == nil` — прежний синтетический прогон на одной сцене
///     (`makeLayeredScene`), сохранён для `layered`, тестов и воспроизводимости;
///   • `scenes != nil` — обучение на реальных аннотированных кадрах
///     (`LayerAnnotator` / `runLayerPipeline`), сцены циклически чередуются.
///
/// Разрешение округляется вверх до кратного 8 (латент /8). У всех сцен
/// датасета размер обязан совпадать — иначе формы GPU-буферов разойдутся.
///
/// - Parameter powerPercent: потолок мощности GPU в процентах; даже 100
///   упирается в жёсткие 90% (`safeDutyCap`).
public func trainLayeredVAE(steps: Int = 60,
                            w: Int = 32, h: Int = 24,
                            seed: UInt64 = 2026,
                            scenes: [LayeredScene]? = nil,
                            lr: Float = 0.01,
                            beta: Float = 0.01,
                            layerWeight: Float = 1,
                            composeWeight: Float = 1,
                            latentChannels: Int = ConvVAEGPU.latentC,
                            cosineSchedule: Bool = false,
                            checkpointPath: String? = nil,
                            powerPercent: Int = defaultGpuPowerPercent)
    -> LayeredTrainReport {

    let dataset = scenes ?? [makeLayeredScene(w: max(((max(w, 8) + 7) / 8) * 8, 8),
                                              h: max(((max(h, 8) + 7) / 8) * 8, 8),
                                              seed: seed)]
    precondition(!dataset.isEmpty, "trainLayeredVAE: пустой датасет")
    let pw = ((max(dataset[0].w, 8) + 7) / 8) * 8
    let ph = ((max(dataset[0].h, 8) + 7) / 8) * 8
    for s in dataset {
        precondition(s.w == pw && s.h == ph,
                     "trainLayeredVAE: все сцены должны быть \(pw)×\(ph), получено \(s.w)×\(s.h)")
    }
    let total = max(steps, 1)

    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let sensor = GpuSensor()
    let dutyGuard = GpuDutyGuard(powerPercent: powerPercent)
    let pacer = GpuDutyPacer(powerPercent: powerPercent)

    // Таргеты живут в VRAM весь прогон: гонять 8·w·h float'ов на каждом
    // шаге дороже, чем посчитать по ним градиент.
    let targets = dataset.map { LayeredTarget(scene: $0) }
    let targetsGPU = targets.map { LayeredTargetGPU(engine: engine, target: $0) }
    let xs = targets.map {
        GPUTensor(engine: engine, data: $0.channelsCHW,
                  rows: LayeredScene.layerChannels)
    }
    let vae = ConvVAEGPU(engine: engine, channels: LayeredScene.layerChannels,
                         latentChannels: latentChannels)

    /// Детерминированный замер потери по ВСЕМ сценам + реконструкции.
    /// Пауза по скважности применяется и здесь: замер — такая же GPU-работа,
    /// как шаг обучения, и раньше именно короткие eval-прогоны перегревали
    /// машину, потому что шли без пауз вообще.
    func evaluate() -> (loss: Double, recons: [[Float]]) {
        var sum = 0.0
        var recons: [[Float]] = []
        recons.reserveCapacity(targets.count)
        for (i, t) in targets.enumerated() {
            // Metal-объекты autorelease'ятся. Без локального пула длинный
            // прогон (20 000+ шагов) копит их до конца процесса и падает
            // по памяти (exit 137) — пул сбрасывает их на каждой итерации.
            autoreleasepool {
                session.begin()
                let cache = vae.forward(session, image: xs[i], w: pw, h: ph,
                                        deterministic: true)
                let gpuMs = session.end()
                let recon = cache.recon.toArray()
                sum += layeredLoss(recon: recon, target: t,
                                   layerWeight: layerWeight,
                                   composeWeight: composeWeight)
                      + Double(beta) * ConvVAEGPU.klMean(cache)
                recons.append(recon)
                pacer.wait(gpuMs: gpuMs)
            }
        }
        return (sum / Double(targets.count), recons)
    }

    let first = evaluate()
    for step in 1...total {
        autoreleasepool {
            let idx = (step - 1) % targets.count
            let stepLR = cosineSchedule
                ? cosineLR(step: step - 1, totalSteps: total, baseLR: lr)
                : lr

            session.begin()
            let cache = vae.forward(session, image: xs[idx], w: pw, h: ph,
                                    deterministic: false, seed: UInt32(step))
            let (_, grads) = vae.backwardLayered(session, cache,
                                                 target: targetsGPU[idx],
                                                 layerWeight: layerWeight,
                                                 composeWeight: composeWeight,
                                                 beta: beta)
            vae.applyAdam(session, grads: grads, t: step, lr: stepLR)
            let gpuMs = session.end()
            pacer.wait(gpuMs: gpuMs)

            // Раз в 10 шагов смотрим на датчик: аварийная температура важнее
            // любого расписания, обучение обязано уступить GPU системе.
            if step % 10 == 0 {
                _ = pacer.emergency(tempC: sensor.read()?.temperatureC)
            }
        }
    }
    let last = evaluate()
    pacer.sleepDebt()

    // Пол качества — по СОБРАННОМУ кадру, а не по 8 каналам по отдельности.
    // Берём первую сцену: она же уходит в PNG-превью конвейера.
    let quality = composedQuality(recon: last.recons[0], target: targets[0])
    let floor = QualityFloor()
    let reconHWC = composeHWC(w: pw, h: ph, recon: last.recons[0])
    let accepted = floor.accepts(reference: targets[0].compositeHWC,
                                 reconstruction: reconHWC)

    var saved = false
    var savedPath: String?
    if let path = checkpointPath {
        do {
            let dir = (path as NSString).deletingLastPathComponent
            if !dir.isEmpty {
                try FileManager.default.createDirectory(atPath: dir,
                                                         withIntermediateDirectories: true)
            }
            try saveState(vae.matrices, to: path, step: total, totalSteps: total,
                          dataFingerprint: layeredFingerprint(targets))
            saved = true
            savedPath = path
        } catch {
            yumeLog("чекпоинт не сохранён: \(error.localizedDescription)")
        }
    }

    return LayeredTrainReport(steps: total,
                              firstLoss: first.loss, lastLoss: last.loss,
                              psnr: quality.psnr, ssim: quality.ssim,
                              acceptedByFloor: accepted,
                              compositeChannels: 3,
                              duty: dutyGuard.duty,
                              sceneCount: targets.count,
                              latentChannels: vae.latentChannels,
                              reconstructionHWC: reconHWC,
                              checkpointPath: savedPath,
                              checkpointSaved: saved,
                              totalGpuMs: pacer.totalGpuMs,
                              totalPauseMs: pacer.totalPauseMs,
                              sleepCalls: pacer.sleepCalls,
                              emergencyPauses: pacer.emergencyPauses,
                              measuredDuty: pacer.measuredDuty)
}
