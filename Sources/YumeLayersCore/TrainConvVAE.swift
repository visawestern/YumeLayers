// YumeLayers — ядро обучения свёрточного VAE (trainConvVAE).
// Выделено из ConvTrainer.swift в отдельный файл по логической ответственности.
import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import Metal

public func trainConvVAE(p: Params, onProgress: TrainProgress? = nil,
                         onStep: ((TrainStepReport) -> Void)? = nil) throws {
    // Регистрируем прогон для watchdog'а остановки: если поток уйдёт в
    // драйвер GPU и не увидит флаг отмены, `cancel()` завершит процесс.
    TrainingControl.shared.beginRun()
    defer { TrainingControl.shared.endRun() }

    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // Краевые входы public API: интерфейс зажимает значения, но функцию
    // вызывают и тесты/скрипты — вырождения ловим явно, а не крашем
    // диапазона «Range requires lowerBound ≤ upperBound».
        let (realW, realH, pw, ph) = try validatedTrainingShape(p)

    var source: DataSource
    var frameCache: FrameCache? = nil
    // Фаза подготовки: сообщаем интерфейсу, что цикл жив, ещё до первого
    // шага — загрузка модели и кадров в VRAM занимает минуты, и без этого
    // отчёта лоадер выглядел бы «зависшим».
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
    source = prep.source
    frameCache = prep.frameCache
    var gpuPool = prep.gpuPool
    let datasetFrames = prep.datasetFrames

    yumeLog("Свёрточный VAE: кадр \(realW)×\(realH) → внутренний \(pw)×\(ph), латент \(pw/8)×\(ph/8)×\(ConvVAEGPU.latentC) (сжатие \(Int(ConvVAEGPU.compressionRatio))×)")

    let vae = ConvVAEGPU(engine: engine)
    // Отпечаток текущих данных — вычисляется один раз до освобождения JPEG.
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

    var t = adamStep
    // Шаги в ЭТОМ прогоне (для прогресса и разгона LR). `t` — общий счётчик
    // Adam, он продолжается с восстановленного значения.
    var stepsThisRun = 0
    let start = Date()
    var stepTimes: [Double] = []
    let pauseMs = Double(max(p.pauseMs, 0))
    var stableRun = true   // false, если обнаружен NaN — чекпоинт сохранять нельзя

    // Термо-регулятор. Работает НА КАЖДОМ шаге и управляет долей работы GPU,
    // а не только «уровнем тревоги»: 4 ноября 2026 система ушла в панику
    // watchdogd, потому что GPU была занята почти непрерывно и WindowServer
    // не мог получить доступ к тому же устройству 120 секунд подряд.
    // Подробности и формулы — в ThermalGovernor.swift.
    var stepBaselineMs = 0.0          // лучший (холодный) замер шага
    var lastHeatLevel: HeatLevel = .nominal
    var pendingSleepMs = 0.0          // пауза, вычисленная на этом шаге
    var gpuMsThisStep = 0.0
    var heatLogAt = Date.distantPast
    // Сглаживание уровня: без него дрейф 328/366 мс при пороге 2.5× дёргал
    // «горячо ↔ критично» на каждом шаге и забивал журнал.
    var heatDebouncer = HeatDebouncer(steps: 3)
    // Последняя прочитанная температура: нужна детектору, чтобы не путать
    // нагрев со сбросом частоты по мощности (см. `heatLevel`).
    var lastTempC: Double? = nil

    // Замкнутый контур по температуре GPU. Читаем телеметрию из IORegistry
    // (без root) раз в 2 секунды и подстраиваем долю работы так, чтобы карта
    // держалась ниже порога троттлинга. Пользователь задаёт только потолок
    // («мощность, %»); фактическую нагрузку находит сам контур.
    let gpuSensor = GpuSensor()
    var power = PowerController(ceiling: effectiveDutyCeiling(powerPercent: p.gpuPowerPercent))
    var currentDuty = power.duty
    var lastSensorAt = Date.distantPast
    var lastSensorLog = Date.distantPast
    var lastDutyCutAt = Date.distantPast
    /// Когда последний раз сообщали про отброшенное показание времени GPU.
    var lastGpuClockWarnAt = Date.distantPast
    yumeLog(String(format: "Нагрузка GPU: потолок %.0f%% (доля работы ≤ %.0f%%), цель %.0f °C",
                   Double(p.gpuPowerPercent), power.ceiling * 100, power.targetTempC))

    // Термо-регулятор: решение принимается НА КАЖДОМ шаге.
    //
    // Раньше здесь стоял thermalGuard, который вызывался раз в 10 шагов и
    // только менял `pauseMs`, — а `pauseMs` применялся исключительно на
    // границе эпохи (43 187 шагов). То есть авто-троттлинг не срабатывал
    // ни разу за всё обучение. Теперь пауза считается из фактического
    // времени работы GPU и применяется сразу.
    func governHeat(gpuMs: Double, wallMs: Double, duty: Double,
                    tempC: Double?) -> ThrottleDecision {
        let raw = throttleDecision(
            gpuMs: gpuMs,
            wallMs: wallMs,
            baselineMs: stepBaselineMs,
            thermalRaw: ProcessInfo.processInfo.thermalState.rawValue,
            // floorMs — это НЕ «минимальная пауза вообще», а фиксированный
            // режим: он нужен только когда авто-регулятор выключен и нагрузку
            // задаёт поле «Пауза, мс». Если авто включён, полом служит сама
            // доля работы, иначе 250 мс из интерфейса перекрывали бы решение
            // контура: при 99% расчётная пауза 1.7 мс, а floor поднимал её
            // до 250 мс — фактическая нагрузка падала до ~40%, и карта
            // не могла догреться до рабочей точки ни при каких настройках.
            floorMs: p.adaptiveThrottle ? 0 : pauseMs,
            maxPauseMs: Double(p.maxPauseMs),
            duty: duty,
            tempC: tempC)
        stepBaselineMs = raw.baselineMs

        // Уровень сглаживаем: одиночные выбросы времени шага — это шум
        // планировщика и драйвера, а не смена теплового режима.
        let level = heatDebouncer.update(raw.level)

        // Пишем в журнал при смене УЖЕ СГЛАЖЕННОГО уровня и не чаще раза
        // в минуту — иначе строки о тепле утонут в прогрессе.
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

    // Пачка кадров за шаг: под активации отведено 30% рабочего набора GPU
    // (60% уже занял пул кадров, 10% — запас). Считаем не по размеру кадра,
    // а по полному объёму активаций одного образца.
        let batchCfg = computeEffectiveBatch(engine: engine, p: p, pw: pw, ph: ph)
        let effBatch = batchCfg.effBatch
        let inputPool = batchCfg.inputPool
        let mseOutBufs = batchCfg.mseOutBufs

    // «Полный проход» по набору: при пачке >1 число оптимизаторских шагов
    // автоматически сокращается (кадры за шаг = effBatch). Пользовательский
    // steps (когда он задан явно) не трогаем.
        var epochSteps = computeEpochSteps(p: p, effBatch: effBatch)

    // Эпохи НИКОГДА не пропускаются: полный прогон с эпохи 0 по всем
    // p.epochs, независимо от resume (resume грузит только веса).
    isPreparing = false
    for ep in 0..<p.epochs {
        if TrainingControl.shared.isCancelled {
            yumeLog("Остановлено пользователем на эпохе \(ep)")
            break
        }
        var lossSum = 0.0
        var psnrSum = 0.0
        var stepsDone = 0
        // цель для превью — первый сэмпл эпохи
        var previewTarget: [Float]? = nil
        var previewRecon: [Float]? = nil

        // Всегда с шага 0: дообучение идёт полной эпохой с начала.
        for stepIdx in 0..<epochSteps {
            if TrainingControl.shared.isCancelled { break }
            var accGrads: [GPUTensor]? = nil
            var lastMse: Double = 0
            let stepStart = Date()

            var batchImgs: [[Float]] = []
            var batchTensors: [GPUTensor] = []
            // Кадры берём ДО открытия command buffer сессии: при перезагрузке
            // окна пул сам коммитит свои буферы.
            var poolFrames: [GPUTensor] = []
            if let pool = gpuPool {
                for _ in 0..<effBatch { poolFrames.append(pool.next()) }
            }
            session.begin()
            if poolFrames.isEmpty {
                for b in 0..<effBatch {
                    // Кэш/данные отдают HWC — сеть ждёт CHW.
                    let img = hwcToChw(source.sequentialSample())
                    batchImgs.append(img)
                    session.upload(inputPool[b], from: img)
                    batchTensors.append(inputPool[b])
                }
            } else {
                // Копирование кадра — ВНУТРИ command buffer сессии. Раньше это
                // был отдельный command buffer и отдельное ожидание на КАЖДЫЙ
                // шаг: лишняя синхронизация с GPU и лишняя нагрузка на драйвер,
                // который на длинных прогонах зависал.
                for b in 0..<effBatch {
                    session.copy(inputPool[b], from: poolFrames[b])
                    batchTensors.append(inputPool[b])
                }
                for _ in 0..<effBatch { batchImgs.append([]) }
            }

            var firstGrads: [GPUTensor]? = nil
            var firstCache: VAECache? = nil
            var recons: [GPUTensor] = []
            var mseBufs: [MTLBuffer] = []
            for b in 0..<effBatch {
                let img = batchTensors[b]
                // Шум VAE — УНИКАЛЬНЫЙ для каждого шага и кадра пачки. С
                // фиксированным seed (как было) модель выучивает один вектор
                // шума и не работает с остальными — оценка через ε = 0 даёт
                // совсем другое качество, чем показывает обучение.
                let noiseSeed = trainingNoiseSeed(step: t, batchIndex: b)
                let batchCache = vae.forward(session, image: img, w: pw, h: ph,
                                             deterministic: false, seed: noiseSeed)
                // β=1.0 (по умолчанию) доминирует над реконструкцией: KL-член
                // схлопывает латент к N(0,1) ещё до того, как декодер успевает
                // что-то выучить — реконструкции выходят тёмными даже на ярких
                // кадрах. Меро β=0.01 даёт MSE-члену вклад ~в 17 раз сильнее,
                // и модель учится сохранять яркость и содержание (проверено
                // замером на одном ярком кадре: 22.5 дБ против 10.0 при β=1).
                let (_, grads) = vae.backward(session, batchCache, target: img,
                                              beta: rgbKLBeta)
                recons.append(batchCache.recon)
                // MSE считается прямо на GPU (без чтения полных тензоров на CPU).
                mseBufs.append(session.encodeMSE(recon: batchCache.recon, target: img,
                                                 into: mseOutBufs[b]))
                if firstCache == nil { firstCache = batchCache }
                if firstGrads == nil { firstGrads = grads }
                if accGrads == nil {
                    // Накопитель создаётся «пустым» (storageModePrivate не гарантирует
                    // нули), поэтому ПЕРВОЕ слагаемое копируется, а не прибавляется.
                    // Раньше здесь был addInPlace поверх мусора — градиенты уезжали
                    // в случайные величины и спасал только клиппинг.
                    accGrads = grads.map { g in
                        let c = GPUTensor(engine: engine, rows: g.rows, cols: g.cols)
                        session.copy(c, from: g)
                        return c
                    }
                } else {
                    for i in 0..<grads.count { session.addInPlace(accGrads![i], grads[i]) }
                }
            }
            gpuMsThisStep = session.end()   // один commit + один wait на всю пачку

            // --- Фаза 3: потеря с GPU (скаляр на каждый кадр пачки) ---
            var mseSum = 0.0
            var firstImg: [Float]? = nil
            var firstR: [Float]? = nil
            for b in 0..<effBatch {
                let mse = Double(mseBufs[b].contents().assumingMemoryBound(to: Float.self)[0])
                    / Double(max(recons[b].n, 1))
                mseSum += mse
                // Для превью нужны реальные пиксели — читаем ОДИН кадр,
                // и только когда превью ещё нет (первый шаг эпохи).
                if b == 0 && previewTarget == nil {
                    if gpuPool != nil {
                        // Из пула VRAM: оригинал берём из исходного кадра,
                        // С ПАДДИНГОМ (pw×ph), чтобы длина совпадала с реконструкцией.
                        firstImg = batchTensors[0].toArray()
                    } else {
                        firstImg = batchImgs[0]
                    }
                    firstR = recons[0].toArray()
                }
            }
            lastMse = mseSum / Double(effBatch)
            // Детект NaN ДО накопления метрик: раньше шаг с NaN успевал
            // попасть в lossSum/stepsDone, и средняя потеря эпохи печаталась
            // как «loss nan».
            if !lastMse.isFinite {
                stableRun = false
                yumeLog("⚠️ loss = NaN на эпохе \(ep) — аварийная остановка обучения (чекпоинт не сохраняется)")
                throw ConvTrainingNaNError()
            }
            lossSum += lastMse
            psnrSum += -10.0 * log10(max(lastMse, 1e-12))
            stepsDone += 1

            if previewTarget == nil {
                previewTarget = firstImg
                previewRecon = firstR
            }
            if ep == 0 && stepIdx == 0 {
                traceFirstStepGrads(firstGrads!, firstCache!,
                                    mse: lastMse, recon0: firstR![0],
                                    img0: firstImg![0])
            }

            // Санация: NaN/Inf в градиентах → 0; глобальный клиппинг нормы.
            // Деление на размер пачки делается в том же проходе (preScale),
            // а не отдельными toArray/setData на каждый тензор.
            // Adam по накопленным градиентам (lr делится на размер пачки)
            t += 1
            stepsThisRun += 1
            // Разгон скорости — только когда состояние оптимизатора не найдено
            // (первый шаг Adam с нулевыми моментами равен ±lr по каждому весу).
            let lrScale = resumeWithoutOptimizer
                ? resumeWarmupFactor(step: stepsThisRun, warmupSteps: resumeWarmupSteps)
                : 1
            gpuMsThisStep += applyAdamStep(session: session, vae: vae,
                                          grads: accGrads!, t: t,
                                          lr: p.lr * lrScale, effBatch: effBatch)

            stepTimes.append(Date().timeIntervalSince(stepStart))

            // Живой прогресс по шагам: эпоха, текущий кадр набора, шаг,
            // скорость. Отчёт идёт каждые 5 шагов (и на последнем) — этого
            // достаточно для лоадера, но не дёргает UI на каждом шаге.
            emitLiveStepReport(onStep: onStep, stepsThisRun: stepsThisRun,
                               pEpochs: p.epochs, epochSteps: epochSteps,
                               stepIdx: stepIdx, effBatch: effBatch,
                               datasetFrames: datasetFrames, ep: ep,
                               stepTimes: stepTimes)

            // Термо-регулятор: решение на КАЖДОМ шаге по фактическому времени GPU.
            // Раз в 2 с пересчитываем долю работы по живой температуре карты.
            if Date().timeIntervalSince(lastSensorAt) >= 2.0 {
                lastSensorAt = Date()
                let stats = gpuSensor.read()
                currentDuty = power.update(stats: stats)
                if let s = stats, s.hasTemperature {
                    lastTempC = s.temperatureC
                    // Предохранитель: выше 86 °C обучение уходит на остывание,
                    // что бы ни решил контур. Перезагрузка случалась на 93 °C.
                    if s.temperatureC >= emergencyTempC {
                        // Остывание — прерываемое: раньше Thread.sleep(20)
                        // игнорировал «Стоп», и интерфейс подвисал до 20 с.
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
                    // Датчик пропал: честно обнуляем, иначе детектор будет
                    // опираться на устаревшее значение.
                    lastTempC = nil
                }
            }
            let wallMs = stepTimes.last.map { $0 * 1000 } ?? 0
            let gpuMs = boundedGpuReport(gpuMsThisStep: gpuMsThisStep, wallMs: wallMs,
                                         lastWarn: &lastGpuClockWarnAt)
            // Дрейф считаем по ЧИСТОМУ времени GPU, а не по стенному: пауза,
            // которую мы сами добавляем, не должна выглядеть как нагрев —
            // иначе регулятор уходит в спираль «больше пауза → больше дрейф →
            // ещё больше пауза». gpuMs от пауз не зависит.
            let decision = governHeat(gpuMs: gpuMs,
                                      wallMs: gpuMs,
                                      duty: currentDuty,
                                      tempC: lastTempC)
            pendingSleepMs = decision.sleepMs

            // Аварийный резерв — ТОЛЬКО когда датчик недоступен. Тогда
            // единственный признак проблемы это просадка времени шага.
            //
            // При живом датчике этот блок был вреден: он каждые 2 минуты
            // срезал нагрузку на 10%, а контур тут же возвращал её к потолку,
            // потому что карта холодная (64 °C). В журнале это выглядело как
            // «снижаю нагрузку» без единого реального изменения — ручка
            // дёргалась туда-сюда, а доля работы оставалась 99%.
            if lastTempC == nil, decision.level >= .hot,
               Date().timeIntervalSince(lastDutyCutAt) > 120 {
                lastDutyCutAt = Date()
                currentDuty = max(currentDuty * 0.90, 0.50)
                yumeLog(String(format: "⚠️ нет данных датчика, шаг замедлился (%@) — снижаю нагрузку до %.0f%%",
                               decision.level.title, currentDuty * 100))
            }

            // Промежуточный чекпоинт каждые 500 шагов: веса + текущий прогресс
            // (эпоха/шаг в метаданных) — чтобы после перезапуска можно было
            // продолжить ровно с места срыва. Сохраняем и состояние
            // оптимизатора: без него продолжение обучения стартует с нулевых
            // моментов Adam и сносит модель первым же шагом.
            if stepIdx > 0 && stepIdx % 500 == 0 {
                saveIntermediateCheckpoint(vae: vae, convPath: p.convPath,
                                          step: t, epoch: ep, stepIdx: stepIdx,
                                          epochSteps: epochSteps,
                                          fingerprint: currentFingerprint)
            }

            // Пауза — ЗДЕСЬ, в конце каждого шага, а не между эпохами.
            // Именно её отсутствие перегрело GPU 4 ноября 2026.
            //
            // Пауза режется на короткие срезы: «Стоп» проверяется между ними,
            // иначе кнопка ждала конца паузы (до 2 с) и выглядела как
            // «не работает».
            sleepInSlices(totalMs: pendingSleepMs)
        }

        try emitEpochReport(ep: ep, lossSum: lossSum, psnrSum: psnrSum,
                            stepsDone: stepsDone, stepTimes: stepTimes,
                            epochSteps: epochSteps, start: start,
                            previewTarget: previewTarget, previewRecon: previewRecon,
                            previewDir: previewDir, pw: pw, ph: ph,
                            onProgress: onProgress)

        try saveEpochCheckpoints(vae: vae, convPath: p.convPath, step: t,
                                 nextEpoch: ep + 1, epochSteps: epochSteps,
                                 fingerprint: currentFingerprint,
                                 stableRun: stableRun)

        // Пауза между эпохами — тоже только по теплу, а не «по расписанию».
        //
        // Раньше здесь всегда спалось max(pauseMs, 1000) мс: за эпоху карта
        // якобы успевает прогреться, и ей нужен отдых. Но замер на 99%
        // (`yume gpuheat 180 0`) показал обратное: за 180 с непрерывной
        // работы температура не поднялась выше 78 °C и сама сползла к 68–70,
        // мощность 5–23 Вт. Греться нечему — значит и остывать не от чего,
        // и фиксированная секунда простоя на каждой границе эпохи это чистый
        // простой. Отдых даём только если карта реально у стоп-порога.
        if epochCoolPause(gpuSensor: gpuSensor, pauseMs: pauseMs,
                          targetTempC: power.targetTempC,
                          maxTempC: power.maxTempC,
                          ep: ep, pEpochs: p.epochs) { break }
    }

    if TrainingControl.shared.isCancelled { throw CancellationError() }
    yumeLog("Готово. Чекпоинт: \(p.convPath)")
}
