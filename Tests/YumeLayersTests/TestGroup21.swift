// YumeLayers — политика логирования недоставленного обработчика завершения GPU.
//
// На AMD Radeon Pro 5500M драйвер регулярно не доставляет addCompletedHandler,
// хотя буфер уже выполнен (status == .completed). Это штатный обход: выходим по
// статусу и продолжаем. Раньше о каждом таком обходе печаталось «⚠️», и длинные
// прогоны тонули в тысячах ложных предупреждений, за которыми не увидеть
// настоящую ошибку. Тест фиксирует троттлинг и классификацию (чистые функции из
// GpuWaitPolicy.swift).
import Foundation

func testGpuHandlerMissPolicy() {
    section("GpuHandlerMissPolicy: троттлинг штатных обходов недоставленного обработчика")

    // completed-обход — штатный, не настоящая проблема; failed — да.
    expect(!gpuHandlerMissIsRealProblem(state: .completed),
           "completed-обход без обработчика — не проблема")
    expect(gpuHandlerMissIsRealProblem(state: .failed),
           "failed-статус — настоящая проблема")

    // Троттлинг: первые 3 штатных обхода печатаем, дальше молчим до 500-го.
    expect(gpuShouldLogHandlerMiss(state: .completed, count: 1),
           "1-й штатный обход — печатаем")
    expect(gpuShouldLogHandlerMiss(state: .completed, count: 3),
           "3-й штатный обход — печатаем")
    expect(!gpuShouldLogHandlerMiss(state: .completed, count: 4),
           "4-й штатный обход — молчим (не flooding)")
    expect(!gpuShouldLogHandlerMiss(state: .completed, count: 42),
           "рядовой обход — молчим")
    expect(gpuShouldLogHandlerMiss(state: .completed, count: 500),
           "каждый 500-й — печатаем (видно нарастание)")
    expect(gpuShouldLogHandlerMiss(state: .failed, count: 9999),
           "ошибка драйвера — печатаем всегда")
    expect(!gpuShouldLogHandlerMiss(state: .completed, count: 0),
           "нулевой счётчик — не печатаем мусор")
}

/// Покадровая остановка: модель цикла обязана обрываться НА ШАГЕ отмены,
/// а не досчитывать всю эпоху. Это регрессионный тест на то, что проверка
/// isCancelled стоит на каждом шаге, а не только на границе эпох.
func testPerStepCancellationModel() {
    section("TrainingControl: покадровая отмена обучения (модуль цикла эпохи/шага)")

    // Отмена на (ep=0, step=3) из 2×10: должно выполниться ровно 3 шага,
    // 0 целых эпох, остановка по пользователю. Эталон «только на границе эпох»
    // ошибочно сделал бы 10 шагов эпохи 0.
    let r1 = runEpochLoop(epochs: 2, stepsPerEpoch: 10) { ep, step in
        ep == 0 && step == 3
    }
    expectEqual(r1.epochsDone, 0, "отмена в середине эпохи 0 → 0 целых эпох")
    expectEqual(r1.stepsDone, 3, "отмена на шаге 3 → ровно 3 шага (не 10)")
    expect(r1.stoppedByUser, "отмечено stoppedByUser")

    // Без отмены: все эпохи и шаги выполнены.
    let r2 = runEpochLoop(epochs: 3, stepsPerEpoch: 5) { _, _ in false }
    expectEqual(r2.epochsDone, 3, "без отмены — все 3 эпохи")
    expectEqual(r2.stepsDone, 15, "без отмены — 15 шагов")
    expect(!r2.stoppedByUser, "без отмены — stoppedByUser = false")

    // Отмена ровно на границе эпох (ep=1, step=0): эпоха 0 целиком, потом стоп.
    let r3 = runEpochLoop(epochs: 4, stepsPerEpoch: 6) { ep, step in ep == 1 && step == 0 }
    expectEqual(r3.epochsDone, 1, "отмена на старте эпохи 1 → 1 целая эпоха")
    expectEqual(r3.stepsDone, 6, "отмена на старте эпохи 1 → 6 шагов эпохи 0")
    expect(r3.stoppedByUser, "остановлено по пользователю")
}

/// Сквозной интеграционный тест продуктового входа обучения (`trainConvVAE` —
/// его вызывает `/api/train` и кнопка): доказывает, что «Стоп» обрывает цикл
/// ПОКАДРОВО, а не на границе эпох.
///
/// Нюанс: `onStep` шлёт отчёт о шаге троттлингом — только на `done % 5 == 0`
/// (UI-каденс, см. `emitLiveStepReport`). Поэтому первый же `.training`-отчёт
/// приходит на done=5. Ставим отмену там. Эпоха длится `steps`=12 шагов:
///   • Эталон «проверка только на границе эпох» досчитал бы эпоху до 12
///     (отчёты на done=5 И done=10) и оборвал бы лишь на старте эпохи 2.
///   • Покадровый цикл обязан оборваться сразу после done=5 (один шаг уже шёл
///     во время постановки флага) → отчётов РОВНО 1, done=5 < 12, эпоха не
///     завершена, бросается CancellationError. Это и есть доказательство.
func testTrainConvVAEPromptCancel() {
    section("Интеграция: trainConvVAE прерывается по отмене посреди эпохи (покадровый Стоп)")
    guard gHasMetal else { skip("Metal недоступен"); return }

    let w = 64, h = 48
    var frames = [MemoryFrame]()
    for f in 0..<4 {
        var rng = SplitMix64(seed: UInt64(7700 + f))
        let img = makeSyntheticScene(w: w, h: h, rng: &rng)
        guard let png = encodePNG(width: w, height: h, rgb: img) else {
            expect(false, "encodePNG кадра \(f)"); return
        }
        frames.append(MemoryFrame(name: "f\(f).png", jpeg: png))
    }

    let base = "/tmp/yume_cancel_ckpt"
    try? FileManager.default.removeItem(atPath: base)
    let stepsPerEpoch = 12
    var p = Params()
    p.width = w; p.height = h
    p.epochs = 40
    p.steps = stepsPerEpoch   // batch=1 → epochSteps = p.steps
    p.batch = 1
    p.lr = 0.002
    p.latentChannels = 16
    p.modelWidth = 1
    p.convPath = "\(base)/conv.safetensors"
    p.vaePath = "\(base)/vae.safetensors"
    p.diffPath = "\(base)/diff.safetensors"
    p.outPrefix = "\(base)/outputs/gen"
    p.memoryFrames = frames
    p.controlFrames = frames
    p.resume = false

    let tc = TrainingControl.shared
    tc.reset()

    var maxDoneSeen = 0
    var trainingStepReports = 0
    var cancelFired = false

    // onStep приходит на потоке обучения → отмена синхронна, гонки по времени нет.
    let onStep: (TrainStepReport) -> Void = { rep in
        guard rep.stage == .training else { return }
        trainingStepReports += 1
        if rep.done > maxDoneSeen { maxDoneSeen = rep.done }
        if !cancelFired {
            cancelFired = true
            // graceSeconds заведомо больше времени до ближайшей проверки: watchdog
            // не успеет выстрелить — цикл оборвётся сам на ближайшем шаге.
            tc.cancel(graceSeconds: 120)
        }
    }

    var didThrow = false
    do {
        try trainConvVAE(p: p, onStep: onStep)
    } catch {
        didThrow = true
    }

    tc.reset()   // снять флаг для последующих тестов

    expect(cancelFired, "отмена запрошена из onStep на первом шаге-отчёте")
    expect(didThrow, "trainConvVAE прерван (бросил ошибку отмены)")
    // Ровно один шаговый отчёт (done=5) и обрыв: эпоха до done=10 НЕ дошла.
    expectEqual(trainingStepReports, 1,
                "ровно 1 шаговый отчёт до обрыва (не досчитывал эпоху до 10)")
    expect(maxDoneSeen == 5, "обрыв на done=5 (первый отчёт), не дальше")
    expect(maxDoneSeen < stepsPerEpoch,
           "эпоха НЕ дочитана (done=\(maxDoneSeen) < шагов в эпохе \(stepsPerEpoch)) → проверка на каждом шаге")
}

/// Стратегия watchdog по зависшему command buffer: чистая функция решений.
/// Адвокат дьявола: здесь НЕТ варианта «перезапустить шаг», и это правильно —
/// перезапуск забитого буфера утаскивает WindowServer (была перезагрузка).
func testWatchdogRestartVsExitPolicy() {
    section("Watchdog: решение ждать/убить (без небезопасного перезапуска шага)")

    // Буфер реально completed — зависания нет, ждём (данные читает status-путь).
    expectEqual(watchdogDecision(bufferCompleted: true, elapsedSeconds: 999, deadlineSeconds: 30),
                .keepWaiting, "completed-буфер — не убиваем")

    // Буфер не завершён, запас времени есть — ждём дальше.
    expectEqual(watchdogDecision(bufferCompleted: false, elapsedSeconds: 5, deadlineSeconds: 30),
                .keepWaiting, "завис, но время есть — ждём")

    // Буфер не завершён, дедлайн исчерпан — убиваем процесс (очистить очередь).
    expectEqual(watchdogDecision(bufferCompleted: false, elapsedSeconds: 30, deadlineSeconds: 30),
                .exitProcess, "завис сверх дедлайна — exit")

    // Граница: ровно на дедлайне — уже exit.
    expectEqual(watchdogDecision(bufferCompleted: false, elapsedSeconds: 30.0, deadlineSeconds: 30),
                .exitProcess, "граница дедлайна — exit")
    // Чуть раньше дедлайна — ещё ждём.
    expectEqual(watchdogDecision(bufferCompleted: false, elapsedSeconds: 29.9, deadlineSeconds: 30),
                .keepWaiting, "чуть раньше дедлайна — ждём")
}

/// Фаза 1 (п. «стабилизация на высокой точности»): на рабочих разрешениях
/// 128×72 и 192×108 кросс-кроп стабилизации НЕ должен сжимать рабочую область
/// к «мылу» 80×56 и обязан оставить размер, кратный 8 (иначе латент VAE /8 не
/// сойдётся по форме). Property-тест без GPU: синтез кадров со сдвигом камеры.
func testStabilizeHighResWorkingArea() {
    section("Фаза 1: стабилизация на высоких разрешениях — область > 80×56 и кратна 8")

    func baseFrame(_ w: Int, _ h: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return makeSyntheticScene(w: w, h: h, rng: &rng)
    }
    // Кадр со сдвигом камеры «вправо на s» (края клампим).
    func shifted(_ b: [Float], _ w: Int, _ h: Int, _ s: Int) -> [Float] {
        var f = [Float](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            for x in 0..<w {
                let sx = min(w - 1, max(0, x + s))
                let si = (y * w + sx) * 3, di = (y * w + x) * 3
                f[di] = b[si]; f[di + 1] = b[si + 1]; f[di + 2] = b[si + 2]
            }
        }
        return f
    }

    // Два рабочих разрешения из Фазы 1.
    for (w, h) in [(128, 72), (192, 108)] {
        let b = baseFrame(w, h, seed: 0x51A)
        let frames = [-3, 0, 2].map { shifted(b, w, h, $0) }
        let (aligned, cw, ch, report) = stabilizeFrames(frames, w: w, h: h, maxShift: 8)

        expect(cw % 8 == 0 && ch % 8 == 0,
               "\(w)×\(h): рабочая область \(cw)×\(ch) кратна 8 (VAE /8)")
        expect(cw >= 80 && ch >= 56,
               "\(w)×\(h): область \(cw)×\(ch) НЕ сжата к мылу (≥ 80×56)")
        expect(cw > w / 2 && ch > h / 2,
               "\(w)×\(h): сохранено > половины (кроп по maxShift=8, а не по половине)")
        expect(!aligned.isEmpty, "\(w)×\(h): хотя бы один кадр выровнен")
        expect(aligned.allSatisfy { $0.count == cw * ch * 3 },
               "\(w)×\(h): каждый выходной кадр размера \(cw)×\(ch)×3")
        expect(report.width == cw && report.height == ch,
               "\(w)×\(h): report соответствует рабочей области")
    }
}

/// Вскрывается живым E2E: `interpolateVideo` строил VAE ДЕФОЛТНОЙ архитектуры
/// (`ConvVAEGPU(engine:)` = plain/width=1) и грузил в него веса residual-модели →
/// `vae.load` бросал «checkpoint: несовпадение формы слоя N». Интерполяция любой
/// residual/groupNorm/широкой модели была сломана, хотя реконструкции этот путь
/// уже умели (читают archVersion/modelWidth из метаданных чекпоинта).
/// Тест: сохранить residual-чекпоинт (archVersion=2) → интерполировать → не бросает,
/// кадры produced. Красный до фикса, зелёный после.
func testInterpolationUsesCheckpointArch() {
    section("Фаза 5/E2E: interpolateVideo строит VAE из архитектуры чекпоинта (residual)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared

    // Residual-VAE (archVersion 2), дефолтные width=1/latent=16.
    let vae = ConvVAEGPU(engine: engine, residual: true, groupNorm: false)
    let dir = "/tmp/yume_interp_ckpt"
    try? FileManager.default.removeItem(atPath: dir)
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    let ckpt = "\(dir)/conv.safetensors"
    try! saveState(vae.matrices, to: ckpt, epoch: 0, step: 0, totalSteps: 1,
                   dataFingerprint: "interp-test",
                   modelWidth: vae.width, latentChannels: vae.latentChannels,
                   archVersion: 2)

    // Две опорные картинки 16×16 (кратно 8).
    func png(_ seed: UInt64) -> String {
        var rng = SplitMix64(seed: seed)
        let img = makeSyntheticScene(w: 16, h: 16, rng: &rng)
        let path = "\(dir)/img_\(seed).png"
        FileManager.default.createFile(atPath: path, contents: encodePNG(width: 16, height: 16, rgb: img))
        return path
    }
    let req = InterpolationRequest(
        imageAPath: png(0xAA), imageBPath: png(0xBB),
        seconds: 0.5, fps: 24,
        outputPath: "\(dir)/out.mp4",
        modelWidth: 16, modelHeight: 16,
        checkpointPath: ckpt)

    var thrown: String? = nil
    var produced = 0
    do {
        let result = try interpolateVideo(req)
        produced = result.frameCount
    } catch {
        thrown = String(describing: error)
    }

    expect(thrown == nil, "residual-интерполяция не падает по форме чекпоинта (было: \(thrown ?? "нет"))")
    expect(produced > 0, "интерполяция выдала кадры (получено \(produced))")
    // Выходной файл создан (то, из-за чего падал E2E: не было .mp4 → ffprobe FAIL).
    expect(FileManager.default.fileExists(atPath: req.outputPath),
           "создан выходной mp4: \(req.outputPath)")
}
