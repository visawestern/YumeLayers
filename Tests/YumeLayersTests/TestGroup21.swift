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
