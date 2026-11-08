// YumeLayers — группа тестов (выделено из main.swift).
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import Metal
import AVFoundation
import UniformTypeIdentifiers
import Accelerate
import VideoToolbox
import IOKit

func testProcessedMarks() {
    section("Отметки обработанных кадров")

    // Пустой массив выравнивается по числу кадров.
    var p: [Bool] = []
    let added = markProcessedFrames(names: ["a", "b", "c"],
                                    reconNames: ["a", "c"], processed: &p)
    expectEqual(added, 2, "отмечено 2 из 3")
    expectEqual(p, [true, false, true], "метки по именам")

    // Повторный вызов не добавляет новых отметок.
    let again = markProcessedFrames(names: ["a", "b", "c"],
                                    reconNames: ["a", "c"], processed: &p)
    expectEqual(again, 0, "повторно — 0 новых")
    expectEqual(p, [true, false, true], "метки не сброшены")

    // Догрузка: появились новые кадры — старые метки сохраняются.
    let added2 = markProcessedFrames(names: ["a", "b", "c", "d"],
                                     reconNames: ["d"], processed: &p)
    expectEqual(added2, 1, "новый кадр отмечен")
    expectEqual(p, [true, false, true, true], "старые метки на месте")

    // Реконструкций нет — ничего не отмечаем.
    var q: [Bool] = []
    expectEqual(markProcessedFrames(names: ["x"], reconNames: [], processed: &q), 0,
                "нет реконструкций → 0")
    expectEqual(q, [false], "флаг остался false")

    // Список кадров короче — лишние метки отбрасываются (префикс сохраняется).
    var r: [Bool] = [true, true, true, true]
    markProcessedFrames(names: ["a", "b"], reconNames: ["a"], processed: &r)
    expectEqual(r, [true, true], "длина приведена к числу кадров, префикс сохранён")
}

// MARK: - 7h. Воспроизводимость


func testReproducibility() {
    section("Воспроизводимость")

    // Синтетическая сцена должна быть одинаковой между вызовами: иначе
    // convcheck/convprofile нельзя сравнивать между прогонами.
    let a = makeNormalSynthetic(w: 32, h: 24)
    let b = makeNormalSynthetic(w: 32, h: 24)
    expectEqual(maxAbsDiff(a, b), 0, "makeNormalSynthetic детерминирован")
    expectEqual(a.count, 32 * 24 * 3, "размер сцены = w×h×3")

    // Явный RNG всё ещё даёт разные сцены — вариативность не потеряна.
    var r1 = SplitMix64(seed: 1)
    var r2 = SplitMix64(seed: 2)
    let s1 = makeSyntheticScene(w: 32, h: 24, rng: &r1)
    let s2 = makeSyntheticScene(w: 32, h: 24, rng: &r2)
    expect(maxAbsDiff(s1, s2) > 1e-6, "разные seed → разные сцены")

    // SplitMix64 воспроизводим и не выдаёт константу.
    var g1 = SplitMix64(seed: 42), g2 = SplitMix64(seed: 42)
    let seq1 = (0..<8).map { _ in g1.next() }
    let seq2 = (0..<8).map { _ in g2.next() }
    expectEqual(seq1, seq2, "SplitMix64: одинаковый seed → одинаковая последовательность")
    expect(Set(seq1).count == 8, "SplitMix64 не залипает на одном значении")

    // Две модели с одним seed инициализируются одинаково.
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let v1 = ConvVAEGPU(engine: engine, seed: 777)
    let v2 = ConvVAEGPU(engine: engine, seed: 777)
    let v3 = ConvVAEGPU(engine: engine, seed: 778)
    expectEqual(maxAbsDiff(v1.e1.W.toArray(), v2.e1.W.toArray()), 0,
                "одинаковый seed → одинаковые веса")
    expect(maxAbsDiff(v1.e1.W.toArray(), v3.e1.W.toArray()) > 1e-9,
           "разный seed → разные веса")
}

// MARK: - 7i. Совместимость чекпоинта с архитектурой


func testCheckpointCompat() {
    section("Чекпоинт: совместимость архитектуры")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine)

    // Корректный набор: W0,b0..W5,b5 в формах текущей модели.
    func correctMats() -> [Matrix] {
        vae.layers.flatMap { l in
            [Matrix(rows: l.oc, cols: l.ic * l.k * l.k, fill: 0.01),
             Matrix(rows: l.oc, cols: 1, fill: 0)]
        }
    }

    do {
        try vae.load(from: correctMats())
        expect(true, "корректные формы загружаются")
    } catch {
        expect(false, "корректные формы загружаются — \(error.localizedDescription)")
    }

    // Слишком мало матриц → код 10.
    do {
        try vae.load(from: Array(correctMats().prefix(6)))
        expect(false, "мало матриц → ошибка")
    } catch {
        expect(isIncompatibleCheckpoint(error), "мало матриц → распознано как несовместимость")
        expectEqual((error as NSError).code, 10, "мало матриц → код 10")
    }

    // Чекпоинт от прежней архитектуры: e3 был 64 канала вместо 2·C, d1 — 64 на входе.
    do {
        var old = correctMats()
        old[4] = Matrix(rows: 64, cols: 32 * 9, fill: 0.01)   // e3.W: старые 64 канала
        old[6] = Matrix(rows: 32, cols: 64 * 9, fill: 0.01)   // d1.W: старые 64 на входе
        try vae.load(from: old)
        expect(false, "старая архитектура → ошибка")
    } catch {
        expect(isIncompatibleCheckpoint(error), "старая архитектура → распознано как несовместимость")
        expectEqual((error as NSError).code, 11, "старая архитектура → код 11")
    }

    // Чужая ошибка (не от load) не должна считаться несовместимостью.
    let ioError = NSError(domain: NSCocoaErrorDomain, code: 260,
                          userInfo: [NSLocalizedDescriptionKey: "нет файла"])
    expect(!isIncompatibleCheckpoint(ioError), "ошибка ввода-вывода — не несовместимость")

    // Смещение формы: верное число матриц, но неверная ширина латента у d2.
    do {
        var broken = correctMats()
        broken[9] = Matrix(rows: 16, cols: 32 * 9 + 1, fill: 0)
        try vae.load(from: broken)
        expect(false, "битая форма d2 → ошибка")
    } catch {
        expect(isIncompatibleCheckpoint(error), "битая форма d2 → распознано")
    }
}

// MARK: - 7j. Журнал обучения не спамит нулём


func testTrainStepLogging() {
    section("Журнал обучения: без спама нулём")

    // Служебный отчёт (подготовка / догрузка VRAM) — done = 0. Не логируется,
    // иначе на каждую догрузку окна приходит 64 одинаковые строки «шаг 0».
    expect(!shouldLogTrainStep(done: 0, total: 129561), "done=0 — служебный отчёт, не пишем")
    expect(!shouldLogTrainStep(done: 0, total: 0), "done=0 при total=0 — не пишем")

    // Настоящие шаги: каждый 200-й.
    expect(shouldLogTrainStep(done: 200, total: 129561), "шаг 200 — пишем")
    expect(!shouldLogTrainStep(done: 201, total: 129561), "шаг 201 — не пишем")
    expect(shouldLogTrainStep(done: 2000, total: 129561), "шаг 2000 — пишем")
    expect(!shouldLogTrainStep(done: 1, total: 129561), "шаг 1 — не пишем")

    // Последний шаг пишется всегда, даже если не кратен 200.
    expect(shouldLogTrainStep(done: 129561, total: 129561), "последний шаг — пишем")
    expect(!shouldLogTrainStep(done: 129560, total: 129561), "предпоследний — не пишем")

    // Нулевой интервал не должен приводить к делению на ноль.
    expect(shouldLogTrainStep(done: 7, total: 100, every: 0), "every=0 → не падаем")

    // Сколько строк даст догрузка окна: 1024 кадра / 16 = 64 отчёта, все done=0.
    let spam = (0..<64).filter { _ in shouldLogTrainStep(done: 0, total: 129561) }.count
    expectEqual(spam, 0, "догрузка окна VRAM больше не пишет ни строки")

    // А за 1000 шагов должно появиться ровно 5 строк (200,400,600,800,1000).
    let lines = (1...1000).filter { shouldLogTrainStep(done: $0, total: 129561) }.count
    expectEqual(lines, 5, "за 1000 шагов — 5 строк прогресса")
}

// MARK: - 7b. Термо-регулятор GPU


func testThermalGovernor() {
    section("Термо-регулятор GPU")
    testDutySleep()
    testTargetDuty()
    testHeatLevel()
    testHeatDriftNoTemp()
    testHeatDebouncer()
    testStepBaseline()
    testThrottleDecision()
}

func testDutySleep() {
    // --- Доля работы: сколько спать после шага ---
    // GPU занят `gpuMs`, значит доля занятости = gpu/(gpu+sleep).
    // Чтобы уложиться в duty: sleep = gpu·(1/duty − 1).
    expectClose(dutySleepMs(gpuMs: 170, duty: 0.55, maxPauseMs: 2000), 170 * (1 / 0.55 - 1),
                0.01, "170 мс при duty 55% → пауза 139 мс")
    expectClose(dutySleepMs(gpuMs: 170, duty: 0.5, maxPauseMs: 2000), 170,
                0.01, "duty 50% → пауза равна времени работы")

    // Главная регрессия: пауза не должна быть нулевой на штатном режиме.
    // Раньше в цикле шага стоял жёсткий Thread.sleep(0.03) — 30 мс на кадр
    // 854×480 при 170 мс работы, то есть доля работы ~85% на много часов.
    expect(dutySleepMs(gpuMs: 170, duty: 0.55, maxPauseMs: 2000) > 100,
           "на штатном режиме пауза больше 100 мс, а не 30")

    // Вырожденные входы не должны давать NaN/отрицательных пауз.
    expectEqual(dutySleepMs(gpuMs: 0, duty: 0.55, maxPauseMs: 2000), 0, "нулевая работа → пауза 0")
    expectEqual(dutySleepMs(gpuMs: -5, duty: 0.55, maxPauseMs: 2000), 0, "отрицательное время → 0")
    expectEqual(dutySleepMs(gpuMs: .nan, duty: 0.55, maxPauseMs: 2000), 0, "NaN → 0")
    expectEqual(dutySleepMs(gpuMs: 170, duty: 0, maxPauseMs: 2000), 0, "duty 0 → не делим на ноль")
    expectEqual(dutySleepMs(gpuMs: 170, duty: 1, maxPauseMs: 2000), 0, "duty 100% → пауза 0")
    expectEqual(dutySleepMs(gpuMs: 170, duty: 1.5, maxPauseMs: 2000), 0, "duty >1 → 0")

    // Пауза ограничена сверху, иначе один шаг уснёт навсегда.
    expectEqual(dutySleepMs(gpuMs: 10_000, duty: 0.5, maxPauseMs: 2000), 2000,
                "пауза ограничена maxPauseMs")

    // 99% — это действительно 99%, а не «почти 99%»: на шаге 170 мс пауза
    // выходит 1.7 мс, то есть ~1% времени. Проверка ловит случай, когда
    // потолок доли работы урезали, а расчёт паузы остался прежним.
    let sleepAt99 = dutySleepMs(gpuMs: 170, duty: 0.99, maxPauseMs: 5000)
    expectClose(sleepAt99, 170 * (1 / 0.99 - 1), 1e-9, "99% → пауза по формуле")
    expect(sleepAt99 < 2.5, "99% → пауза меньше 2.5 мс на шаге 170 мс")

}

func testTargetDuty() {
    // --- Целевая доля работы по уровню нагрева ---
    let dn = targetDuty(for: .nominal)
    let dw = targetDuty(for: .warm)
    let dh = targetDuty(for: .hot)
    let dc = targetDuty(for: .critical)
    expect(dn < 1 && dn > 0.2, "штатная доля работы в разумных пределах")
    expect(dn > dw && dw > dh && dh > dc, "чем жарче, тем ниже доля работы")
    expect(dc > 0, "критический уровень всё же не 0 — обучение не встаёт навсегда")

}

func testHeatLevel() {
    // --- Уровень нагрева ---
    // Без базовой линии дрейф измерить не из чего.
    expectEqual(heatLevel(stepMs: 500, baselineMs: 0, thermalRaw: 0), .nominal,
                "нет базовой линии → nominal")
    // Штатный разброс не должен поднимать тревогу.
    expectEqual(heatLevel(stepMs: 200, baselineMs: 170, thermalRaw: 0), .nominal,
                "дрейф 1.18× — это шум")
    expectEqual(heatLevel(stepMs: 220, baselineMs: 170, thermalRaw: 0), .warm,
                "дрейф 1.29× — warm")
    // Ровно тот случай из лога: 0.8 с против 0.2 с.
    expectEqual(heatLevel(stepMs: 800, baselineMs: 200, thermalRaw: 0), .critical,
                "дрейф 4× (как в логе 0.2→0.8 с) — critical")

    // Показания ОС перебивают дрейф.
    expectEqual(heatLevel(stepMs: 200, baselineMs: 200, thermalRaw: 2), .hot,
                "thermalState .serious → hot")
    expectEqual(heatLevel(stepMs: 200, baselineMs: 200, thermalRaw: 3), .critical,
                "thermalState .critical → critical")
    // Но и при спокойной ОС четырёхкратный дрейф уже тревога.
    expectEqual(heatLevel(stepMs: 200, baselineMs: 200, thermalRaw: 0), .nominal,
                "нет дрейфа и нет жара → nominal")

}

func testHeatDriftNoTemp() {
    // --- Дрейф без температуры — это не нагрев ---
    // Ровно случай из журнала 4 ноября: шаг 366 мс против базы 139 мс
    // (2.6×) при 64 °C. Раньше это давало «критично» на холодной карте.
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0, tempC: 64), .warm,
                "2.6× дрейф при 64 °C — это сброс по мощности, а не нагрев")
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0, tempC: 70), .warm,
                "при 70 °C — тоже не выше warm")
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0, tempC: 83), .hot,
                "при 83 °C дрейф уже можно считать нагревом, но не критичным")
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0, tempC: 86), .critical,
                "у потолка — критично, без скидок")
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0), .critical,
                "без датчика остаётся прежнее поведение по дрейфу")

}

func testHeatDebouncer() {
    // --- Гаситель дребезга ---
    // 328/366 мс на пороге 2.5× переключали уровень каждый шаг.
    var deb = HeatDebouncer(steps: 3)
    var seen: [HeatLevel] = []
    for i in 0..<10 {
        seen.append(deb.update(i % 2 == 0 ? .hot : .critical))
    }
    expect(seen.allSatisfy { $0 == .nominal },
           "чередование горячо/критично не меняет уровень (дребезг погашен)")

    // Устойчивый переход принимается ровно после `steps` замеров.
    var deb2 = HeatDebouncer(steps: 3)
    expectEqual(deb2.update(.hot), .nominal, "1-й замер — ещё не переключились")
    expectEqual(deb2.update(.hot), .nominal, "2-й замер — ещё не переключились")
    expectEqual(deb2.update(.hot), .hot, "3-й замер — переключились")
    expectEqual(deb2.update(.hot), .hot, "дальше держится")
    // Одиночный выброс вниз не сбрасывает уровень.
    expectEqual(deb2.update(.nominal), .hot, "одиночный сброс проигнорирован")
    expectEqual(deb2.update(.nominal), .hot, "два сброса подряд — ещё держим")
    expectEqual(deb2.update(.nominal), .nominal, "три подряд — снизились")

}

func testStepBaseline() {
    // --- Базовая линия шага ---
    expectEqual(updateStepBaseline(0, sample: 170), 170, "первый замер задаёт базу")
    expectEqual(updateStepBaseline(170, sample: 200), 170, "база не растёт от медленного шага")
    expectEqual(updateStepBaseline(170, sample: 100), 100, "база опускается к быстрому шагу")
    expectEqual(updateStepBaseline(170, sample: Double.nan), 170, "NaN не портит базу")
    expectEqual(updateStepBaseline(170, sample: 0), 170, "ноль не портит базу")

}

func testThrottleDecision() {
    // --- Решение на шаг (это вызывает цикл обучения) ---
    // Первый шаг: базы нет, ОС спокойна, пользовательская пауза 250 мс.
    let d1 = throttleDecision(gpuMs: 170, wallMs: 170, baselineMs: 0, thermalRaw: 0,
                              floorMs: 250, maxPauseMs: 2000)
    expectEqual(d1.level, .nominal, "первый шаг — норма")
    expectEqual(d1.sleepMs, 250, "пауза не меньше пользовательской")
    expectClose(d1.baselineMs, 170, 1e-9, "база запомнила первый замер")

    // Без пользовательской паузы работает чистая доля работы.
    let d2 = throttleDecision(gpuMs: 170, wallMs: 170, baselineMs: 0, thermalRaw: 0,
                              floorMs: 0, maxPauseMs: 2000)
    expectClose(d2.sleepMs, 139.09, 0.01, "пауза по доле работы 55%")

    // Тот же кадр, но GPU разогрелась: тепловой уровень режет долю работы.
    let d3 = throttleDecision(gpuMs: 170, wallMs: 170, baselineMs: 170, thermalRaw: 3,
                              floorMs: 0, maxPauseMs: 2000)
    expectEqual(d3.level, .critical, "critical по показаниям ОС")
    expectClose(d3.sleepMs, 170 * (1 / 0.12 - 1), 0.5, "на critical пауза в 7 раз больше работы")

    // Дрейф из лога: 0.8 с против базы 0.2 с — регулятор обязан среагировать
    // без всяких показаний ОС.
    let d4 = throttleDecision(gpuMs: 700, wallMs: 800, baselineMs: 200, thermalRaw: 0,
                              floorMs: 250, maxPauseMs: 2000)
    expectEqual(d4.level, .critical, "дрейф 4× → critical без показаний ОС")
    expect(d4.sleepMs > 1000, "на дрейфе пауза больше секунды")

    // Потолок соблюдается всегда, даже на критическом уровне.
    let d5 = throttleDecision(gpuMs: 5000, wallMs: 5000, baselineMs: 100, thermalRaw: 3,
                              floorMs: 250, maxPauseMs: 2000)
    expectEqual(d5.sleepMs, 2000, "пауза не превышает maxPauseMs")

    // База не портится медленным шагом: иначе дрейф перестанет быть виден.
    let d6 = throttleDecision(gpuMs: 170, wallMs: 900, baselineMs: 170, thermalRaw: 0,
                              floorMs: 0, maxPauseMs: 2000)
    expectClose(d6.baselineMs, 170, 1e-9, "медленный шаг не поднимает базу")
    expect(d6.level >= .hot, "медленный шаг распознан как нагрев")
}


// MARK: - 7c. Замкнутый контур по температуре GPU


func testPowerController() {
    section("Контур по температуре GPU")
    testDutyCeiling()
    testPowerNoTelemetryAndThermal()
    testPowerClosedLoop()
    testFreqNoiseIgnored()
    testParseGpuStats()
    testCalibrationThresholds()
    testDriftVsHeat()
    testEpochCoolPause()
}

func testDutyCeiling() {
    // Потолок из «мощности» в процентах.
    expectClose(dutyCeiling(powerPercent: 100), 0.98, 1e-9, "100% → 0.98")
    expectClose(dutyCeiling(powerPercent: 85), 0.833, 1e-9, "85% → 0.833")
    expectClose(dutyCeiling(powerPercent: 50), 0.49, 1e-9, "50% → 0.49")
    expectClose(dutyCeiling(powerPercent: 0), 0.10, 1e-9, "0% → нижний предел 0.10")
    expectClose(dutyCeiling(powerPercent: 200), 0.98, 1e-9, "больше 100% → 0.98")
    // 100% — это не «без паузы вообще»: 2% простоя оставлены WindowServer.
    expect(maxSafeDuty < 1.0, "даже 100% оставляет слоты WindowServer")
    expectClose(maxSafeDuty, 0.98, 1e-9, "100% мощности → 98% доли работы")

}

func testPowerNoTelemetryAndThermal() {
    // Без телеметрии режим не меняем: гадать хуже, чем держать текущий.
    var none = PowerController(ceiling: 0.765, startDuty: 0.5)
    expectClose(none.update(stats: nil), 0.5, 1e-9, "нет датчика — доля работы та же")

    // Перегрев: вниз, но не больше 5% за замер — иначе нагрузка «дёргается».
    var hot = PowerController(ceiling: 0.9, startDuty: 0.6)
    let h1 = hot.update(stats: GpuStats(temperatureC: 90, coreClockMHz: 900,
                                        activityPercent: 95, powerW: 45))
    expectClose(h1, 0.55, 1e-9, "90 °C → минус 5% за замер")

    // Холодно: поднимается до потолка и не выше.
    var cold = PowerController(ceiling: 0.765, startDuty: 0.5)
    var d = 0.5
    for _ in 0..<40 {
        d = cold.update(stats: GpuStats(temperatureC: 60, coreClockMHz: 1400,
                                        activityPercent: 95, powerW: 40))
    }
    expectClose(d, 0.765, 1e-9, "на холоде упирается в потолок, а не в 1.0")

}

func testPowerClosedLoop() {
    // Замкнутый контур с простой тепловой моделью: должен сойтись и не
    // раскачаться. Это главная проверка — именно раскачка и перегревала.
    var loop = PowerController(ceiling: 0.9, startDuty: 0.9)
    var temp = 70.0
    var duties: [Double] = []
    for _ in 0..<200 {
        temp += (65.0 + 25.0 * loop.duty - temp) * 0.15
        duties.append(loop.update(stats: GpuStats(temperatureC: temp, coreClockMHz: 1200,
                                                  activityPercent: 95, powerW: 40)))
    }
    let settled = duties.suffix(20).reduce(0, +) / 20
    let earlier = duties[duties.count - 60]
    expect(abs(settled - earlier) < 0.06, "контур сходится, а не раскачивается")
    expect(temp <= 85.5, "температура не выходит за потолок 85 °C")
    expect(duties.min()! > 0.05, "нагрузка не проваливается в ноль")

}

func testFreqNoiseIgnored() {
    // Детектор троттлинга по частоте сознательно убран: замер показал, что
    // Core Clock в PerformanceStatistics скачет 0…468 МГц на реальной
    // нагрузке (срезы попадают в промежутки между command buffer'ами).
    // Решение по такому сигналу — реакция на шум, поэтому проверяем,
    // что шум частоты долю работы НЕ меняет.
    var noise = PowerController(ceiling: 0.9, startDuty: 0.8)
    _ = noise.update(stats: GpuStats(temperatureC: 65, coreClockMHz: 1400,
                                     activityPercent: 95, powerW: 40))
    let noisyDuty = noise.update(stats: GpuStats(temperatureC: 65, coreClockMHz: 40,
                                                 activityPercent: 2, powerW: 5))
    expect(noisyDuty > 0.8, "падение частоты в срезе не считается троттлингом")
    expectClose(noise.peakClockMHz, 1400, 1e-9, "пик частоты — только для журнала")

}

func testParseGpuStats() {
    // --- Разбор словаря телеметрии ---
    let raw: [String: Any] = ["Temperature(C)": 67, "Core Clock(MHz)": 268,
                              "GPU Activity(%)": 18, "Total Power(W)": 17]
    let st = parseGpuStats(raw)
    expect(st != nil, "словарь разобран")
    expectEqual(Int(st?.temperatureC ?? 0), 67, "температура прочитана")
    expectEqual(Int(st?.coreClockMHz ?? 0), 268, "частота прочитана")
    expectEqual(Int(st?.powerW ?? 0), 17, "мощность прочитана")
    expect(st?.hasTemperature == true, "температура осмысленна")

    // Без температуры запись бесполезна (узел без телеметрии).
    expect(parseGpuStats(["Core Clock(MHz)": 300]) == nil, "без температуры → nil")
    expect(parseGpuStats(["Temperature(C)": 0]) == nil, "нулевая температура → nil")
    // Значения могут прийти строками — разбор должен выдержать.
    expectEqual(Int(parseGpuStats(["Temperature(C)": "71"])?.temperatureC ?? 0), 71,
                "температура строкой разобрана")

}

func testCalibrationThresholds() {
    // Пороги откалиброваны замерами `yume gpuheat`: при 74 °C на краю
    // кристалла шаг уже замедлялся, значит запас кончился. 93 °C — точка, на
    // которой машина уходит в перезагрузку, поэтому 85 °C — жёсткий потолок,
    // а 82 °C — рабочая точка с тремя градусами на реакцию регулятора.
    let defaults = PowerController(ceiling: 0.765)
    expectClose(defaults.targetTempC, 82, 1e-9, "рабочая точка 82 °C")
    expectClose(defaults.maxTempC, 85, 1e-9, "потолок 85 °C — выше уходить нельзя")
    expectClose(emergencyTempC, 86, 1e-9, "аварийный порог 86 °C")
    expect(defaults.targetTempC < defaults.maxTempC, "рабочая точка ниже потолка")
    expect(emergencyTempC > defaults.maxTempC, "аварийный порог выше потолка")

}

func testDriftVsHeat() {
    // Дрейф считается по времени GPU, поэтому собственная пауза не может
    // выглядеть как нагрев. Иначе: больше пауза → больше дрейф → ещё пауза.
    var base = 0.0
    var steady = HeatLevel.nominal
    for _ in 0..<30 {
        let d = throttleDecision(gpuMs: 140, wallMs: 140, baselineMs: base, thermalRaw: 0,
                                 floorMs: 250, maxPauseMs: 2000)
        base = d.baselineMs
        steady = d.level
    }
    expectEqual(steady, .nominal, "стабильное время GPU не даёт ложного нагрева")

    // А настоящий рост времени GPU — даёт.
    var b2 = 0.0
    var risen: HeatLevel = .nominal
    for i in 0..<10 {
        let ms: Double = i < 5 ? 140 : 260
        let d = throttleDecision(gpuMs: ms, wallMs: ms, baselineMs: b2, thermalRaw: 0,
                                 floorMs: 0, maxPauseMs: 2000)
        b2 = d.baselineMs
        risen = d.level
    }
    expect(risen >= .warm, "рост времени GPU с 140 до 260 мс распознан как нагрев")

}

func testEpochCoolPause() {
    // --- Пауза между эпохами: только по теплу ---
    // Пауза — тепловой инструмент, а не часть расписания. Холодная карта
    // не должна «остывать»: замер на 99% показал, что она и так не греется.
    expectEqual(epochCoolPauseMs(tempC: 60, pauseMs: 250), 0,
                "60 °C → паузы между эпохами нет")
    expectEqual(epochCoolPauseMs(tempC: 81, pauseMs: 250), 0,
                "ниже рабочей точки → паузы нет")
    expectClose(epochCoolPauseMs(tempC: 82, pauseMs: 250), 500, 1e-9,
                "рабочая точка → половина паузы")
    expectClose(epochCoolPauseMs(tempC: 84, pauseMs: 250), 500, 1e-9,
                "между точкой и потолком → половина паузы")
    expectClose(epochCoolPauseMs(tempC: 85, pauseMs: 250), 1000, 1e-9,
                "у потолка → полная пауза, но не меньше 1000 мс")
    expectClose(epochCoolPauseMs(tempC: 88, pauseMs: 3000), 3000, 1e-9,
                "своя большая пауза уважается")
    // Нет датчика — не гадаем и не стоим: пошаговый регулятор всё равно
    // держит нагрузку, а эта пауза случается раз в эпоху.
    expectEqual(epochCoolPauseMs(tempC: nil, pauseMs: 250), 0, "нет датчика → 0")
    expectEqual(epochCoolPauseMs(tempC: .nan, pauseMs: 250), 0, "NaN → 0")
    expectEqual(epochCoolPauseMs(tempC: 0, pauseMs: 250), 0, "нулевая температура → 0")
}


// MARK: - 8. Чистые функции UI/логики (регрессия)


func testPureLogic() {
    section("Чистая логика")

    // Пагинатор.
    expectEqual(paginatorVisiblePages(current: 0, pages: 5, limit: 3), [0, 1, 2],
                "страницы: начало")
    expectEqual(paginatorVisiblePages(current: 4, pages: 5, limit: 3), [2, 3, 4],
                "страницы: конец")
    expectEqual(paginatorVisiblePages(current: 2, pages: 5, limit: 3), [1, 2, 3],
                "страницы: середина")
    expectEqual(paginatorVisiblePages(current: 0, pages: 0, limit: 3), [],
                "страницы: пусто")
    expectEqual(paginatorDotWidth(pageNumber: 9), 28, "ширина кружка: 1 цифра")
    expectEqual(paginatorDotWidth(pageNumber: 100), 34, "ширина кружка: 3 цифры")
    expectEqual(paginatorDotWidth(pageNumber: 1000), 40, "ширина кружка: 4 цифры")
    expectEqual(dotCapacity(columnWidth: 446, dotWidth: 28), 9, "вместимость 446/28")
    expectEqual(paginatorRowWidth(columnWidth: 494), 446, "ширина строки пагинатора")

    // Батчи.
    expectEqual(batchRanges(total: 10, batch: 4), [0..<4, 4..<8, 8..<10], "батчи 4/4/2")
    expectEqual(batchRanges(total: 0, batch: 4), [], "батчи: пусто")
    expectEqual(batchRanges(total: 3, batch: 8), [0..<3], "батчи: один неполный")

    // Контрольная выборка.
    expect(isControlFrame(index: 0, fps: 24), "контрольный: кадр 0")
    expect(isControlFrame(index: 24, fps: 24), "контрольный: кадр 24")
    expect(!isControlFrame(index: 23, fps: 24), "не контрольный: кадр 23")
    expect(isControlFrame(index: 0, fps: 0), "контрольный: fps=0 безопасен")

    // Прогресс.
    expectEqual(progressInFrames(stepDone: 10, totalSteps: 100, batch: 8).done, 80,
                "прогресс в кадрах: done")
    expectEqual(progressInFrames(stepDone: 10, totalSteps: 100, batch: 8).total, 800,
                "прогресс в кадрах: total")
    expectEqual(progressInFrames(stepDone: 10, totalSteps: 100, batch: 0).done, 10,
                "прогресс: батч 0 → 1")

    // Логирование прогресса.
    expect(shouldLogProgress(done: 0, total: 100, lastLoggedPercent: 0), "лог: старт")
    expect(shouldLogProgress(done: 100, total: 100, lastLoggedPercent: 95), "лог: финиш")
    expect(shouldLogProgress(done: 5, total: 100, lastLoggedPercent: 0), "лог: 5%")
    expect(!shouldLogProgress(done: 4, total: 100, lastLoggedPercent: 0), "лог: 4% не пишем")

    // Автотюнинг.
    let tuned = autoTuneValues(frames: 1000, batch: 4, resume: false,
                               currentSteps: 20, currentEpochs: 20, currentLr: 0.001,
                               stepsEditedByUser: false, epochsEditedByUser: false,
                               lrEditedByUser: false)
    expectEqual(tuned.steps, 250, "автотюнинг: шаги = кадры/батч")
    expectEqual(tuned.epochs, 3, "автотюнинг: эпохи = 3 при 1000 кадрах")

    // Живой статус.
    expectEqual(formatThousands(43187), "43 187", "разряды через пробел")
    expect(trainLivePhase(secondsSinceStep: 5, stepSeconds: 0.5) == .running,
           "статус: идёт")
    expect(trainLivePhase(secondsSinceStep: 95, stepSeconds: 0.5) == .stalled(seconds: 95),
           "статус: зависло")

    // Батч и фильтр.
    expectEqual(frameOffsetForSlot(0), 9, "коллаж: слот 0 → смещение 9")
    expectEqual(frameOffsetForSlot(5), 0, "коллаж: слот 5 → смещение 0")
}

// MARK: - Рендер слоёв (ground-truth разметка)


