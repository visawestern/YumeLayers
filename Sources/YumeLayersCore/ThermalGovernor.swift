// Термо-регулятор GPU.
//
// Зачем это отдельным файлом и с чистыми функциями: 4 ноября 2026 машина
// ушла в принудительную перезагрузку с паникой watchdogd
// («no successful checkins from WindowServer in 120 seconds»).
// Причина — не сбой шейдера, а доля работы: обучение держало дискретную
// GPU занятой почти всё время, WindowServer не получал доступа к тому же
// GPU и не успевал «отметиться» за 120 с. Watchdog выключил систему.
//
// В логе это видно как дрейф времени шага: 0.2 с/шаг в 10:02 → 0.8 с/шаг
// в 10:09 (4×) при 4815 МБ занятой VRAM. Раньше «щадящая пауза» 250 мс
// применялась ТОЛЬКО между эпохами (43 187 шагов), а внутри эпохи стоял
// жёсткий Thread.sleep(0.03) — то есть доля работы ~85% на десять часов.
//
// Здесь считаются три вещи, и все три — чистые функции, чтобы их можно
// было проверить без GPU и без ожидания реального перегрева:
//   1. heatLevel — насколько всё плохо (показания ОС + дрейф времени шага);
//   2. targetDuty — какую долю времени GPU позволено быть занятым;
//   3. dutySleepMs — сколько спать после шага, чтобы в эту долю уложиться.

import Foundation

/// Уровень нагрева. Порядок важен: сравнение через `Comparable`.
public enum HeatLevel: Int, Comparable, Equatable, Sendable {
    case nominal = 0
    case warm = 1
    case hot = 2
    case critical = 3

    public static func < (a: HeatLevel, b: HeatLevel) -> Bool { a.rawValue < b.rawValue }

    public var title: String {
        switch self {
        case .nominal:  return "норма"
        case .warm:     return "тепло"
        case .hot:      return "горячо"
        case .critical: return "критично"
        }
    }
}

/// Доля времени, которую GPU разрешено быть занятой на каждом уровне.
///
/// 55% на штатном режиме — это не «запас на всякий случай»: WindowServer
/// композитит окна на той же GPU, и ему нужны слоты между нашими шагами.
/// 12% на критическом — обучение продолжается (пусть и медленно), но GPU
/// большую часть времени свободна и успевает остыть.
public func targetDuty(for level: HeatLevel) -> Double {
    switch level {
    case .nominal:  return 0.55
    case .warm:     return 0.45
    case .hot:      return 0.30
    case .critical: return 0.12
    }
}

/// Пауза после шага, при которой GPU занята не более `duty` доли времени.
///
/// Если шаг занял `gpuMs`, то при паузе `s` доля работы равна `gpuMs/(gpuMs+s)`.
/// Отсюда `s = gpuMs·(1/duty − 1)`. Результат ограничен сверху `maxPauseMs`,
/// чтобы один шаг не уснул навсегда, и снизу нулём.
public func dutySleepMs(gpuMs: Double, duty: Double, maxPauseMs: Double) -> Double {
    guard gpuMs.isFinite, gpuMs > 0 else { return 0 }
    guard duty.isFinite, duty > 0, duty < 1 else { return 0 }
    guard maxPauseMs.isFinite, maxPauseMs > 0 else { return 0 }
    let s = gpuMs * (1.0 / duty - 1.0)
    guard s.isFinite, s > 0 else { return 0 }
    return min(s, maxPauseMs)
}

/// Уровень нагрева по двум независимым признакам.
///
/// `thermalRaw` — это `ProcessInfo.processInfo.thermalState.rawValue`
/// (0 nominal, 1 fair, 2 serious, 3 critical). Он запаздывает: macOS
/// сообщает о перегреве, когда он уже случился, поэтому второй признак —
/// дрейф времени шага относительно лучшего замера.
///
/// `tempC` — фактическая температура GPU. Она здесь не для красоты, а чтобы
/// не путать две разные причины замедления:
///
///   • нагрев — карта у потолка, сбрасывает частоту от температуры;
///   • сброс по мощности — карта холодная, но драйвер всё равно режет
///     частоту, потому что упёрся в бюджет питания.
///
/// Второе выглядит в логе точно как первое: шаг замедляется в 2.5 раза при
/// 64 °C. Без этой проверки регулятор честно докладывал «критично» на
/// холодной карте и дёргал нагрузку туда-сюда без всякой пользы.
public func heatLevel(stepMs: Double, baselineMs: Double, thermalRaw: Int,
                      tempC: Double? = nil) -> HeatLevel {
    // Показания ОС — старший приоритет: спорить с ними нечем.
    if thermalRaw >= 3 { return .critical }
    if thermalRaw >= 2 { return .hot }

    guard baselineMs.isFinite, baselineMs > 0,
          stepMs.isFinite, stepMs > 0 else { return .nominal }

    let ratio = stepMs / baselineMs
    var level: HeatLevel
    if ratio < 1.25 { level = .nominal }
    else if ratio < 1.75 { level = .warm }
    else if ratio < 2.5 { level = .hot }
    else { level = .critical }

    // Дрейф без температуры — это не нагрев. Холодная карта не может быть
    // «критично горячей»: значит частоту срезали по мощности, и снижать
    // нагрузку бессмысленно — она не вернёт частоту.
    if let t = tempC, t.isFinite, t > 0 {
        if t < defaultTargetTempC {
            level = min(level, .warm)
        } else if t < defaultMaxTempC {
            level = min(level, .hot)
        }
    }
    return level
}

/// Гаситель дребезга уровня нагрева.
///
/// Дрейф времени шага — шумная величина: соседние шаги дают 328 и 366 мс, а
/// порог между «горячо» и «критично» проходит ровно посередине (2.5× от
/// базы). Без гистерезиса уровень переключался на каждом шаге, и журнал
/// превращался в «горячо → критично → горячо» по десять раз в секунду.
///
/// Новый уровень принимается, только если продержался `steps` замеров подряд.
public struct HeatDebouncer {
    public private(set) var level: HeatLevel
    public let steps: Int
    private var pending: HeatLevel
    private var count: Int = 0

    public init(steps: Int = 3) {
        self.steps = max(1, steps)
        self.level = .nominal
        self.pending = .nominal
    }

    public mutating func update(_ raw: HeatLevel) -> HeatLevel {
        if raw == level {
            pending = raw
            count = 0
            return level
        }
        if raw == pending {
            count += 1
        } else {
            pending = raw
            count = 1
        }
        if count >= steps {
            level = raw
            count = 0
        }
        return level
    }
}

/// Обновление базовой (лучшей) длительности шага — скользящий минимум.
///
/// Минимум, а не среднее: среднее «съезжает» вместе с троттлингом, и тогда
/// дрейф перестаёт быть виден. Минимум фиксирует, на что машина способна
/// в холодном состоянии, и любой рост относительно него — уже сигнал.
public func updateStepBaseline(_ baseline: Double, sample: Double) -> Double {
    guard sample.isFinite, sample > 0 else { return baseline }
    guard baseline.isFinite, baseline > 0 else { return sample }
    return min(baseline, sample)
}

/// Готовое решение на один шаг: сколько спать и на каком мы уровне.
public struct ThrottleDecision: Equatable, Sendable {
    public let level: HeatLevel
    public let sleepMs: Double
    public let baselineMs: Double

    public init(level: HeatLevel, sleepMs: Double, baselineMs: Double) {
        self.level = level
        self.sleepMs = sleepMs
        self.baselineMs = baselineMs
    }
}

/// Свести всё вместе: замер шага + состояние ОС → пауза и новый baseline.
///
/// `floorMs` — нижняя граница паузы (пользовательская «пауза между шагами»):
/// даже если GPU отработала мгновенно, спать меньше неё не будем.
///
/// `duty` — если задан, доля работы берётся из него (замкнутый контур по
/// температуре), а не из уровня нагрева. Это основной режим.
///
/// `tempC` — фактическая температура GPU: без неё дрейф времени шага
/// невозможно отличить от нагрева (см. `heatLevel`).
public func throttleDecision(gpuMs: Double,
                             wallMs: Double,
                             baselineMs: Double,
                             thermalRaw: Int,
                             floorMs: Double,
                             maxPauseMs: Double,
                             duty: Double? = nil,
                             tempC: Double? = nil) -> ThrottleDecision {
    let sample = wallMs.isFinite && wallMs > 0 ? wallMs : gpuMs
    let base = updateStepBaseline(baselineMs, sample: sample)
    let level = heatLevel(stepMs: sample, baselineMs: base,
                          thermalRaw: thermalRaw, tempC: tempC)
    let effectiveDuty = duty ?? targetDuty(for: level)
    let byDuty = dutySleepMs(gpuMs: gpuMs, duty: effectiveDuty, maxPauseMs: maxPauseMs)
    let floor = floorMs.isFinite ? max(0, floorMs) : 0
    let sleep = min(max(byDuty, floor), max(maxPauseMs, 0))
    return ThrottleDecision(level: level, sleepMs: sleep, baselineMs: base)
}

// MARK: - Замкнутый контур по температуре

/// Доля работы при «100% мощности».
///
/// 0.98, а не 1.0: два процента паузы оставляем WindowServer — он композитит
/// на этой же GPU, и полностью непрерывная очередь рано или поздно доводит
/// до «no successful checkins» (паника 4 ноября 2026). При шаге 350 мс это
/// 7 мс на шаг — незаметно для темпа, но драйверу даёт слоты обслужить
/// чужую работу между нашими command buffer'ами.
///
/// Почему не 0.90, как было: замер `yume gpuheat 150 0` показал, что на 90%
/// карта держится 68–71 °C при цели 82 — потолок нагрузки и был тем, что
/// не давало догреться. Температурный контур при этом не отдыхал ни секунды.
///
/// Важно: 98% против 99% на время шага не влияет никак. Замедление шага
/// с 139 до 350 мс — это сброс частоты по мощности, а не наш простой
/// (1% от 350 мс = 3.5 мс). Проценты здесь про запас для системы, а не
/// про скорость.
public let maxSafeDuty = 0.98

// MARK: - Потолок доли работы = физический предел 98%

/// Рабочий потолок доли работы GPU.
///
/// `maxSafeDuty = 0.98` — физический предел, при котором WindowServer ещё
/// получает слоты между нашими command buffer'ами. Полностью непрерывная
/// очередь рано или поздно доводит до «no successful checkins» (паника
/// 4 ноября 2026), поэтому 2% паузы остаются системе.
///
/// Величина НЕ фиксирована на 90%: пользователь задаёт «Нагрузку GPU, %»,
/// и это значение читается во всех путях (обучение, прогрев, guard, API).
/// 98% означает 98% времени GPU занята, 2% уходит на остывание и композитинг.
public let safeDutyCap = maxSafeDuty

/// Значение «мощности GPU» по умолчанию в GUI и CLI (проценты).
public let defaultGpuPowerPercent = 90

/// Потолок доли работы с учётом физического предела.
///
/// Прямая семантика: **98% мощности = GPU занята 98% времени**.
/// Процент не масштабируется скрытыми коэффициентами: 85% → 0.85,
/// 98% → 0.98. Значения выше физического предела (0.98) упираются в него.
public func effectiveDutyCeiling(powerPercent: Int) -> Double {
    let p = Double(min(max(powerPercent, 0), 100)) / 100.0
    return min(max(p, 0.10), safeDutyCap)
}

/// Жёсткий предохранитель для циклов, которые принимают долю работы напрямую
/// (`runGpuHeat`), минуя «мощность в процентах» и `GpuDutyGuard`.
///
/// Некорректные значения (NaN, отрицательные) не превращаются в полную
/// нагрузку; допустимый максимум — физический предел `safeDutyCap`.
public func clampedGpuDuty(_ duty: Double) -> Double {
    guard duty.isFinite else { return 0.10 }
    return min(max(duty, 0.05), safeDutyCap)
}

/// Исполняемый предохранитель для ЛЮБОГО цикла обучения, включая короткие
/// тестовые прогоны. Без него даже 60 шагов подряд занимают очередь GPU
/// целиком — ровно так и начиналась авария 4 ноября.
public struct GpuDutyGuard {
    public let duty: Double
    public let maxPauseMs: Double

    public init(powerPercent: Int = defaultGpuPowerPercent,
                maxPauseMs: Double = 2000) {
        self.duty = effectiveDutyCeiling(powerPercent: powerPercent)
        self.maxPauseMs = max(0, maxPauseMs)
    }

    /// Пауза после шага длительностью `gpuMs`, мс. Чистая функция — тестируется.
    public func pauseMs(gpuMs: Double) -> Double {
        dutySleepMs(gpuMs: gpuMs, duty: duty, maxPauseMs: maxPauseMs)
    }

    /// Применить паузу на реальном потоке.
    public func wait(gpuMs: Double) {
        let ms = pauseMs(gpuMs: gpuMs)
        if ms > 0 { Thread.sleep(forTimeInterval: ms / 1000) }
    }

    /// Аварийное остывание, если датчик вышел за `emergencyTempC`.
    public func emergencyPauseMs(tempC: Double?) -> Double {
        guard let t = tempC, t.isFinite, t >= emergencyTempC else { return 0 }
        return maxPauseMs
    }
}

/// Пейсер доли работы для ДЛИННЫХ прогонов.
///
/// `GpuDutyGuard.wait` спит после КАЖДОГО command buffer. Когда шаг короткий
/// (80×56, ~1 мс GPU), положенная пауза ~0.1 мс, но `Thread.sleep` на macOS
/// стоит порядка нескольких мс и контекстных переключений: 8 000 шагов
/// превращались в 250 с вместо десятков секунд. Здесь паузы накапливаются
/// в долг и сбрасываются одним сном, когда долг >= `minSleepMs`. Итоговое
/// соотношение «GPU занята / всего» остаётся ровно 90% — меняется только
/// число обращений к планировщику, а не тепловой режим.
///
/// Все счётчики наблюдаемы: отчёт обучения показывает реально потраченное
/// время простоя, а не предположение.
public final class GpuDutyPacer {
    public let duty: Double
    public let maxPauseMs: Double
    public let minSleepMs: Double

    private var debtMs: Double = 0
    public private(set) var totalGpuMs: Double = 0
    public private(set) var totalPauseMs: Double = 0
    public private(set) var sleepCalls: Int = 0
    public private(set) var emergencyPauses: Int = 0

    public init(powerPercent: Int = defaultGpuPowerPercent,
                maxPauseMs: Double = 2000,
                minSleepMs: Double = 2.0) {
        self.duty = effectiveDutyCeiling(powerPercent: powerPercent)
        self.maxPauseMs = max(0, maxPauseMs)
        self.minSleepMs = max(0, minSleepMs)
    }

    public var pendingPauseMs: Double { debtMs }

    /// Зачесть один command buffer; при накоплении долга — уснуть один раз.
    @discardableResult
    public func wait(gpuMs: Double) -> Double {
        let g = (gpuMs.isFinite && gpuMs > 0) ? gpuMs : 0
        totalGpuMs += g
        debtMs += dutySleepMs(gpuMs: g, duty: duty, maxPauseMs: maxPauseMs)
        if debtMs >= minSleepMs { return sleepDebt() }
        return 0
    }

    /// Сбросить накопленный долг (вызывать в конце прогона).
    @discardableResult
    public func sleepDebt() -> Double {
        guard debtMs > 0 else { return 0 }
        let ms = debtMs
        Thread.sleep(forTimeInterval: ms / 1000)
        debtMs = 0
        totalPauseMs += ms
        sleepCalls += 1
        return ms
    }

    /// Аварийное остывание: НЕ откладывается в долг, применяется сразу.
    @discardableResult
    public func emergency(tempC: Double?) -> Double {
        guard let t = tempC, t.isFinite, t >= emergencyTempC else { return 0 }
        emergencyPauses += 1
        let ms = maxPauseMs
        Thread.sleep(forTimeInterval: ms / 1000)
        totalPauseMs += ms
        return ms
    }

    /// Фактическая доля работы по отношению GPU-времени к GPU+паузам.
    public var measuredDuty: Double {
        let total = totalGpuMs + totalPauseMs
        return total > 0 ? totalGpuMs / total : 0
    }
}

/// Доля работы-потолок по пользовательской «мощности» в процентах.
///
/// 100% → 0.99, 85% → 0.8415, 50% → 0.495. Ниже 10% не опускаемся:
/// обучение должно двигаться, пусть и медленно.
public func dutyCeiling(powerPercent: Int) -> Double {
    let p = Double(min(max(powerPercent, 0), 100)) / 100.0
    return min(max(maxSafeDuty * p, 0.10), maxSafeDuty)
}

/// Температура GPU, выше которой уходить нельзя, °C.
///
/// 93 °C — точка, на которой машина уходит в перезагрузку (watchdog по
/// WindowServer). 85 °C — жёсткий потолок с запасом 8 °C.
/// Это НЕ цель, а граница: контур работает ниже и подходит к ней только
/// если иначе не хватает производительности.
public let defaultMaxTempC = 85.0

/// Рабочая точка, к которой стремимся, °C.
///
/// 82 при потолке 85 — это и есть «максимум без остановок»: три градуса
/// запаса хватает, чтобы регулятор успевал среагировать за один замер (2 с)
/// и ни разу не довёл дело до принудительного остывания.
public let defaultTargetTempC = 82.0

/// Абсолютный предохранитель: выше этой температуры обучение уходит на
/// принудительное остывание, независимо от доли работы.
public let emergencyTempC = 86.0

/// Аварийный ответ на перегрев: НЕ полный стоп, а резкое снижение нагрузки +
/// короткий ограниченный cooldown (пользователь: «вместо стопа полного на переборе
/// температуры делай снижение нагрузки»). Обучение продолжается на пониженном duty,
/// карта охлаждается на короткой паузе, а не висит 20 секунд. Чистая функция,
/// тестируется без GPU.
public func emergencyThermalResponse(tempC: Double, currentDuty: Double,
                                     threshold: Double = emergencyTempC,
                                     maxCooldownSeconds: Double = 2)
    -> (duty: Double, cooldownSeconds: Double) {
    guard tempC.isFinite, tempC >= threshold else { return (currentDuty, 0) }
    let over = tempC - threshold
    // Резко режем долю работы (не в нуль — продолжаем), с полом 0.05.
    let newDuty = max(currentDuty * 0.35, 0.05)
    // Короткий cooldown: растёт с перегревом, но жёстко ограничен сверху (не 20 с).
    let cooldown = min(max(maxCooldownSeconds, 0), 0.5 + over * 0.4)
    return (newDuty, cooldown)
}

/// Пауза на остывание между эпохами, мс.
///
/// Пауза — это тепловой инструмент, а не часть расписания. Если карта
/// холоднее рабочей точки, остывать не от чего, и любая пауза здесь —
/// чистый простой: обучение просто стоит.
///
/// Замер `yume gpuheat 180 0` (180 с непрерывной нагрузки на 99%): пик 78 °C,
/// дальше температура сама сползла к 68–70 °C, мощность 5–23 Вт. То есть
/// карта не успевает нагреться даже за непрерывную работу — значит
/// фиксированная секунда на каждой границе эпохи ничего не охлаждает.
///
/// - Returns: 0, если температура ниже рабочей точки; половину базовой паузы
///   между рабочей точкой и потолком; полную — у самого потолка.
///   Нет данных с датчика → 0: пошаговый регулятор всё равно держит нагрузку
///   на стартовой доле работы, а эта пауза случается раз в эпоху.
public func epochCoolPauseMs(tempC: Double?, pauseMs: Double,
                             targetTempC: Double = defaultTargetTempC,
                             maxTempC: Double = defaultMaxTempC) -> Double {
    guard let t = tempC, t.isFinite, t > 0 else { return 0 }
    let base = max(pauseMs.isFinite ? pauseMs : 0, 1000)
    if t >= maxTempC { return base }
    if t >= targetTempC { return base / 2 }
    return 0
}

/// Регулятор доли работы по фактической температуре GPU.
///
/// Почему не «фиксированные 85%»: доля работы — это не мощность. При одной
/// и той же доле работы температура зависит от того, что именно считает GPU,
/// от температуры в комнате и от состояния системы охлаждения. Фиксированное
/// число либо перегревает машину, либо недогружает её. Поэтому контур
/// замкнут по температуре: он сам находит ту долю работы, при которой карта
/// держится ниже порога троттлинга, и плавно к ней подходит.
///
/// Порог 70/75 °C откалиброван замером на этой машине (`yume gpuheat 40 85`):
/// драйвер отдаёт температуру края кристалла, а троттлинг начинается по
/// температуре точки — она на 20–25 °C выше. При 74 °C на краю время шага
/// уже ползло вверх (138 → 194 мс), то есть запас кончился. Отсюда цель 70 °C.
public struct PowerController {
    /// Текущая разрешённая доля работы (0…1).
    public private(set) var duty: Double
    /// Максимальная наблюдённая частота ядра — база для детектора троттлинга.
    public private(set) var peakClockMHz: Double = 0
    /// Доля работы, выше которой не поднимаемся (из «мощности» в процентах).
    public let ceiling: Double
    /// Температура, к которой стремимся.
    public let targetTempC: Double
    /// Температура, при которой режем нагрузку жёстко.
    public let maxTempC: Double

    /// Предел изменения доли работы за один замер — это и есть «плавность»:
    /// нагрузка меняется ступеньками не больше 5% за 2 секунды.
    public let slew: Double

    public init(ceiling: Double, targetTempC: Double = defaultTargetTempC, maxTempC: Double = defaultMaxTempC,
                startDuty: Double = 0.80, slew: Double = 0.05) {
        self.ceiling = min(max(ceiling, 0.05), maxSafeDuty)
        self.targetTempC = targetTempC
        self.maxTempC = maxTempC
        self.slew = max(slew, 0.001)
        // Старт близко к потолку, а не с половины: задача контура — держать
        // максимум, который держится, и опускаться только когда стало жарко.
        // С 0.50 до 0.99 при шаге +3% за замер уходило бы полминуты, и
        // первые шаги шли бы заметно медленнее, чем машина позволяет.
        self.duty = min(max(startDuty, 0.05), self.ceiling)
    }

    /// Один шаг регулирования. Возвращает новую долю работы.
    ///
    /// Если телеметрии нет — ничего не меняем: лучше держать текущий режим,
    /// чем гадать.
    ///
    /// Детектора троттлинга по частоте здесь сознательно НЕТ. Замер показал,
    /// что `Core Clock(MHz)` и `GPU Activity(%)` в PerformanceStatistics —
    /// мгновенные срезы: на реальной нагрузке они скачут от 0 до 468 МГц,
    /// потому что попадают в промежутки между нашими command buffer'ами.
    /// Строить на них решение — реагировать на шум. Настоящий троттлинг
    /// виден по времени шага (в замере: 138 → 194 мс при 74 °C), и именно
    /// его отслеживает `heatLevel` по дрейфу `gpuMs`.
    @discardableResult
    public mutating func update(stats: GpuStats?) -> Double {
        guard let s = stats, s.hasTemperature else { return duty }

        // Пик частоты держим только для журнала.
        if s.coreClockMHz > peakClockMHz { peakClockMHz = s.coreClockMHz }

        var want = duty
        if s.temperatureC >= maxTempC {
            want = duty * 0.60            // перегрев: резко вниз
        } else if s.temperatureC >= targetTempC + 2 {
            want = duty * 0.85            // выше цели: снижаемся
        } else if s.temperatureC >= targetTempC {
            want = duty * 0.95            // на цели: слегка придерживаем
        } else if s.temperatureC <= targetTempC - 6 {
            want = duty + 0.030           // явно холодно: разгоняемся
        } else if s.temperatureC <= targetTempC - 3 {
            want = duty + 0.015           // прохладно: добавляем понемногу
        }

        // Ограничение скорости изменения — чтобы нагрузка была плавной,
        // а не «дёргалась» между перегревом и простоем.
        want = min(max(want, duty - slew), duty + slew)

        duty = min(max(want, 0.05), ceiling)
        return duty
    }

    /// Аварийное снижение нагрузки на перегреве (термозащита). Применяет
    /// `emergencyThermalResponse` к ВНУТРЕННЕМУ duty (не локальной переменной),
    /// чтобы снижение сохранилось между замерами; возвращает короткий cooldown.
    /// Контур сам поднимет duty обратно, когда карта остынет — обучение не стоит.
    @discardableResult
    mutating func emergencyThrottle(tempC: Double) -> Double {
        let r = emergencyThermalResponse(tempC: tempC, currentDuty: duty)
        duty = r.duty
        return r.cooldownSeconds
    }
}

