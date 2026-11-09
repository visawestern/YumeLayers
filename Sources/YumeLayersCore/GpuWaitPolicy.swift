// Политика ожидания GPU и оценка «занятости» карты.
//
// Зачем отдельный файл с чистыми функциями: драйвер дискретной карты
// (AMD Radeon Pro 5500M на macOS) в двух местах ведёт себя нечестно, и оба
// раза это ломало обучение.
//
// 1) ПОКАЗАНИЯ ВРЕМЕНИ. `MTLCommandBuffer.gpuStartTime/gpuEndTime` иногда
//    приходят пустыми или с незаполненным началом. Тогда «время GPU»
//    оказывается больше времени всего шага (наблюдалось 18 000 мс при
//    12 мс реальной работы). Регулятор нагрузки умеет превращать это в паузу
//    (s = gpuMs·(1/duty − 1)), и пауза упиралась в потолок 2 с на КАЖДЫЙ шаг:
//    обучение замедлялось примерно в 100 раз при полностью простаивающей
//    карте (0–6 % загрузки, 8 Вт).
//
// 2) ПОТЕРЯ ОБРАБОТЧИКА. Ожидание построено на `addCompletedHandler`:
//    обработчик сигналит семафор. Если Metal не доставил обработчик (а буфер
//    при этом уже выполнен), ожидание висит до предела и процесс умирает с
//    «GPU не ответила за 30 с» — хотя карта была свободна всё это время.
//
// Обе проблемы лечатся здесь, чистыми функциями: показание времени
// ограничивается стенным временем шага (физический потолок), а ожидание
// спрашивает у буфера его статус, если обработчик не сработал.
//
// Файл намеренно без Metal: политика проверяется обычными юнит-тестами.

import Foundation

/// Состояние command buffer с точки зрения ожидания.
public enum GpuBufferState: Equatable, Sendable {
    /// Ещё выполняется.
    case running
    /// Выполнен (успешно или с ошибкой — различает драйвер).
    case completed
    /// Выполнен с ошибкой драйвера.
    case failed
}

/// Что делать при очередной проверке command buffer.
public enum GpuWaitOutcome: Equatable, Sendable {
    /// Буфер закончен — можно читать результат.
    case done
    /// Ещё работает, время есть — ждём дальше.
    case keepWaiting
    /// Время вышло — аварийный выход.
    case timedOut
}

/// Решение по одному тику ожидания command buffer.
///
/// `handlerSignaled` — сработал ли обработчик завершения.
/// `state` — состояние, о котором сообщил сам буфер.
/// Если обработчик потерялся, но буфер уже выполнен — работа закончена:
/// ждать сигнала, которого не будет, бессмысленно.
public func gpuWaitOutcome(state: GpuBufferState,
                           handlerSignaled: Bool,
                           elapsedSeconds: Double,
                           deadlineSeconds: Double) -> GpuWaitOutcome {
    if handlerSignaled || state == .completed || state == .failed { return .done }
    if elapsedSeconds >= deadlineSeconds { return .timedOut }
    return .keepWaiting
}

/// Правдоподобное «время работы GPU» по показаниям драйвера.
///
/// GPU не может быть занят дольше, чем длился сам шаг по стенным часам, —
/// это физический потолок. Показание, которое его превышает, невозможно и
/// отбрасывается: иначе один сбой драйвера превращает обучение в
/// последовательность двухсекундных пауз.
///
/// - Показание в пределах стенного времени (+ допуск на ожидание в очереди)
///   берётся как есть.
/// - Показание больше стенного — заменяется стенным.
/// - Показания нет (0/NaN) — берётся стенное время.
/// - Не знаем ничего — 0 (не NaN: пауза должна быть конечной).
public func boundedGpuMs(reportedMs: Double,
                         wallMs: Double,
                         toleranceMs: Double = 50) -> Double {
    let reported = reportedMs.isFinite ? reportedMs : 0
    let wall = wallMs.isFinite ? wallMs : 0
    if wall <= 0 {
        // Стенного времени нет — ограничивать нечем, но и мусор не пускаем.
        return reported > 0 ? reported : 0
    }
    if reported <= 0 { return wall }
    if reported > wall + max(toleranceMs, 0) { return wall }
    return reported
}

/// Реальная ли это проблема или штатный обход потерянного обработчика.
///
/// На AMD Radeon Pro 5500M драйвер регулярно НЕ доставляет
/// `addCompletedHandler`, хотя буфер давно выполнен (`status == .completed`).
/// Это штатный путь: мы выходим по статусу и продолжаем — предупреждать об
/// этом на КАЖДОМ буфере нельзя, иначе лог тонет в тысячах ложных «⚠️» и
/// настоящая ошибка становится невидимой. Подлинная проблема — только когда
/// буфер завершился СО СТАТУСОМ ОШИБКИ (`.failed`): тогда данные нечитаемы.
public func gpuHandlerMissIsRealProblem(state: GpuBufferState) -> Bool {
    state == .failed
}

/// Печатать ли это конкретное «обработчик не доставлен» событие.
///
/// Штатные обходы (`.completed`) логируются дозированно: первые `logFirst`
/// штук (видно, что драйвер вообще теряет обработчики) и далее каждое
/// `logEvery`-е (видно нарастание). Реальная ошибка (`.failed`) печатается
/// всегда. `count` — порядковый номер штатного обхода в этом процессе
/// (начиная с 1). Возвращает false для «молча» пропущенных рядовых обходов.
public func gpuShouldLogHandlerMiss(state: GpuBufferState,
                                    count: Int,
                                    logFirst: Int = 3,
                                    logEvery: Int = 500) -> Bool {
    if gpuHandlerMissIsRealProblem(state: state) { return true }
    guard count >= 1 else { return false }
    if count <= logFirst { return true }
    let every = max(logEvery, 1)
    return count % every == 0
}

/// Модель цикла «эпохи × шаги» с проверкой отмены на КАЖДОМ шаге (а не только
/// на границе эпох). Чистая функция: `isCancelled(epoch, step)` говорит, отменён
/// ли прогон на данном шаге. Возвращает, сколько эпох/шагов реально выполнено и
/// остановлен ли по пользователю. Используется и как эталон поведения, и как
/// проверяемая специфика того же кода в `runEpochs`.
public func runEpochLoop(epochs: Int, stepsPerEpoch: Int,
                         isCancelled: (_ epoch: Int, _ step: Int) -> Bool)
    -> (epochsDone: Int, stepsDone: Int, stoppedByUser: Bool) {
    var epochsDone = 0
    var totalSteps = 0
    for ep in 0..<max(epochs, 0) {
        if isCancelled(ep, 0) { return (epochsDone, totalSteps, true) }
        for step in 0..<max(stepsPerEpoch, 0) {
            if isCancelled(ep, step) { return (epochsDone, totalSteps, true) }
            totalSteps += 1
        }
        epochsDone += 1
    }
    return (epochsDone, totalSteps, false)
}

/// Пауза регулятора, нарезанная на короткие срезы.
///
/// Регулятор может добавить до 2 с паузы после шага. Если спать одним куском,
/// «Стоп» ждёт конца паузы — пользователь видит «ничего не происходит».
/// Короткие срезы позволяют проверить флаг отмены между ними.
public func pauseSlices(totalMs: Double, sliceMs: Double = 100) -> [Double] {
    guard totalMs.isFinite, totalMs > 0, sliceMs.isFinite, sliceMs > 0 else { return [] }
    var slices: [Double] = []
    var left = totalMs
    while left > 0 {
        let s = min(sliceMs, left)
        slices.append(s)
        left -= s
    }
    return slices
}

/// Решение watchdog'а по зависшему command buffer: ЖДАТЬ или УБИТЬ процесс.
///
/// Осознанно НЕТ ветки «перезапустить шаг». Причина: перезапуск зависшего
/// command buffer невозможен — очередь Metal уже забита этим буфером, а
/// попытка отправить новый и ждать его утаскивает за собой WindowServer,
/// который обращается к той же карте (4 ноября 2026 так случилось
/// перезагружение машины). Единственный безопасный выход из ПО-настоящему
/// застрявшего буфера — завершить процесс (exit 70), ОСВОБОДИВ очередь.
///
/// Ждём дальше, только если у времени ЕЩЁ есть запас И буфер не завершён
/// успешно. Буфер completed → ждём ничего (данные читай, watchdog не нужен).
public enum WatchdogDecision: Equatable, Sendable {
    case keepWaiting
    case exitProcess
}

public func watchdogDecision(bufferCompleted: Bool,
                             elapsedSeconds: Double,
                             deadlineSeconds: Double) -> WatchdogDecision {
    // Успешно завершённый буфер — это НЕ зависание; продолжаем по статусу.
    if bufferCompleted { return .keepWaiting }
    if elapsedSeconds >= deadlineSeconds { return .exitProcess }
    return .keepWaiting
}
