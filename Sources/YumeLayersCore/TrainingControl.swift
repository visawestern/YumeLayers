// YumeLayers — потококобезопасное управление обучением (остановка из GUI).
//
// Выделено из `ConvTrainer.swift` для соблюдения лимита 600 строк/файл.
import Foundation

/// Потокобезопасное управление обучением (для остановки из GUI).
public final class TrainingControl {
    public static let shared = TrainingControl()
    private let lock = NSLock()
    private var _cancelled = false
    private var _paused = false
    /// Идёт ли прогон прямо сейчас (нужно watchdog'у остановки).
    private var _running = false
    /// Поколение прогона: старый watchdog не должен убивать новый прогон.
    private var runToken = 0

    public var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _cancelled
    }

    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return _running
    }

    /// Пауза: поток обучения встаёт на границе шага, но прогон НЕ отменяется и
    /// НЕ сбрасывается — веса, Adam-моменты и счётчик шагов остаются. Отличие от
    /// «Стоп» в том, что `resume()` продолжает с того же места.
    public var isPaused: Bool {
        lock.lock(); defer { lock.unlock() }
        return _paused
    }

    public func pause() { lock.lock(); _paused = true; lock.unlock() }
    public func resume() { lock.lock(); _paused = false; lock.unlock() }

    /// Блокирует поток обучения, пока стоит пауза, и отпускает при снятии паузы
    /// ИЛИ при отмене (пауза + «Стоп» → выход, а не вечное ожидание). Спит
    /// короткими срезами, чтобы «Стоп» отрабатывал почти мгновенно, и НЕ трогает
    /// watchdog (пауза — намеренное состояние, а не зависание).
    public func waitWhilePaused(sliceMs: Double = 100) {
        while true {
            lock.lock()
            let p = _paused, c = _cancelled
            lock.unlock()
            if !p || c { return }
            Thread.sleep(forTimeInterval: max(sliceMs, 1) / 1000)
        }
    }

    /// Начало прогона. Флаг отмены НЕ сбрасывается: сброс делает `reset()`
    /// перед стартом, иначе «Стоп», нажатый между стартом и входом в цикл,
    /// был бы молча потерян.
    func beginRun() {
        lock.lock()
        _running = true
        runToken += 1
        lock.unlock()
    }

    func endRun() {
        lock.lock()
        _running = false
        runToken += 1
        lock.unlock()
    }

    /// Остановка. Если поток обучения заблокирован в драйвере GPU, он не
    /// увидит флаг никогда (проверка — только между шагами). Поэтому через
    /// `graceSeconds` с запускается watchdog: не ответил — завершаем процесс,
    /// иначе приложение остаётся навсегда «обучающимся» и не реагирует.
    public func cancel(graceSeconds: Double = 20) {
        lock.lock()
        _cancelled = true
        let running = _running
        let token = runToken
        lock.unlock()
        guard running else { return }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + graceSeconds) {
            TrainingControl.shared.enforceStop(token: token, graceSeconds: graceSeconds)
        }
    }

    private func enforceStop(token: Int, graceSeconds: Double) {
        lock.lock()
        let stuck = watchdogShouldFire(running: _running,
                                       currentToken: runToken, scheduledToken: token)
        lock.unlock()
        guard stuck else { return }
        yumeLog("❌ Обучение не ответило на остановку за \(Int(graceSeconds)) с: "
              + "поток заблокирован в драйвере GPU (аллокация/ожидание command buffer). "
              + "Принудительное завершение процесса — иначе приложение не остановить.")
        exit(70)
    }

    public func reset() {
        lock.lock(); defer { lock.unlock() }
        _cancelled = false
        _paused = false
    }
}

/// Действие управляющего контура на границе шага обучения. Чистая функция —
/// самая опасная ветка цикла (решает: считать шаг / стоять / выйти). Отмена
/// приоритетнее паузы: «пауза + Стоп» обязаны выйти, иначе поток завис бы в
/// паузе навсегда и не отреагировал на остановку.
public enum TrainingGateAction: Equatable, Sendable {
    case proceed    // идём дальше
    case wait       // пауза: встать и ждать снятия/отмены
    case stop       // отмена: прервать прогон
}

public func trainingGateAction(cancelled: Bool, paused: Bool) -> TrainingGateAction {
    if cancelled { return .stop }
    if paused { return .wait }
    return .proceed
}

/// Правило срабатывания watchdog-остановки — вынесено как чистая функция,
/// потому что это самая рискованная ветка: она решает, убивать ли процесс.
/// Watchdog обязан fire только когда прогон В ТОМ ЖЕ поколении (`scheduledToken`)
/// и всё ещё идёт (`running`). Если началась новая попытка обучения (`runToken`
/// вырос в `beginRun`/`endRun`) — старый watchdog обязан промолчать, иначе он
/// убьёт живой новый прогон. Если прогон уже завершился — убивать нечего.
public func watchdogShouldFire(running: Bool,
                               currentToken: Int,
                               scheduledToken: Int) -> Bool {
    running && currentToken == scheduledToken
}
