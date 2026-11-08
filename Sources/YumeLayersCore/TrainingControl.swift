// YumeLayers — потококобезопасное управление обучением (остановка из GUI).
//
// Выделено из `ConvTrainer.swift` для соблюдения лимита 600 строк/файл.
import Foundation

/// Потокобезопасное управление обучением (для остановки из GUI).
public final class TrainingControl {
    public static let shared = TrainingControl()
    private let lock = NSLock()
    private var _cancelled = false
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
        let stuck = _running && runToken == token
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
    }
}
