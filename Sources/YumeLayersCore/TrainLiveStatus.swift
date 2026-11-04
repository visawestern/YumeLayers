import Foundation

// Живой статус обучения для интерфейса.
//
// Задача: показать не только проценты, но и реальный прогресс — эпоху,
// текущий кадр набора и шаг внутри эпохи. Отдельно считается «здоровье»
// цикла: если от тренировки давно не приходило шагов, интерфейс должен
// честно сказать «нет шагов N с», а не крутить вечный лоадер.
//
// Все функции чистые и покрыты selftest (см. ConvTrainer.runSelfTest).

/// Стадия цикла обучения.
public enum TrainStage: Equatable {
    /// Подготовка: модель и кадры грузятся на GPU, шагов ещё нет.
    case preparing
    /// Само обучение: идут шаги.
    case training
}

/// Один отчёт тренировочного цикла в UI.
public struct TrainStepReport: Equatable {
    /// Глобальный номер шага (1-based), для полосы прогресса.
    public let done: Int
    /// Всего шагов за весь прогон (эпохи × шаги в эпохе).
    public let total: Int
    /// Текущая эпоха (1-based).
    public let epoch: Int
    /// Всего эпох.
    public let epochs: Int
    /// Текущий шаг внутри эпохи (1-based).
    public let step: Int
    /// Шагов в одной эпохе.
    public let stepsPerEpoch: Int
    /// Первый кадр текущей пачки (1-based, 0 — кадров нет).
    public let frame: Int
    /// Всего кадров в наборе.
    public let framesTotal: Int
    /// Размер пачки (сколько кадров обрабатывается за шаг).
    public let batch: Int
    /// Среднее время шага, секунды (nil — ещё неизвестно).
    public let secondsPerStep: Double?
    /// Пометка вместо обычной строки (например, прогресс загрузки VRAM).
    public let note: String?
    /// Стадия цикла.
    public let stage: TrainStage

    public init(done: Int, total: Int,
                epoch: Int, epochs: Int,
                step: Int, stepsPerEpoch: Int,
                frame: Int, framesTotal: Int,
                batch: Int, secondsPerStep: Double?,
                note: String? = nil,
                stage: TrainStage = .training) {
        self.done = done
        self.total = total
        self.epoch = epoch
        self.epochs = epochs
        self.step = step
        self.stepsPerEpoch = stepsPerEpoch
        self.frame = frame
        self.framesTotal = framesTotal
        self.batch = batch
        self.secondsPerStep = secondsPerStep
        self.note = note
        self.stage = stage
    }
}

/// Стоит ли писать в журнал строку «обучение: шаг …».
///
/// `done == 0` — это НЕ шаг, а служебный отчёт: подготовка модели, догрузка
/// окна VRAM. Такие отчёты приходят по одному на каждые 16 декодированных
/// кадров (1024 кадра окна → 64 отчёта), и прежнее условие `done % 200 == 0`
/// для нуля всегда истинно — отсюда 64 одинаковые строки «шаг 0» на каждую
/// догрузку. Настоящие шаги 1-based (`t += 1` до формирования отчёта),
/// поэтому `done > 0` — точный признак шага.
///
/// - Parameters:
///   - done: глобальный номер шага (1-based); 0 — служебный отчёт.
///   - total: всего шагов за прогон.
///   - every: писать каждый N-й шаг.
public func shouldLogTrainStep(done: Int, total: Int, every: Int = 200) -> Bool {
    guard done > 0 else { return false }
    let n = max(every, 1)
    return done % n == 0 || done == total
}

/// Состояние живости тренировочного цикла по времени с последнего шага.
public enum TrainLivePhase: Equatable {    /// Шаги идут штатно.
    case running
    /// Шаг давно не приходил, но это ещё в пределах нормы для медленных шагов.
    case waiting(seconds: Int)
    /// Шагов нет дольше допустимого — вероятно, зависло.
    case stalled(seconds: Int)
}

/// Разряды через пробел: 43187 → «43 187».
/// Своя реализация вместо NumberFormatter: результат не зависит от локали,
/// а тесты в selftest должны быть детерминированными.
public func formatThousands(_ value: Int) -> String {
    let negative = value < 0
    let digits = String(abs(value))
    var result = ""
    for (i, ch) in digits.enumerated() {
        if i > 0 && (digits.count - i) % 3 == 0 { result.append(" ") }
        result.append(ch)
    }
    return negative ? "-" + result : result
}

/// Строка под спиннером: «Эпоха 1/3 · кадр 12 480 из 43 187 · шаг 12 480/43 187 · 0.46 с/шаг».
/// Если в отчёте есть пометка (фаза подготовки, загрузка VRAM) — показывается
/// она: обычные «эпоха/кадр/шаг» на этом этапе ещё не значимы.
public func trainLiveText(_ r: TrainStepReport) -> String {
    if let note = r.note, !note.isEmpty { return note }
    var parts: [String] = []

    let epochs = max(1, r.epochs)
    let epoch = min(max(1, r.epoch), epochs)
    parts.append("Эпоха \(epoch)/\(epochs)")

    if r.framesTotal > 0 {
        let frame = min(max(1, r.frame), r.framesTotal)
        parts.append("кадр \(formatThousands(frame)) из \(formatThousands(r.framesTotal))")
        if r.batch > 1 { parts.append("батч \(r.batch)") }
    }

    let stepsPerEpoch = max(1, r.stepsPerEpoch)
    let step = min(max(1, r.step), stepsPerEpoch)
    parts.append("шаг \(formatThousands(step))/\(formatThousands(stepsPerEpoch))")

    if let s = r.secondsPerStep, s.isFinite, s >= 0 {
        parts.append(String(format: "%.2f с/шаг", s))
    }
    return parts.joined(separator: " · ")
}

/// Порог «нормы» между шагами: не меньше 45 с (UI получает отчёт каждые
/// 5 шагов) и не меньше 40 шагов при медленном обучении. Даже на медленной
/// машине это не даёт ложного «зависло» на штатных паузах.
public func trainLivePhase(secondsSinceStep: TimeInterval,
                           stepSeconds: Double?) -> TrainLivePhase {
    let since = max(0, secondsSinceStep)
    let perStep = max(0, stepSeconds ?? 0)
    let base = max(45.0, perStep * 40.0)
    if since <= base { return .running }
    let seconds = Int(since.rounded())
    if since <= base * 2 { return .waiting(seconds: seconds) }
    return .stalled(seconds: seconds)
}

/// Короткая длительность для строки статуса: 50 с, 2 мин 5 с, 1 ч 20 мин.
public func formatShortDuration(_ seconds: Int) -> String {
    let s = max(0, seconds)
    if s < 60 { return "\(s) с" }
    let m = s / 60
    if m < 60 { return "\(m) мин \(s % 60) с" }
    return "\(m / 60) ч \(m % 60) мин"
}

/// Текст состояния: идёт / давно нет шага / похоже на зависание.
public func trainLivePhaseText(_ phase: TrainLivePhase) -> String {
    switch phase {
    case .running:
        return "Обучение идёт"
    case .waiting(let seconds):
        return "Последний шаг \(formatShortDuration(seconds)) назад"
    case .stalled(let seconds):
        return "Нет шагов \(formatShortDuration(seconds)) — возможно, зависло"
    }
}

/// Текст состояния с учётом стадии: на подготовке (загрузка модели и кадров
/// в VRAM) шагов ещё нет, поэтому «Обучение идёт» вводит в заблуждение.
public func trainLiveStateText(phase: TrainLivePhase, stage: TrainStage) -> String {
    guard stage == .preparing else { return trainLivePhaseText(phase) }
    switch phase {
    case .running:
        return "Подготовка данных"
    case .waiting(let seconds):
        return "Подготовка данных · последнее обновление \(formatShortDuration(seconds)) назад"
    case .stalled(let seconds):
        return "Подготовка данных не идёт \(formatShortDuration(seconds)) — возможно, зависло"
    }
}

/// Пометка фазы подготовки: модель и кадры ещё не на GPU.
public func trainPreparationNote() -> String {
    "Подготовка: загрузка модели на GPU…"
}

/// Пометка прогресса загрузки кадров в VRAM.
public func trainVRAMLoadNote(loaded: Int, total: Int) -> String {
    "Загрузка кадров в VRAM: \(max(0, loaded))/\(max(0, total))"
}

/// Номер кадра набора, который обрабатывается прямо сейчас (1-based).
///
/// `epochIndex`/`stepIndex` — с нуля (как в тренировочном цикле), `batch` —
/// сколько кадров берётся за шаг. Набор циклический: после последнего кадра
/// обучение снова идёт с первого. Набора нет — 0 (кадр не показываем).
public func currentFrameIndex(epochIndex: Int, stepIndex: Int, stepsPerEpoch: Int,
                              batch: Int, datasetFrames: Int) -> Int {
    guard datasetFrames > 0 else { return 0 }
    let globalStep = max(0, epochIndex) * max(1, stepsPerEpoch) + max(0, stepIndex)
    let consumed = globalStep * max(1, batch)
    return consumed % datasetFrames + 1
}
