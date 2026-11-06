import Foundation

// YumeLayers Core — чистая логика доставки кадров (без Metal/UI).
// Выделена для TDD: поведение эмиттера проверяется в `selftest`.

/// Планировщик публикации кадров: строгий порядок + пропуск лидирующих
/// чёрных. Не зависит от потоков — состояние меняется атомарными вызовами.
public struct FrameDeliveryPlanner {
    /// Следующий чанк, который можно отдавать.
    public private(set) var nextChunk = 0
    /// Известен ли пропуск лидирующих чёрных.
    public private(set) var skipKnown = false
    /// Сколько лидирующих полностью чёрных кадров пропущено.
    public private(set) var skip = 0
    /// Сколько кадров уже отдано наружу (для глобальной нумерации).
    public private(set) var written = 0

    public init() {}

    /// Фиксирует пропуск лидирующих чёрных (один раз, после готовности первого чанка).
    public mutating func learnSkipIfNeeded(leadingBlacks: Int) {
        if !skipKnown {
            skip = max(0, leadingBlacks)
            skipKnown = true
        }
    }

    /// Чанки, которые можно отдать прямо сейчас: строго подряд, начиная
    /// с текущего, только готовые. Пока первый чанк не готов — пусто.
    public func emittableRanges(chunkDone: [Bool]) -> [Range<Int>] {
        guard skipKnown else { return [] }
        var c = nextChunk
        while c < chunkDone.count && chunkDone[c] { c += 1 }
        if c > nextChunk {
            return [nextChunk..<c]
        }
        return []
    }

    /// Отметить чанки как отданные.
    public mutating func markEmittedChunks(_ count: Int) {
        nextChunk += count
    }

    /// Учесть отданные кадры (для нумерации).
    public mutating func recordWritten(frameCount: Int) {
        written += frameCount
    }
}

/// Правило обновления коллажа: первые 3 кадра — сразу, далее каждые 200.
public func shouldRefreshCollage(atFrame n: Int) -> Bool {
    n <= 3 || n % 200 == 0
}

/// Разбиение диапазона на батчи фиксированного размера (для параллельной
/// обработки кадров без ручного копирования границ).
public func batchRanges(total: Int, batch: Int) -> [Range<Int>] {
    guard total > 0, batch > 0 else { return [] }
    var out: [Range<Int>] = []
    var lo = 0
    while lo < total {
        let hi = min(total, lo + batch)
        out.append(lo..<hi)
        lo = hi
    }
    return out
}

/// Отбор контрольного кадра для визуальной оценки качества: 1 кадр на
/// секунду видео. При fps N берём каждый N-й кадр (индекс 0 — всегда).
/// Для 24 fps — кадры 0, 24, 48, …; для 30 fps — 0, 30, 60, …
public func isControlFrame(index: Int, fps: Double) -> Bool {
    let step = max(1, Int(fps.rounded()))
    return index % step == 0
}

/// Освобождение JPEG-байтов чанка из RAM после того, как кадры загружены
/// в VRAM (float32 уже резидентен — исходные JPEG больше не нужны).
/// Возвращает число освобождённых байт. Защита от повторного освобождения:
/// если JPEG уже пуст (count == 0), ничего не делаем.
public func releaseFrameJPEG(_ frames: inout [MemoryFrame], range: Range<Int>) -> Int {
    var freed = 0
    for k in range where k < frames.count {
        if frames[k].jpeg.count > 0 {
            freed += frames[k].jpeg.count
            frames[k].jpeg = Data()
        }
    }
    return freed
}

/// Есть ли непустой существующий файл (для бэкапа чекпоинтов).
public func isNonEmptyFile(_ path: String) -> Bool {
    let fm = FileManager.default
    guard fm.fileExists(atPath: path) else { return false }
    guard let attrs = try? fm.attributesOfItem(atPath: path),
          let size = attrs[.size] as? Int else { return false }
    return size > 0
}

/// Создаёт бэкап файла (safetensors) рядом с оригиналом с суффиксом времени.
/// Возвращает путь бэкапа (или nil, если файла нет/он пуст/ошибка копирования).
/// Готовая к тестам чистая функция — не зависит от GUI.
/// Гарантирует уникальность имени: если копия с таким суффиксом уже есть
/// (например, повторный вызов в ту же секунду), добавляет числовой индекс.
public func createCheckpointBackup(fromPath path: String) -> String? {
    guard isNonEmptyFile(path) else { return nil }
    let stamp = DateFormatter.localizedString(from: Date(), dateStyle: .medium, timeStyle: .medium)
        .replacingOccurrences(of: "/", with: "-")
        .replacingOccurrences(of: ":", with: "-")
        .replacingOccurrences(of: " ", with: "_")
    let base = path + ".backup-\(stamp)"
    let fm = FileManager.default
    var candidate = base
    var index = 1
    while fm.fileExists(atPath: candidate) {
        index += 1
        candidate = "\(base)-\(index)"
    }
    do {
        try fm.copyItem(atPath: path, toPath: candidate)
        return candidate
    } catch {
        return nil
    }
}

/// Создаёт бэкапы всех указанных путей, которые существуют и непусты.
public func createBackups(forPaths paths: [String]) -> [String] {
    paths.compactMap { createCheckpointBackup(fromPath: $0) }
}

/// Итог подготовки чекпоинта к старту обучения.
public enum CheckpointPrepareOutcome: Equatable {
    /// Файла не было — обучение начнётся с новых весов.
    case fresh
    /// Старый чекпоинт удалён; рядом создан бэкап по пути `backup`.
    case removed(backup: String)
    /// resume=true — файл сохранён и будет использован для дообучения.
    case keep
}

/// Готовит чекпоинт к старту обучения:
/// - `resume=true` — файл не трогаем (`.keep`);
/// - `resume=false` и существует непустой файл — делаем бэкап и удаляем
///   оригинал, чтобы обучение гарантированно началось с новых весов
///   (`.removed`);
/// - иначе — `.fresh`.
/// Если бэкап не удался, оригинал НЕ удаляется (возвращаем `.fresh`).
/// Чистая функция — покрыта тестами.
public func prepareCheckpointForTraining(convPath: String,
                                         resume: Bool) -> CheckpointPrepareOutcome {
    if resume { return .keep }
    guard isNonEmptyFile(convPath) else { return .fresh }
    guard let backup = createCheckpointBackup(fromPath: convPath) else { return .fresh }
    try? FileManager.default.removeItem(atPath: convPath)
    return .removed(backup: backup)
}

// ---------- Редактируемое значение: видимость элементов в режиме правки ----------

/// Состояние видимости элементов редактируемого поля: в обычном режиме показываем
/// лейбл-значение и карандаш; в режиме правки — поле ввода с кнопками
/// ✓ (применить) и ✕ (отменить) ВНУТРИ поля.
public struct EditFieldVisibility {
    public let showValue: Bool
    public let showEditButton: Bool
    public let showInputWithActions: Bool
    public init(showValue: Bool, showEditButton: Bool, showInputWithActions: Bool) {
        self.showValue = showValue
        self.showEditButton = showEditButton
        self.showInputWithActions = showInputWithActions
    }
}

/// Чистая логика: какой элемент показывать в зависимости от режима редактирования.
/// — не редактируем: лейбл-значение + карандаш (поле скрыто);
/// — редактируем: поле ввода с ✓/✕ внутри (лейбл и карандаш скрыты).
public func editFieldVisibility(isEditing: Bool) -> EditFieldVisibility {
    isEditing
        ? EditFieldVisibility(showValue: false, showEditButton: false, showInputWithActions: true)
        : EditFieldVisibility(showValue: true, showEditButton: true, showInputWithActions: false)
}

// ---------- Пул кадров в VRAM: размер порции дозагрузки ----------

/// Вычисляет размер порции дозагрузки кадров в VRAM-пул.
///
/// Зачем порции: при заливке одного большого чанка (напр. 1024 кадра) требуется
/// ~2 × N × frameBytes единовременной памяти — N × frameBytes на staging-буферы
/// (storageModeShared) + N × frameBytes на временные float-массивы при
/// декодировании JPEG. На машине с 32 ГБ RAM и свободными ~2 ГБ такой пик
/// (≈10 ГБ) приводит к свопу/зависанию.
///
/// Правило: бюджет = min(25% физической RAM, 50% рабочего набора GPU);
/// порция = бюджет / (2 × frameBytes), ограничена [64, 1024].
/// Возвращает число кадров на порцию.
public func frameLoadChunkSize(frameBytes: Int, availableRAM: UInt64,
                               maxWorkingSet: UInt64) -> Int {
    guard frameBytes > 0 else { return 64 }
    let ramBudget = availableRAM / 4                     // 25% RAM
    let gpuBudget = maxWorkingSet / 2                    // 50% рабочего набора
    let budget = min(ramBudget, gpuBudget)
    let perChunk = UInt64(frameBytes) * 2
    guard perChunk > 0 else { return 64 }
    let desired = Int(budget / perChunk)
    return min(1024, max(64, desired))
}

/// Реальный прогресс выполнения в обработанных кадрах (а не в шагах).
/// Один шаг обрабатывает `batch` кадров, поэтому:
///   done_frames  = stepDone × batch
///   total_frames = totalSteps × batch
/// Это даёт честный процент от реального объёма работы.
public func progressInFrames(stepDone: Int, totalSteps: Int, batch: Int)
    -> (done: Int, total: Int) {
    let b = max(1, batch)
    return (stepDone * b, totalSteps * b)
}

/// Целевая страница пагинации при переходе в край: начало (0) или конец
/// (pages-1). Защита от пустых/вырожденных диапазонов.
public func edgePage(current: Int, pages: Int, toStart: Bool) -> Int {
    guard pages > 1 else { return 0 }
    let target = toStart ? 0 : pages - 1
    return min(max(0, target), pages - 1)
}

/// Следует ли записать прогресс в журнал на текущем шаге.
/// Правило: логируем старт (done == 0), завершение (done >= total) и каждые
/// 5% прогресса. Это даёт объективную картину движения, не засоряя журнал
/// на каждый кадр. Возвращает true, когда нужно писать.
public func shouldLogProgress(done: Int, total: Int, lastLoggedPercent: Int) -> Bool {
    guard total > 0 else { return done == 0 }
    if done <= 0 { return true }                      // старт
    if done >= total { return true }                  // финиш
    let percent = Int((Double(done) / Double(total)) * 100)
    // Логируем при пересечении очередной границы 5%.
    let threshold = (lastLoggedPercent / 5 + 1) * 5
    return percent >= threshold
}

/// Результат автоподбора параметров обучения.
public struct AutoTuneResult {
    public let steps: Int
    public let epochs: Int
    /// Новое значение скорости обучения, если авто-подбор его меняет (иначе nil).
    public let lr: Float?
    public let changed: Bool
    public init(steps: Int, epochs: Int, lr: Float? = nil, changed: Bool) {
        self.steps = steps
        self.epochs = epochs
        self.lr = lr
        self.changed = changed
    }
}

/// Отмечает кадры, для которых построена реконструкция (галочка ✓ в коллаже).
///
/// `names[i]` — имя i-го кадра в карусели, `reconNames` — имена готовых
/// реконструкций. Возвращает число впервые отмеченных кадров.
/// Длину `processed` приводит к длине `names`, не теряя уже поставленные метки.
@discardableResult
public func markProcessedFrames(names: [String], reconNames: Set<String>,
                                processed: inout [Bool]) -> Int {
    if processed.count < names.count {
        processed.append(contentsOf: repeatElement(false, count: names.count - processed.count))
    } else if processed.count > names.count {
        processed.removeLast(processed.count - names.count)
    }
    var added = 0
    for i in 0..<names.count where reconNames.contains(names[i]) {
        if !processed[i] {
            processed[i] = true
            added += 1
        }
    }
    return added
}

/// Автоподбор эпох/шагов под число кадров С УЧЁТОМ галочки «Resume».
/// Правило:
/// — если `resume` включён (мы продолжаем обучение с чекпоинта) — НЕ
///   пересчитываем шаги/эпохи: оставляем текущие (changed=false). Продолжение
///   должно идти с тех параметров, что уже заданы. НО скорость обучения
///   авто-снижается (×0.5), если она не была задана вручную — т.к. модель уже
///   частично сошлась, и большой LR рискует «перепрыгнуть» минимум.
/// — иначе: шаги = frames/batch (если не редактированы вручную), эпохи =
///   min(10, max(3, 3000/frames)) (если не редактированы); LR не трогаем.
public func autoTuneValues(frames: Int, batch: Int, resume: Bool,
                           currentSteps: Int, currentEpochs: Int,
                           currentLr: Float, stepsEditedByUser: Bool,
                           epochsEditedByUser: Bool, lrEditedByUser: Bool) -> AutoTuneResult {
    guard frames > 0 else {
        return AutoTuneResult(steps: currentSteps, epochs: currentEpochs, changed: false)
    }
    if resume {
        // Продолжение: шаги/эпохи не трогаем, но LR снижаем автоматически,
        // если пользователь не выставил его вручную.
        let lr = lrEditedByUser ? nil : currentLr * 0.5
        return AutoTuneResult(steps: currentSteps, epochs: currentEpochs,
                              lr: lr, changed: false)
    }
    let b = max(1, batch)
    var newSteps = currentSteps
    var newEpochs = currentEpochs
    var changed = false
    if !stepsEditedByUser {
        newSteps = max(1, frames / b)
        changed = true
    }
    if !epochsEditedByUser {
        newEpochs = min(10, max(3, 3000 / frames))
        changed = true
    }
    return AutoTuneResult(steps: newSteps, epochs: newEpochs, changed: changed)
}

/// Отпечаток обучающих данных (детерминированный, не зависит от запуска):
/// число кадров + байты равномерной выборки кадров (до 64). Используется,
/// чтобы на resume понять, тот ли это файл, на котором учились. Детерминизм
/// обеспечивает FNV-1a 64 бита (не зависит от хэш-стандарта Swift).
public func dataFingerprint(frames: [MemoryFrame]) -> String {
    guard !frames.isEmpty else { return "empty" }
    var h: UInt64 = 14695981039346656037
    func mix(_ b: UInt8) { h = (h ^ UInt64(b)) &* 1099511628211 }
    func mixBytes(_ d: Data) {
        for b in d { mix(b) }
    }
    // Число кадров участвует в отпечатке.
    for b in String(frames.count).utf8 { mix(b) }
    // Равномерная выборка (до 64 кадров) байтов JPEG.
    let step = max(1, frames.count / 64)
    var i = 0
    while i < frames.count {
        let f = frames[i]
        if !f.jpeg.isEmpty { mixBytes(f.jpeg) }
        i += step
    }
    return String(format: "%016llx", h)
}
