// YumeLayers — выгрузка результата реконструкций.
//
// Реконструкции после обучения живут в памяти интерфейса («имя кадра →
// PNG-данные»). Здесь две чистые функции, которые превращают это в полезный
// результат на диске и в корректную разметку сетки кадров:
//
//   • `reconstructedFrameIndices` — какие позиции сетки показывать;
//   • `saveReconstructions`       — запись PNG-файлов в папку.
//
// Файл без Metal и без UI: и функции, и их поведение проверяются обычными
// юнит-тестами.

import Foundation

/// Индексы кадров набора, для которых есть реконструкция.
///
/// Порядок — как в `frameNames` (порядок кадров набора), а не как в
/// словаре реконструкций: словарь в Swift не упорядочен, и по нему
/// нельзя построить сетку превью.
///
/// - Параметры:
///   - frameNames: имена кадров набора по порядку;
///   - reconNames: имена кадров, для которых есть реконструкция.
/// - Возвращает: индексы в `frameNames` с реконструкцией, по возрастанию.
public func reconstructedFrameIndices(frameNames: [String], reconNames: [String]) -> [Int] {
    guard !frameNames.isEmpty, !reconNames.isEmpty else { return [] }
    let available = Set(reconNames)
    var out: [Int] = []
    out.reserveCapacity(min(frameNames.count, reconNames.count))
    for (index, name) in frameNames.enumerated() where available.contains(name) {
        out.append(index)
    }
    return out
}

/// Сохраняет реконструкции в папку (по одному PNG-файлу на кадр).
///
/// - Параметры:
///   - recons: «имя кадра → PNG-данные»;
///   - dir: папка назначения (создаётся, если её нет).
/// - Возвращает: число записанных файлов.
/// - Ошибка: если файл записать не удалось (нет прав, диск полон).
public func saveReconstructions(_ recons: [String: Data], to dir: String) throws -> Int {
    let fm = FileManager.default
    try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
    let target = URL(fileURLWithPath: dir, isDirectory: true)
    var written = 0
    // Сортировка по имени — файлы на диске должны идти в предсказуемом
    // порядке (иначе каталог «прыгает» при каждом сохранении).
    for (name, data) in recons.sorted(by: { $0.key < $1.key }) {
        let safe = (name as NSString).lastPathComponent
        guard !safe.isEmpty, safe != ".", safe != ".." else { continue }
        try data.write(to: target.appendingPathComponent(safe))
        written += 1
    }
    return written
}
