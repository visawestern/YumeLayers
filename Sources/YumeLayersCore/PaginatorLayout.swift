import Foundation

// Геометрия пагинатора кадров.
//
// Задача: число видимых кружков-страниц должно считаться от ширины центральной
// колонки, а не от ширины самого пагинатора. Иначе получается замкнутый круг:
// кружочки распирают строку → строка «видит» большую ширину → показывает ещё
// больше кружочков → колонка растёт и уезжает за экран.
//
// Все функции чистые и покрыты selftest (см. ConvTrainer.runSelfTest).

/// Зазор между элементами пагинатора, pt.
public let paginatorGap: Double = 4
/// Ширина кнопки-стрелки (в начало/назад/вперёд/в конец), pt.
public let paginatorArrowWidth: Double = 28
/// Сколько стрелок стоит в строке пагинатора.
public let paginatorArrowCount = 4
/// Горизонтальный отступ центральной колонки, pt (по 24 с каждой стороны).
/// Строка пагинатора уже колонки ровно на два таких отступа.
public let paginatorSideInset: Double = 24

/// Фактическая ширина строки пагинатора: ширина центральной колонки минус
/// её горизонтальные отступы. Минимум — 120 pt, чтобы строка не вырождалась
/// в ноль на некорректных значениях.
public func paginatorRowWidth(columnWidth: Double) -> Double {
    guard columnWidth.isFinite else { return 120 }
    return max(120, columnWidth - 2 * paginatorSideInset)
}

/// Ширина кружка под номер страницы: чем больше цифр, тем шире овал,
/// чтобы номер не обрезался. 1–2 цифры → 28, 3 → 34, 4+ → 40.
public func paginatorDotWidth(pageNumber: Int) -> Double {
    let digits = String(max(1, pageNumber)).count
    switch digits {
    case ...2: return 28
    case 3: return 34
    default: return 40
    }
}

/// Сколько кружков влезает в колонку шириной `columnWidth`.
/// Учитываются четыре стрелки по краям и зазоры между элементами.
/// Минимум — один кружок (пагинатор всегда управляем).
public func dotCapacity(columnWidth: Double, dotWidth: Double) -> Int {
    guard columnWidth.isFinite, dotWidth.isFinite else { return 1 }
    let available = columnWidth
        - Double(paginatorArrowCount) * paginatorArrowWidth
        - Double(paginatorArrowCount + 1) * paginatorGap
    let step = max(16, dotWidth) + paginatorGap
    let count = Int(floor((available + paginatorGap) / step))
    return max(1, count)
}

/// Смещение кадра внутри страницы коллажа (0…9) по номеру слота.
/// Порядок в L-раскладке: нижний ряд (слоты 5…9) — смещения 0…4 слева→справа,
/// правый столбец (слоты 0…4) — смещения 9…5 снизу→вверх.
/// Чистая функция: вынесена из GUI, чтобы поведение коллажа покрывалось тестом.
public func frameOffsetForSlot(_ slot: Int) -> Int {
    if slot >= 5 { return slot - 5 }
    return 9 - slot
}

/// Окно видимых страниц (0-based) вокруг текущей: текущая держится примерно
/// в центре, у краёв окно прижимается к началу/концу. Пустой набор страниц
/// даёт пустой результат — без обращения к несуществующим индексам.
public func paginatorVisiblePages(current: Int, pages: Int, limit: Int) -> [Int] {
    guard pages > 0 else { return [] }
    let lim = max(1, limit)
    if pages <= lim { return Array(0..<pages) }
    let cur = min(max(0, current), pages - 1)
    var start = cur - lim / 2
    if start < 0 { start = 0 }
    var end = start + lim
    if end > pages {
        end = pages
        start = end - lim
    }
    return Array(start..<end)
}
