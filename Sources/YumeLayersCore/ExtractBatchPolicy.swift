import Foundation

/// Политика асинхронной выдачи извлечённых кадров в интерфейс.
///
/// Извлечение 44 000 кадров не должно дёргать main thread на каждый кадр:
/// кадры копятся в буфере и отдаются пачками. Первая пачка — страница
/// коллажа (10 кадров), чтобы картинки появились сразу; дальше — крупные
/// пачки, а коллаж перерисовывается только пока заполняется первая
/// страница. Остальные кадры доступны через пагинацию.
public struct ExtractBatchPolicy: Sendable {
    /// Размер первой пачки — ровно первая страница коллажа.
    public let firstFlush: Int
    /// Размер последующих пачек.
    public let flushSize: Int
    /// Максимальное время ожидания неполной пачки, мс.
    public let flushIntervalMs: Double

    public init(firstFlush: Int = 10, flushSize: Int = 64,
                flushIntervalMs: Double = 200) {
        self.firstFlush = firstFlush
        self.flushSize = flushSize
        self.flushIntervalMs = flushIntervalMs
    }

    /// Пора ли отдавать накопленные кадры в интерфейс.
    ///
    /// - Parameters:
    ///   - pending: сколько кадров уже накоплено в буфере.
    ///   - sinceLastFlushMs: мс с прошлого сброса.
    ///   - flushedFirstBatch: был ли уже хотя бы один сброс.
    public func shouldFlush(pending: Int, sinceLastFlushMs: Double,
                            flushedFirstBatch: Bool) -> Bool {
        guard pending > 0 else { return false }
        let threshold = flushedFirstBatch ? flushSize : firstFlush
        return pending >= threshold || sinceLastFlushMs >= flushIntervalMs
    }

    /// Нужно ли перерисовывать коллаж после добавления пачки.
    ///
    /// Коллаж показывает только текущую страницу. Во время извлечения
    /// перерисовываем его лишь пока заполняется первая страница: дальше
    /// картинки грузятся по клику на пагинации, а не на каждый кадр.
    /// - Parameters:
    ///   - page: текущая страница коллажа.
    ///   - totalBefore: сколько кадров было до пачки.
    ///   - totalAfter: сколько стало после пачки.
    ///   - pageSize: кадров на странице.
    public func shouldRenderCollage(page: Int, totalBefore: Int, totalAfter: Int,
                                    pageSize: Int) -> Bool {
        guard totalAfter > totalBefore else { return false }
        guard page == 0 else { return false }
        // Пока первая страница не заполнена — показываем кадры по мере
        // поступления; после этого состав первой страницы не меняется.
        return totalBefore < pageSize
    }
}
