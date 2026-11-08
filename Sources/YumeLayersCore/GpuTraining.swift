import Foundation
import Metal

public final class GpuSession {
    let engine: MetalEngine
    let pool: BufferPool
    var cmd: MTLCommandBuffer!
    var enc: MTLComputeCommandEncoder!
    /// Стейджинг-буферы, отданные в текущий command buffer (возврат в пул после wait).
    var staged: [MTLBuffer] = []

    init(_ engine: MetalEngine) {
        self.engine = engine
        self.pool = BufferPool(engine)
    }

    func begin() {
        cmd = engine.queue.makeCommandBuffer()!
        enc = cmd.makeComputeCommandEncoder()!
    }

    /// Потолок ожидания одного command buffer, секунды.
    ///
    /// `waitUntilCompleted()` при зависании GPU не возвращается НИКОГДА.
    /// 4 ноября 2026 это стоило системе перезагрузки: пока наш процесс ждал,
    /// WindowServer не мог получить доступ к той же GPU и не «отмечался»
    /// 120 секунд — сработал watchdogd и выключил машину.
    /// Ждать 30 с вместо бесконечности мало что меняет по существу, но даёт
    /// шанс завершиться самим, а не утащить за собой оконный сервер.
    static var gpuDeadlineSeconds: Double = 30

    /// Результат асинхронного коммита: позволяет отправить command buffer в
    /// очередь и продолжить работу, не дожидаясь GPU. `wait()` блокирует до
    /// завершения, возвращает время GPU (мс) и возвращает стейджинг в пул.
    ///
    /// ВАЖНО: возврат стейджинга в пул намеренно происходит в `wait()` на
    /// вызывающем потоке, а НЕ в completed-handler'е — `BufferPool` не
    /// потокобезопасен, а handler летит на потоке Metal.
    public final class CommittedStep {
        let buffer: MTLCommandBuffer
        let pool: BufferPool
        let staged: [MTLBuffer]
        private let sem = DispatchSemaphore(value: 0)

        init(_ buffer: MTLCommandBuffer, pool: BufferPool, staged: [MTLBuffer]) {
            self.buffer = buffer
            self.pool = pool
            self.staged = staged
        }

        /// Обработчик завершения только сигналит: время GPU считается в
        /// `wait()`, где известно стенное время ожидания (показания драйвера
        /// без стенной границы небезопасны — см. `boundedGpuMs`).
        func signal() {
            sem.signal()
        }

        /// Блокирует до завершения GPU (с жёстким пределом `gpuDeadlineSeconds`,
        /// чтобы висячий buffer не утащил за собой WindowServer) и возвращает
        /// время GPU в мс.
        ///
        /// Страховка от потери обработчика: если Metal не доставил
        /// `addCompletedHandler`, но буфер уже выполнен (`cmd.status ==
        /// .completed`), ждать семафор бессмысленно — это и был источник
        /// «GPU не ответила за 30 с» при полностью свободной карте.
        ///
        /// Возвращаемое время GPU ограничено стенным временем ожидания:
        /// пустые показания драйвера (gpuStartTime = 0) больше не превращаются
        /// в двухсекундную паузу на каждом шаге.
        func wait() -> Double {
            let start = Date()
            while true {
                if sem.wait(timeout: .now() + 0.005) == .success { break }
                let elapsed = Date().timeIntervalSince(start)
                let state = MetalEngine.bufferState(buffer.status)
                switch gpuWaitOutcome(state: state, handlerSignaled: false,
                                      elapsedSeconds: elapsed,
                                      deadlineSeconds: GpuSession.gpuDeadlineSeconds) {
                case .done:
                    yumeLog("⚠️ обработчик завершения не доставлен — статус буфера: "
                          + "\(state). Продолжаю по статусу.")
                    break
                case .timedOut:
                    yumeLog("❌ GPU не ответила за \(Int(GpuSession.gpuDeadlineSeconds)) с — "
                          + "command buffer завис. Статус: \(state), ошибка: "
                          + "\(buffer.error?.localizedDescription ?? "нет"). "
                          + "Аварийный выход, чтобы не перегрузить WindowServer.")
                    exit(70)
                case .keepWaiting:
                    continue
                }
                break
            }
            for s in staged { pool.put(s) }
            let wallMs = Date().timeIntervalSince(start) * 1000
            let reported = (buffer.gpuEndTime - buffer.gpuStartTime) * 1000
            let bounded = boundedGpuMs(reportedMs: reported, wallMs: wallMs)
            if reported > wallMs + 50 {
                yumeLog(String(format: "⚠️ драйвер сообщил %.0f мс GPU при %.0f мс стенного времени — показание отброшено",
                               reported, wallMs))
            }
            return bounded
        }
    }

    /// Асинхронная отправка command buffer в очередь GPU БЕЗ блокировки.
    ///
    /// Это основа совмещения CPU и GPU (double-buffering): следующий шаг
    /// кодируется, пока текущий ещё считается на видеокарте. В синхронном
    /// `end()` CPU простаивал всё время работы GPU; на длинных прогонах
    /// (30k+ шагов) накладные расходы Metal (commit+ожидание на каждый шаг)
    /// сопоставимы со временем счёта — порядка 180 с «пустого» времени на 30k
    /// шагов. Перекрытие убирает этот простой из критического пути.
    func commit() -> CommittedStep {
        enc.endEncoding()
        let committed = cmd!
        let myStaged = staged
        staged.removeAll()
        let step = CommittedStep(committed, pool: pool, staged: myStaged)
        committed.addCompletedHandler { [weak step] _ in
            step?.signal()   // время GPU считается в wait() — там есть стенная граница
        }
        committed.commit()
        return step
    }

    /// Синхронное завершение: отправить и дождаться. Сохранён для обратной
    /// совместимости (тесты, eval, короткие прогоны). Семантически равно
    /// `commit().wait()`.
    @discardableResult
    func end() -> Double {
        let ms = commit().wait()
        // GPU закончил команды: можно вернуть освобождённые буферы в пул.
        engine.reclaimRetiredBuffers()
        return ms
    }

    /// Копирует данные с CPU в тензор ВНУТРИ текущего command buffer:
    /// завершаем активный compute-энкодер → blit → новый compute-энкодер.
    /// Один command buffer, без отдельного ожидания и без аллокаций.
    /// Стейджинг берётся из пула и возвращается в него после завершения шага.
}
