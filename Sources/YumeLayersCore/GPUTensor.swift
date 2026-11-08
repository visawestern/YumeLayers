// YumeLayers — GPU-тензор поверх Metal.
//
// Выделено из `GpuTraining.swift` для соблюдения лимита 600 строк/файл.
// Безвременная владеет приватным MTLBuffer и выполняет blit-операции
// (инициализация, чтение/запись, копирование).
import Foundation
import Metal

public final class GPUTensor {
    let engine: MetalEngine
    let buf: MTLBuffer
    let rows: Int
    let cols: Int
    let n: Int  // rows * cols

    init(engine: MetalEngine, rows: Int, cols: Int) {
        self.engine = engine
        self.rows = rows; self.cols = cols; self.n = rows * cols
        let bytes = max(rows * cols, 1) * 4
        self.buf = engine.makePrivateBuffer(length: bytes, what: "тензор \(rows)×\(cols)")
    }

    deinit {
        // Возврат приватного буфера в пул переиспользования. Сразу в свободный
        // пул не кладём: GPU может ещё читать данные. Фактическое переиспользование
        // произойдёт после `reclaimRetiredBuffers()` (вызывается из `end()`).
        engine.retireBuffer(buf)
    }

    init(engine: MetalEngine, data: [Float], rows: Int) {
        precondition(rows > 0 && data.count % rows == 0,
                     "GPUTensor: rows=\(rows) должен делить \(data.count) элементов")
        self.engine = engine
        self.rows = rows; self.cols = data.count / rows; self.n = data.count
        let bytes = max(data.count, 1) * 4
        guard let cmd = engine.queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else {
            fatalError(allocationFailureMessage(
                bytes: bytes,
                availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                    - engine.device.currentAllocatedSize,
                what: "command buffer тензора \(rows)×\(data.count / max(rows, 1)) (с данными)"))
        }
        let staging = engine.takeStaging(bytes)
        staging.contents().copyMemory(from: data, byteCount: data.count * 4)
        let priv = engine.makePrivateBuffer(length: bytes, what: "тензор \(rows)×\(data.count / max(rows, 1)) (с данными)")
        blit.copy(from: staging, sourceOffset: 0, to: priv, destinationOffset: 0, size: data.count * 4)
        blit.endEncoding()
        engine.waitForCompletion(cmd, what: "инициализация тензора \(rows)×\(data.count / max(rows, 1))")
        engine.putStaging(staging)
        self.buf = priv
    }

    func toArray() -> [Float] {
        let bytes = max(n, 1) * 4
        let staging = engine.takeStaging(bytes)
        defer { engine.putStaging(staging) }
        guard let cmd = engine.queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else {
            fatalError(allocationFailureMessage(
                bytes: bytes,
                availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                    - engine.device.currentAllocatedSize,
                what: "command buffer чтения тензора \(rows)×\(cols)"))
        }
        blit.copy(from: buf, sourceOffset: 0, to: staging, destinationOffset: 0, size: n * 4)
        blit.endEncoding()
        engine.waitForCompletion(cmd, what: "чтение тензора \(rows)×\(cols)")
        return Array(UnsafeBufferPointer(start: staging.contents().bindMemory(to: Float.self, capacity: max(n, 1)),
                                         count: n))
    }

    /// Запись данных с CPU (staging → private).
    func setData(_ data: [Float]) {
        precondition(data.count == n)
        let staging = engine.takeStaging(n * 4)
        defer { engine.putStaging(staging) }
        guard let cmd = engine.queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else {
            fatalError(allocationFailureMessage(
                bytes: n * 4,
                availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                    - engine.device.currentAllocatedSize,
                what: "command buffer записи тензора \(rows)×\(cols)"))
        }
        staging.contents().copyMemory(from: data, byteCount: n * 4)
        blit.copy(from: staging, sourceOffset: 0, to: buf, destinationOffset: 0, size: n * 4)
        blit.endEncoding()
        engine.waitForCompletion(cmd, what: "запись тензора \(rows)×\(cols)")
    }

    func copyFrom(_ src: GPUTensor) {
        precondition(src.n == n)
        guard let cmd = engine.queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else {
            fatalError(allocationFailureMessage(
                bytes: n * 4,
                availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                    - engine.device.currentAllocatedSize,
                what: "command buffer копирования тензора \(rows)×\(cols)"))
        }
        blit.copy(from: src.buf, sourceOffset: 0, to: buf, destinationOffset: 0, size: n * 4)
        blit.endEncoding()
        engine.waitForCompletion(cmd, what: "копирование тензора \(rows)×\(cols)")
    }

    /// Точечная запись (для численной проверки градиентов).
    func setValue(_ index: Int, _ value: Float) {
        precondition(index >= 0 && index < n, "setValue: индекс \(index) вне тензора из \(n)")
        let staging = engine.takeStaging(4)
        defer { engine.putStaging(staging) }
        staging.contents().bindMemory(to: Float.self, capacity: 1)[0] = value
        guard let cmd = engine.queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else {
            fatalError(allocationFailureMessage(
                bytes: 4,
                availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                    - engine.device.currentAllocatedSize,
                what: "command buffer точечной записи"))
        }
        blit.copy(from: staging, sourceOffset: 0, to: buf, destinationOffset: index * 4, size: 4)
        blit.endEncoding()
        engine.waitForCompletion(cmd, what: "точечная запись тензора \(rows)×\(cols)")
    }

    func getValue(_ index: Int) -> Float {
        precondition(index >= 0 && index < n, "getValue: индекс \(index) вне тензора из \(n)")
        let staging = engine.takeStaging(4)
        defer { engine.putStaging(staging) }
        guard let cmd = engine.queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else {
            fatalError(allocationFailureMessage(
                bytes: 4,
                availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                    - engine.device.currentAllocatedSize,
                what: "command buffer точечного чтения"))
        }
        blit.copy(from: buf, sourceOffset: index * 4, to: staging, destinationOffset: 0, size: 4)
        blit.endEncoding()
        engine.waitForCompletion(cmd, what: "точечное чтение тензора \(rows)×\(cols)")
        return staging.contents().bindMemory(to: Float.self, capacity: 1)[0]
    }
}
