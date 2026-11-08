// YumeLayers — пул переиспользуемых буферов.
//
// Выделено из `GpuTraining.swift` для соблюдения лимита 600 строк/файл.
// Избегаем повторных аллокаций MTLBuffer: буферы возвращаются в пул по ключу
// размера и выдаются заново.
import Foundation
import Metal

public final class BufferPool {
    let engine: MetalEngine
    var free: [String: [MTLBuffer]] = [:]
    init(_ engine: MetalEngine) { self.engine = engine }
    func get(_ bytes: Int) -> MTLBuffer {
        let key = String(bytes)
        if var arr = free[key], !arr.isEmpty {
            let buffer = arr.removeLast()
            free[key] = arr
            return buffer
        }
        return engine.device.makeBuffer(length: bytes, options: [])!
    }
    func put(_ b: MTLBuffer) {
        let key = String(b.length)
        if free[key] == nil { free[key] = [] }
        free[key]!.append(b)
    }
}
