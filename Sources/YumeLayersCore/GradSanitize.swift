// YumeLayers — санация градиентов (NaN/Inf → 0 + клиппинг нормы).
//
// Выделено из `ConvTrainer.swift` для соблюдения лимита 600 строк/файл.
//
// ВАЖНО: раньше здесь на КАЖДЫЙ тензор вызывались `toArray()` и `setData()`,
// а каждая из них создавала новый shared MTLBuffer и отдельный command
// buffer с ожиданием. На 12 градиентах это ~24–48 аллокаций и столько же
// синхронизаций GPU за шаг; за 15k шагов драйвер AMD накапливал сотни тысяч
// ресурсов и вставал внутри `newBufferWithBytes` (стек: sanitizeGrads →
// setData → newBufferWithBytes → IOAccelResourceCreate → mach_msg).
//
// Теперь весь пакет пересылается ОДНИМ staging-буфером из пула: один blit
// на чтение, CPU-санация в памяти и один blit на запись. Аллокаций после
// прогрева нет, синхронизаций — две вместо десятков.
import Foundation

/// Замена NaN/Inf в градиентах на 0 + клиппинг суммарной нормы (L2).
func sanitizeGrads(_ grads: [GPUTensor], maxNorm: Float = 1.0,
                   preScale: Float = 1.0) {
    guard let first = grads.first, !grads.isEmpty else { return }
    let engine = first.engine
    let total = grads.reduce(0) { $0 + $1.n }
    guard total > 0 else { return }
    let totalBytes = total * 4

    // Смещения каждого тензора внутри общего staging-буфера.
    var offsets = [Int]()
    offsets.reserveCapacity(grads.count)
    var off = 0
    for g in grads {
        offsets.append(off)
        off += g.n * 4
    }

    let staging = engine.takeStaging(totalBytes)

    // 1) Все градиенты → staging одним blit-ом.
    guard let readCmd = engine.queue.makeCommandBuffer(),
          let readBlit = readCmd.makeBlitCommandEncoder() else {
        engine.putStaging(staging)
        fatalError(allocationFailureMessage(
            bytes: totalBytes,
            availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                - engine.device.currentAllocatedSize,
            what: "command buffer чтения градиентов"))
    }
    for (i, g) in grads.enumerated() {
        readBlit.copy(from: g.buf, sourceOffset: 0, to: staging,
                      destinationOffset: offsets[i], size: g.n * 4)
    }
    readBlit.endEncoding()
    engine.waitForCompletion(readCmd, what: "чтение градиентов для санации")

    // 2) Санация и клиппинг — на CPU, в одной непрерывной памяти.
    let p = staging.contents().bindMemory(to: Float.self, capacity: total)
    var totalSq: Float = 0
    var fixed = 0
    for (i, g) in grads.enumerated() {
        let base = offsets[i] / 4
        var s: Float = 0
        for k in 0..<g.n {
            var v = p[base + k]
            if preScale != 1 { v *= preScale }
            if v.isNaN || v.isInfinite {
                v = 0
                fixed += 1
            } else {
                s += v * v
            }
            p[base + k] = v
        }
        totalSq += s
    }
    let norm = sqrt(totalSq)
    if norm > maxNorm {
        let scale = maxNorm / norm
        for k in 0..<total { p[k] *= scale }
    }

    // 3) staging → все градиенты одним blit-ом.
    guard let writeCmd = engine.queue.makeCommandBuffer(),
          let writeBlit = writeCmd.makeBlitCommandEncoder() else {
        engine.putStaging(staging)
        fatalError(allocationFailureMessage(
            bytes: totalBytes,
            availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                - engine.device.currentAllocatedSize,
            what: "command buffer записи градиентов"))
    }
    for (i, g) in grads.enumerated() {
        writeBlit.copy(from: staging, sourceOffset: offsets[i], to: g.buf,
                       destinationOffset: 0, size: g.n * 4)
    }
    writeBlit.endEncoding()
    engine.waitForCompletion(writeCmd, what: "запись градиентов после санации")
    engine.putStaging(staging)

    if fixed > 0 {
        yumeLog("санация градиентов: \(fixed) значений NaN/Inf → 0; норма \(norm)")
    }
}
