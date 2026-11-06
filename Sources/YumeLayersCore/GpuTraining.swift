// YumeLayers — обуча тензоры и конвейер (один command buffer на шаг).
import Foundation
import Metal

/// Тензор на GPU (приватная память видеокарты, без PCIe-копий на каждом шаге).
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
        guard let buf = engine.device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            // Раньше здесь стоял `!`: при OOM пользователь получал падение на
            // разыменовании nil без единого слова о том, чего не хватило.
            fatalError(allocationFailureMessage(
                bytes: bytes,
                availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                    - engine.device.currentAllocatedSize,
                what: "тензор \(rows)×\(cols)"))
        }
        self.buf = buf
    }

    init(engine: MetalEngine, data: [Float], rows: Int) {
        self.engine = engine
        self.rows = rows; self.cols = data.count / rows; self.n = data.count
        let bytes = data.count * 4
        let staging = engine.device.makeBuffer(bytes: data, length: bytes,
                                               options: .storageModeShared)!
        let priv = engine.device.makeBuffer(length: bytes, options: .storageModePrivate)!
        let cmd = engine.queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: staging, sourceOffset: 0, to: priv, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        self.buf = priv
    }

    func toArray() -> [Float] {
        let bytes = n * 4
        let staging = engine.device.makeBuffer(length: bytes, options: .storageModeShared)!
        let cmd = engine.queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: buf, sourceOffset: 0, to: staging, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return Array(UnsafeBufferPointer(start: staging.contents().bindMemory(to: Float.self, capacity: n),
                                         count: n))
    }

    /// Запись данных с CPU (staging → private).
    func setData(_ data: [Float]) {
        precondition(data.count == n)
        let staging = engine.device.makeBuffer(bytes: data, length: n * 4,
                                               options: .storageModeShared)!
        let cmd = engine.queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: staging, sourceOffset: 0, to: buf, destinationOffset: 0, size: n * 4)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    func copyFrom(_ src: GPUTensor) {
        precondition(src.n == n)
        let cmd = engine.queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: src.buf, sourceOffset: 0, to: buf, destinationOffset: 0, size: n * 4)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    /// Точечная запись (для численной проверки градиентов).
    func setValue(_ index: Int, _ value: Float) {
        var v = value
        let staging = engine.device.makeBuffer(bytes: &v, length: 4, options: .storageModeShared)!
        let cmd = engine.queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: staging, sourceOffset: 0, to: buf, destinationOffset: index * 4, size: 4)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    func getValue(_ index: Int) -> Float {
        let staging = engine.device.makeBuffer(length: 4, options: .storageModeShared)!
        let cmd = engine.queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: buf, sourceOffset: index * 4, to: staging, destinationOffset: 0, size: 4)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return staging.contents().bindMemory(to: Float.self, capacity: 1)[0]
    }
}

/// Пул переиспользуемых буферов (избегаем аллокаций).
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

/// Сессия GPU: накапливает операции, выполняет одним коммитом.
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
        private var gpuMs: Double = 0

        init(_ buffer: MTLCommandBuffer, pool: BufferPool, staged: [MTLBuffer]) {
            self.buffer = buffer
            self.pool = pool
            self.staged = staged
        }

        func signal(gpuMs: Double) {
            self.gpuMs = gpuMs.isFinite ? gpuMs : 0
            sem.signal()
        }

        /// Блокирует до завершения GPU (с жёстким пределом `gpuDeadlineSeconds`,
        /// чтобы висячий buffer не утащил за собой WindowServer) и возвращает
        /// время GPU в мс.
        func wait() -> Double {
            if sem.wait(timeout: .now() + GpuSession.gpuDeadlineSeconds) == .timedOut {
                yumeLog("❌ GPU не ответила за \(Int(GpuSession.gpuDeadlineSeconds)) с — "
                      + "command buffer завис. Аварийный выход, чтобы не перегрузить WindowServer.")
                exit(70)
            }
            for s in staged { pool.put(s) }
            return gpuMs
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
            let ms = (committed.gpuEndTime - committed.gpuStartTime) * 1000
            step?.signal(gpuMs: ms)
        }
        committed.commit()
        return step
    }

    /// Синхронное завершение: отправить и дождаться. Сохранён для обратной
    /// совместимости (тесты, eval, короткие прогоны). Семантически равно
    /// `commit().wait()`.
    @discardableResult
    func end() -> Double {
        return commit().wait()
    }

    /// Копирует данные с CPU в тензор ВНУТРИ текущего command buffer:
    /// завершаем активный compute-энкодер → blit → новый compute-энкодер.
    /// Один command buffer, без отдельного ожидания и без аллокаций.
    /// Стейджинг берётся из пула и возвращается в него после завершения шага.
    func upload(_ t: GPUTensor, from floats: [Float]) {
        precondition(floats.count == t.n)
        // Metal: в command buffer может быть только один активный энкодер.
        // Завершаем текущий compute-энкодер (он пустой) перед созданием blit.
        enc.endEncoding()
        let staging = pool.get(t.n * 4)
        staging.contents().copyMemory(from: floats, byteCount: t.n * 4)
        staged.append(staging)
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: staging, sourceOffset: 0, to: t.buf, destinationOffset: 0, size: t.n * 4)
        blit.endEncoding()
        enc = cmd.makeComputeCommandEncoder()!
    }

    /// MSE между двумя тензорами на GPU (в текущем command buffer).
    /// Возвращает shared-буфер с одним float (сумма квадратов), который можно
    /// прочитать после `end()` — без чтения полных тензоров на CPU.
    func encodeMSE(recon: GPUTensor, target: GPUTensor) -> MTLBuffer {
        precondition(recon.n == target.n)
        let out = engine.device.makeBuffer(length: 4, options: .storageModeShared)!
        var nv = UInt32(recon.n)
        enc.setComputePipelineState(engine.msePS)
        enc.setBuffer(recon.buf, offset: 0, index: 0)
        enc.setBuffer(target.buf, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&nv, length: 4, index: 3)
        let tg = MTLSize(width: 1024, height: 1, depth: 1)
        let groups = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        return out
    }

    private func setBuffers(_ buf: [MTLBuffer?]) {
        for (i, b) in buf.enumerated() {
            if let b = b { enc.setBuffer(b, offset: 0, index: i) }
        }
    }
    private func setUInt(_ v: UInt32, index: Int) {
        var x = v
        enc.setBytes(&x, length: 4, index: index)
    }
    private func dispatch(_ threadsW: Int, _ threadsH: Int, _ groupsW: Int, _ groupsH: Int) {
        let tg = MTLSize(width: threadsW, height: threadsH, depth: 1)
        let threads = MTLSize(width: groupsW, height: groupsH, depth: 1)
        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
    }

    private func dispatchGroups(_ threadsW: Int, _ threadsH: Int, _ groupsW: Int, _ groupsH: Int) {
        let tg = MTLSize(width: threadsW, height: threadsH, depth: 1)
        let groups = MTLSize(width: groupsW, height: groupsH, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }

    private func dispatch3D(_ tw: Int, _ th: Int, _ td: Int,
                            _ gw: Int, _ gh: Int, _ gd: Int) {
        let tg = MTLSize(width: tw, height: th, depth: td)
        let threads = MTLSize(width: gw, height: gh, depth: gd)
        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
    }

    // C = A (M×K) × B (K×N); transA/transB позволяют использовать Aᵀ, Bᵀ
    func matmul(_ A: GPUTensor, _ B: GPUTensor, transA: Bool = false, transB: Bool = false,
                M: Int? = nil, N: Int? = nil, K: Int? = nil) -> GPUTensor {
        let m = M ?? (transA ? A.cols : A.rows)
        let k = K ?? (transA ? A.rows : A.cols)
        let n = N ?? (transB ? B.rows : B.cols)
        let C = GPUTensor(engine: engine, rows: m, cols: n)
        // Тайловое — только когда ОБА измерения велики; при крошечных M/N и огромном K
        // тайловый с барьерами на каждой итерации K медленнее наивного.
        if m * k * n >= 1_000_000 && m > 32 && n > 32 {
            enc.setComputePipelineState(engine.matmulTiledPS)
            setBuffers([A.buf, B.buf, C.buf])
            setUInt(UInt32(m), index: 3)
            setUInt(UInt32(k), index: 4)
            setUInt(UInt32(n), index: 5)
            setUInt(transA ? 1 : 0, index: 6)
            setUInt(transB ? 1 : 0, index: 7)
            dispatchGroups(16, 16, (m + 15) / 16, (n + 15) / 16)
            return C
        }
        enc.setComputePipelineState(engine.matmulPS)
        setBuffers([A.buf, B.buf, C.buf])
        setUInt(UInt32(m), index: 3)
        setUInt(UInt32(k), index: 4)
        setUInt(UInt32(n), index: 5)
        setUInt(transA ? 1 : 0, index: 6)
        setUInt(transB ? 1 : 0, index: 7)
        dispatch(16, 16, m, n)
        return C
    }

    // R = A + B (поэлементно)
    func add(_ A: GPUTensor, _ B: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.addPS)
        setBuffers([A.buf, B.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    // C = A + bias (1×cols), broadcast по строкам
    func addBias(_ A: GPUTensor, _ bias: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.addBiasPS)
        setBuffers([A.buf, bias.buf, C.buf])
        setUInt(UInt32(A.cols), index: 3)
        dispatch(256, 1, A.n, 1)
        return C
    }

    // R = A * B (поэлементно)
    func mul(_ A: GPUTensor, _ B: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.mulPS)
        setBuffers([A.buf, B.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    // R = A - B (поэлементно)
    func sub(_ A: GPUTensor, _ B: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.subPS)
        setBuffers([A.buf, B.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    // R = s * A
    func smul(_ s: Float, _ A: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.smulPS)
        setBuffers([A.buf, C.buf])
        var sv = s
        enc.setBytes(&sv, length: 4, index: 2)
        dispatch(256, 1, A.n, 1)
        return C
    }

    // R = tanh(A)
    func tanh(_ A: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.tanhPS)
        setBuffers([A.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    // R = 1 - tanh²(A)
    func tanhGrad(_ A: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.tanhGradPS)
        setBuffers([A.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    // C = Σ_строк A (результат 1×cols)
    func sumCols(_ A: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: 1, cols: A.cols)
        // Ядро sumcols_kernel не использует буфер work (редукция идёт одним
        // потоком на столбец). Раньше здесь брался буфер из пула и сразу
        // возвращался в пул — до завершения command buffer. Теперь буфер
        // не нужен вовсе: в слот 2 передаём сам результат.
        enc.setComputePipelineState(engine.sumColsPS)
        setBuffers([A.buf, C.buf, C.buf])
        setUInt(UInt32(A.rows), index: 3)
        setUInt(UInt32(A.cols), index: 4)
        dispatch(256, 1, A.cols, 1)
        return C
    }

    // C = A*√α + B*√(1-α)
    func qsample(_ A: GPUTensor, _ B: GPUTensor, alpha: Float) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.qsamplePS)
        setBuffers([A.buf, B.buf, C.buf])
        var av = alpha
        enc.setBytes(&av, length: 4, index: 3)
        dispatch(256, 1, A.n, 1)
        return C
    }

    // C = (Z - (1-α)·pred)/α
    func denoise(_ Z: GPUTensor, _ pred: GPUTensor, alpha: Float) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: Z.rows, cols: Z.cols)
        enc.setComputePipelineState(engine.denoisePS)
        setBuffers([Z.buf, pred.buf, C.buf])
        var av = alpha
        enc.setBytes(&av, length: 4, index: 3)
        dispatch(256, 1, Z.n, 1)
        return C
    }

    // Adam: param -= lr * mHat / sqrt(vHat+eps)
    func adamUpdate(_ param: GPUTensor, _ grad: GPUTensor,
                    _ m: GPUTensor, _ v: GPUTensor, t: Int, lr: Float) {
        enc.setComputePipelineState(engine.adamPS)
        setBuffers([param.buf, grad.buf, m.buf, v.buf])
        var tv = UInt32(t), lrv = lr
        enc.setBytes(&tv, length: 4, index: 4)
        enc.setBytes(&lrv, length: 4, index: 5)
        dispatch(256, 1, param.n, 1)
    }

    /// Копирование тензора ВНУТРИ текущего command buffer (blit, без ожидания).
    /// Нужно, чтобы инициализировать накопитель градиентов значением первого
    /// слагаемого, а не складывать его с неинициализированной памятью.
    func copy(_ dst: GPUTensor, from src: GPUTensor) {
        precondition(dst.n == src.n, "copy: размеры должны совпадать")
        enc.endEncoding()
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: src.buf, sourceOffset: 0, to: dst.buf, destinationOffset: 0,
                  size: dst.n * 4)
        blit.endEncoding()
        enc = cmd.makeComputeCommandEncoder()!
    }

    /// Обнуление тензора ВНУТРИ текущего command buffer.
    /// Metal не гарантирует нулевое содержимое нового приватного буфера,
    /// поэтому любой аккумулятор обязан обнуляться явно.
    func zeroFill(_ t: GPUTensor) {
        enc.endEncoding()
        let blit = cmd.makeBlitCommandEncoder()!
        blit.fill(buffer: t.buf, range: 0..<(t.n * 4), value: 0)
        blit.endEncoding()
        enc = cmd.makeComputeCommandEncoder()!
    }

    // ---------- Свёрточный путь ----------

    /// im2col: X (ic, ih, iw) → C (ic*kh*kw, oh*ow)
    func im2col(_ X: GPUTensor, ih: Int, iw: Int, ic: Int,
                kh: Int, kw: Int, stride: Int, pad: Int) -> GPUTensor {
        let oh = (ih + 2 * pad - kh) / stride + 1
        let ow = (iw + 2 * pad - kw) / stride + 1
        let rows = ic * kh * kw
        let C = GPUTensor(engine: engine, rows: rows, cols: oh * ow)
        enc.setComputePipelineState(engine.im2col3dPS)
        setBuffers([X.buf, C.buf])
        setUInt(UInt32(ic), index: 2)
        setUInt(UInt32(ih), index: 3)
        setUInt(UInt32(iw), index: 4)
        setUInt(UInt32(kh), index: 5)
        setUInt(UInt32(kw), index: 6)
        setUInt(UInt32(stride), index: 7)
        setUInt(UInt32(pad), index: 8)
        setUInt(UInt32(oh), index: 9)
        setUInt(UInt32(ow), index: 10)
        dispatch3D(16, 8, 1, ow, oh, rows)
        return C
    }

    /// col2im: градиент (ic*kh*kw, oh*ow) → (ic, ih, iw)
    func col2im(_ C: GPUTensor, ic: Int, ih: Int, iw: Int,
                kh: Int, kw: Int, stride: Int, pad: Int, oh: Int, ow: Int) -> GPUTensor {
        let G = GPUTensor(engine: engine, rows: ic, cols: ih * iw)
        if stride == 1 {
            enc.setComputePipelineState(engine.col2im3dS1PS)
            setBuffers([C.buf, G.buf])
            setUInt(UInt32(ic), index: 2)
            setUInt(UInt32(ih), index: 3)
            setUInt(UInt32(iw), index: 4)
            setUInt(UInt32(kh), index: 5)
            setUInt(UInt32(kw), index: 6)
            setUInt(UInt32(pad), index: 7)
            setUInt(UInt32(oh), index: 8)
            setUInt(UInt32(ow), index: 9)
            dispatch3D(16, 8, 1, iw, ih, ic)
            return G
        }
        enc.setComputePipelineState(engine.col2im3dPS)
        setBuffers([C.buf, G.buf])
        setUInt(UInt32(ic), index: 2)
        setUInt(UInt32(ih), index: 3)
        setUInt(UInt32(iw), index: 4)
        setUInt(UInt32(kh), index: 5)
        setUInt(UInt32(kw), index: 6)
        setUInt(UInt32(stride), index: 7)
        setUInt(UInt32(pad), index: 8)
        setUInt(UInt32(oh), index: 9)
        setUInt(UInt32(ow), index: 10)
        dispatch3D(16, 8, 1, iw, ih, ic)
        return G
    }

    /// (c, h, w) → (c, 2h, 2w)
    func upsample2x(_ X: GPUTensor, h: Int, w: Int) -> GPUTensor {
        let c = X.rows
        let Y = GPUTensor(engine: engine, rows: c, cols: 4 * h * w)
        enc.setComputePipelineState(engine.upsample3dPS)
        setBuffers([X.buf, Y.buf])
        setUInt(UInt32(h), index: 2)
        setUInt(UInt32(w), index: 3)
        dispatch3D(16, 8, 1, 2 * w, 2 * h, c)
        return Y
    }

    /// (c, 2h, 2w) → (c, h, w), сумма 2×2
    /// gradW[r,c] = Σ_k A[r,k]·B[c,k] (A: oc×K, B: kk×K → oc×kk), групповая редукция.
    func gradW(_ A: GPUTensor, _ B: GPUTensor, oc: Int, kk: Int) -> GPUTensor {
        let K = A.cols
        let C = GPUTensor(engine: engine, rows: oc, cols: kk)
        // Детерминированный plain-вариант: один поток на элемент (без shared-памяти).
        enc.setComputePipelineState(engine.gradWPlainPS)
        setBuffers([A.buf, B.buf, C.buf])
        setUInt(UInt32(K), index: 3)
        setUInt(UInt32(kk), index: 4)
        dispatch(16, 16, oc, kk)
        return C
    }

    /// Прямой backprop входа свёртки (stride=1): dx (ic×ih×iw) из dOut (oc×ih×iw) и W.
    func convBackInputS1(_ dOut: GPUTensor, _ W: GPUTensor,
                         ic: Int, oc: Int, ih: Int, iw: Int, k: Int, pad: Int) -> GPUTensor {
        let dx = GPUTensor(engine: engine, rows: ic, cols: ih * iw)
        enc.setComputePipelineState(engine.convBackInputS1PS)
        setBuffers([dOut.buf, W.buf, dx.buf])
        setUInt(UInt32(ic), index: 3)
        setUInt(UInt32(oc), index: 4)
        setUInt(UInt32(ih), index: 5)
        setUInt(UInt32(iw), index: 6)
        setUInt(UInt32(k), index: 7)
        setUInt(UInt32(k), index: 8)
        setUInt(UInt32(pad), index: 9)
        dispatch3D(16, 8, 1, iw, ih, ic)
        return dx
    }

    /// (c, 2h, 2w) → (c, h, w), сумма 2×2
    func downsample2x(_ X: GPUTensor, h: Int, w: Int) -> GPUTensor {        let c = X.rows
        let Y = GPUTensor(engine: engine, rows: c, cols: h * w)
        enc.setComputePipelineState(engine.downsample3dPS)
        setBuffers([X.buf, Y.buf])
        setUInt(UInt32(h), index: 2)
        setUInt(UInt32(w), index: 3)
        dispatch3D(16, 8, 1, w, h, c)
        return Y
    }

    func biasRow(_ A: GPUTensor, _ b: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.biasRow3dPS)
        setBuffers([A.buf, b.buf, C.buf])
        setUInt(UInt32(A.cols), index: 3)
        dispatch3D(16, 8, 1, A.cols, A.rows, 1)
        return C
    }

    func biasRowGrad(_ G: GPUTensor) -> GPUTensor {
        let out = GPUTensor(engine: engine, rows: G.rows, cols: 1)
        if G.cols > 4096 {
            let chunks = max(8, min(128, G.cols / 4096))
            // Обнуление — в том же command buffer (раньше здесь создавался
            // второй буфер с отдельным commit+wait внутри открытой сессии).
            zeroFill(out)
            enc.setComputePipelineState(engine.biasRowGradAtomicPS)
            setBuffers([G.buf, out.buf])
            setUInt(UInt32(G.cols), index: 2)
            setUInt(UInt32(chunks), index: 3)
            dispatch3D(64, 1, 1, chunks, G.rows, 1)
            return out
        }
        enc.setComputePipelineState(engine.biasRowGradPS)
        setBuffers([G.buf, out.buf])
        setUInt(UInt32(G.cols), index: 2)
        dispatch(64, 1, G.rows, 1)
        return out
    }

    func relu(_ A: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.reluPS)
        setBuffers([A.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    func reluGrad(_ Y: GPUTensor, _ G: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: Y.rows, cols: Y.cols)
        enc.setComputePipelineState(engine.reluGradPS)
        setBuffers([Y.buf, G.buf, C.buf])
        dispatch(256, 1, Y.n, 1)
        return C
    }

    func sigmoid(_ A: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.sigmoidPS)
        setBuffers([A.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    func sigmoidGrad(_ Z: GPUTensor, _ G: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: Z.rows, cols: Z.cols)
        enc.setComputePipelineState(engine.sigmoidGradPS)
        setBuffers([Z.buf, G.buf, C.buf])
        dispatch(256, 1, Z.n, 1)
        return C
    }

    func addInPlace(_ A: GPUTensor, _ B: GPUTensor) {
        enc.setComputePipelineState(engine.addInPlacePS)
        setBuffers([A.buf, B.buf])
        dispatch(256, 1, A.n, 1)
    }

    // ---------- Вариационный (VAE) путь ----------

    /// R = exp(A)
    func exp(_ A: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.expPS)
        setBuffers([A.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    /// σ = exp(0.5·clamp(logvar, -10, 10))
    func expHalf(_ LV: GPUTensor) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: LV.rows, cols: LV.cols)
        enc.setComputePipelineState(engine.expHalfPS)
        setBuffers([LV.buf, C.buf])
        dispatch(256, 1, LV.n, 1)
        return C
    }

    /// R = A + s
    func addConst(_ A: GPUTensor, _ s: Float) -> GPUTensor {
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.addConstPS)
        setBuffers([A.buf, C.buf])
        var sv = s
        enc.setBytes(&sv, length: 4, index: 2)
        dispatch(256, 1, A.n, 1)
        return C
    }

    /// C = A + B·D (сэмплирование z = mu + σ·ε)
    func fma(_ A: GPUTensor, _ B: GPUTensor, _ D: GPUTensor) -> GPUTensor {
        precondition(A.n == B.n && A.n == D.n, "fma: размеры должны совпадать")
        let C = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.fmaPS)
        setBuffers([A.buf, B.buf, D.buf, C.buf])
        dispatch(256, 1, A.n, 1)
        return C
    }

    /// X (2c, plane) → (mu (c, plane), logvar (c, plane))
    func split2(_ X: GPUTensor, c: Int, plane: Int) -> (mu: GPUTensor, logvar: GPUTensor) {
        precondition(X.n == 2 * c * plane, "split2: неверная форма входа")
        let mu = GPUTensor(engine: engine, rows: c, cols: plane)
        let lv = GPUTensor(engine: engine, rows: c, cols: plane)
        enc.setComputePipelineState(engine.split2PS)
        setBuffers([X.buf, mu.buf, lv.buf])
        setUInt(UInt32(c), index: 3)
        setUInt(UInt32(plane), index: 4)
        dispatch(16, 8, plane, c)
        return (mu, lv)
    }

    /// (mu, logvar) (c, plane) → X (2c, plane) — обратное к split2
    func merge2(_ A: GPUTensor, _ B: GPUTensor, c: Int, plane: Int) -> GPUTensor {
        precondition(A.n == c * plane && B.n == c * plane, "merge2: неверная форма входа")
        let X = GPUTensor(engine: engine, rows: 2 * c, cols: plane)
        enc.setComputePipelineState(engine.merge2PS)
        setBuffers([A.buf, B.buf, X.buf])
        setUInt(UInt32(c), index: 3)
        setUInt(UInt32(plane), index: 4)
        dispatch(16, 8, plane, c)
        return X
    }

    /// Детерминированный стандартный нормальный шум (seed, index) → N(0,1).
    func randn(seed: UInt32, rows: Int, cols: Int) -> GPUTensor {
        let T = GPUTensor(engine: engine, rows: rows, cols: cols)
        enc.setComputePipelineState(engine.randnPS)
        setBuffers([T.buf])
        setUInt(seed, index: 1)
        setUInt(UInt32(T.n), index: 2)
        dispatch(256, 1, T.n, 1)
        return T
    }

    /// KL(N(mu, σ²) ‖ N(0,1)) — сумма по всем элементам, читается после `end()`.
    func kl(mu: GPUTensor, logvar: GPUTensor) -> MTLBuffer {
        precondition(mu.n == logvar.n, "kl: размеры должны совпадать")
        let out = engine.device.makeBuffer(length: 4, options: .storageModeShared)!
        enc.setComputePipelineState(engine.klPS)
        setBuffers([mu.buf, logvar.buf, out])
        setUInt(UInt32(mu.n), index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        return out
    }

    /// Градиенты вариационного выхода (см. vae_grad_kernel).
    func vaeGrad(dz: GPUTensor, mu: GPUTensor, logvar: GPUTensor, eps: GPUTensor,
                 betaN: Float, betaHalfN: Float) -> (dmu: GPUTensor, dlogvar: GPUTensor) {
        precondition(dz.n == mu.n && dz.n == logvar.n && dz.n == eps.n,
                     "vaeGrad: размеры должны совпадать")
        let dmu = GPUTensor(engine: engine, rows: mu.rows, cols: mu.cols)
        let dlv = GPUTensor(engine: engine, rows: mu.rows, cols: mu.cols)
        enc.setComputePipelineState(engine.vaeGradPS)
        setBuffers([dz.buf, mu.buf, logvar.buf, eps.buf, dmu.buf, dlv.buf])
        var b1 = betaN, b2 = betaHalfN
        enc.setBytes(&b1, length: 4, index: 6)
        enc.setBytes(&b2, length: 4, index: 7)
        dispatch(256, 1, mu.n, 1)
        return (dmu, dlv)
    }

    // ---------- 8-канальный layered путь ----------

    /// `dL/d(recon)` по всем 8 каналам: MSE по слоям + MSE собранного кадра.
    /// См. `layered_grad_kernel` и CPU-эталон `layeredLossGradient`.
    ///
    /// Возвращает тензор (8, w·h) в CHW. Вызывается ДО `sigmoidGrad`: это
    /// градиент по выходу сигмоиды d3, а не по логитам.
    public func layeredGrad(recon: GPUTensor, target: LayeredTargetGPU,
                            layerWeight: Float = 1,
                            composeWeight: Float = 1) -> GPUTensor {
        precondition(recon.n == LayeredScene.layerChannels * target.plane,
                     "layeredGrad: реконструкция должна быть 8·w·h")
        let g = GPUTensor(engine: engine, rows: LayeredScene.layerChannels,
                          cols: target.plane)
        enc.setComputePipelineState(engine.layeredGradPS)
        setBuffers([recon.buf, target.channels.buf, target.composite.buf, g.buf])
        setUInt(UInt32(target.plane), index: 4)
        var wl = layerWeight, wc = composeWeight
        enc.setBytes(&wl, length: 4, index: 5)
        enc.setBytes(&wc, length: 4, index: 6)
        dispatch(256, 1, target.plane, 1)
        return g
    }
}
