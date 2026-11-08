// YumeLayers — Metal GPU-движок для матричного умножения.
// Шейдер компилируется из исходника (без внешних .metal файлов).
import Foundation
import Metal

public final class MetalEngine {
    public static let shared = MetalEngine()

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let deviceName: String

    // MARK: - Пул переиспользования приватных буферов (критично для длинных прогонов)

    /// Сколько раз реально выделен новый приватный буфер (через makeBuffer).
    /// Растёт только при промахе пула; с переиспользованием почти не растёт.
    public private(set) var totalBufferAllocations = 0
    private let bufferPoolLock = NSLock()
    /// Свободные буферы по размеру (байты → стек), готовые к выдаче.
    private var pooledBuffers: [Int: [MTLBuffer]] = [:]
    /// Освобождённые (retire), но ещё не готовые к выдаче: их вернёт в пул
    /// `reclaimRetiredBuffers()` ПОСЛЕ `end()`, когда GPU закончил с ними.
    private var retiredBuffers: [MTLBuffer] = []

    /// Берёт приватный буфер из пула (nil — пул не имеет подходящего).
    func takeBuffer(_ bytes: Int) -> MTLBuffer? {
        bufferPoolLock.lock(); defer { bufferPoolLock.unlock() }
        if var stack = pooledBuffers[bytes], !stack.isEmpty {
            let b = stack.removeLast()
            pooledBuffers[bytes] = stack
            return b
        }
        return nil
    }

    /// Возвращает буфер в отложенную очередь (не в свободный пул сразу —
    /// GPU может ещё читать его данные).
    func retireBuffer(_ b: MTLBuffer) {
        bufferPoolLock.lock(); defer { bufferPoolLock.unlock() }
        retiredBuffers.append(b)
    }

    /// Переносит отложенные буферы в свободный пул. Безопасно вызывать только
    /// после `end()` (все GPU-команды завершились).
    func reclaimRetiredBuffers() {
        bufferPoolLock.lock(); defer { bufferPoolLock.unlock() }
        guard !retiredBuffers.isEmpty else { return }
        for b in retiredBuffers {
            pooledBuffers[b.length, default: []].append(b)
        }
        retiredBuffers.removeAll()
    }

    /// Создаёт приватный буфер, пытаясь взять его из пула (переиспользование).
    /// При промахе — реальный makeBuffer и инкремент счётчика.
    func makePrivateBuffer(length: Int, what: String) -> MTLBuffer {
        if let pooled = takeBuffer(length) { return pooled }
        guard let b = device.makeBuffer(length: length, options: .storageModePrivate) else {
            fatalError(allocationFailureMessage(
                bytes: length,
                availableBytes: Int(device.recommendedMaxWorkingSetSize)
                    - device.currentAllocatedSize,
                what: what))
        }
        bufferPoolLock.lock(); totalBufferAllocations += 1; bufferPoolLock.unlock()
        return b
    }

    // MARK: - Пул shared staging-буферов

    /// Сколько реально выделено shared-буферов (staging для CPU↔GPU).
    ///
    /// Раньше `GPUTensor.toArray/setData` создавали новый `MTLBuffer` на
    /// КАЖДЫЙ вызов, а `sanitizeGrads` вызывал их по разу на тензор на каждом
    /// шаге. За 15k шагов это сотни тысяч ресурсов, после которых драйвер AMD
    /// встаёт внутри `newBufferWithBytes` (проверено стеком процесса).
    /// Теперь staging берётся из пула и возвращается после `waitUntilCompleted`.
    public private(set) var totalStagingAllocations = 0
    private let stagingPoolLock = NSLock()
    private var pooledStaging: [Int: [MTLBuffer]] = [:]

    /// Берёт shared-буфер нужного размера из пула (или выделяет новый).
    func takeStaging(_ bytes: Int) -> MTLBuffer {
        let key = max(bytes, 1)
        stagingPoolLock.lock()
        if var stack = pooledStaging[key], let b = stack.popLast() {
            pooledStaging[key] = stack
            stagingPoolLock.unlock()
            return b
        }
        stagingPoolLock.unlock()
        guard let b = device.makeBuffer(length: key, options: .storageModeShared) else {
            fatalError(allocationFailureMessage(
                bytes: key,
                availableBytes: Int(device.recommendedMaxWorkingSetSize)
                    - device.currentAllocatedSize,
                what: "staging-буфер \(key / 1024) КиБ"))
        }
        stagingPoolLock.lock(); totalStagingAllocations += 1; stagingPoolLock.unlock()
        return b
    }

    /// Возвращает staging в пул. Вызывать только после завершения GPU-команды,
    /// которая его читала/писала.
    func putStaging(_ b: MTLBuffer) {
        stagingPoolLock.lock(); defer { stagingPoolLock.unlock() }
        pooledStaging[b.length, default: []].append(b)
    }

    /// Ждёт command buffer с жёстким пределом времени.
    ///
    /// `waitUntilCompleted()` при зависании GPU не возвращается никогда —
    /// именно так обучение «висело» десятки минут. Для одиночных blit-ов
    /// (чтение/запись тензоров) используем тот же предел, что и `GpuSession`.
    ///
    /// Ожидание построено на ДВУХ источниках сигнала:
    ///   • обработчик завершения (`addCompletedHandler`) — обычный путь;
    ///   • статус самого буфера (`cmd.status`) — страховка. Обработчик Metal
    ///     иногда не доставляется, и тогда ожидание на семафоре висело до
    ///     предела и убивало процесс («GPU не ответила за 30 с»), хотя буфер
    ///     давно был выполнен.
    ///
    /// Возвращает «время GPU» в мс, ограниченное стенным временем ожидания:
    /// пустые/невнятные показания драйвера (gpuStartTime = 0) больше не
    /// превращаются в гигантскую паузу регулятора.
    @discardableResult
    func waitForCompletion(_ cmd: MTLCommandBuffer, what: String,
                           seconds: Double = 30) -> Double {
        let sem = DispatchSemaphore(value: 0)
        cmd.addCompletedHandler { _ in sem.signal() }
        cmd.commit()
        let start = Date()
        // Драйвер иногда не доставляет обработчик завершения (см. Фазу 4).
        // Даём ему короткий грейс; если обработчик не пришёл — уходим в
        // плотный опрос статуса (0.5 мс), а не платим по 5 мс на каждую
        // потерянную доставку. Это снимает основную деградацию длинных
        // прогонов и не зависит от надёжности обработчиков.
        var handled = sem.wait(timeout: .now() + 0.02) == .success
        if !handled {
            var logged = false
            while true {
                let elapsed = Date().timeIntervalSince(start)
                let state = MetalEngine.bufferState(cmd.status)
                switch gpuWaitOutcome(state: state, handlerSignaled: false,
                                      elapsedSeconds: elapsed, deadlineSeconds: seconds) {
                case .done:
                    if !logged {
                        yumeLog("⚠️ обработчик завершения не доставлен (\(what)) — "
                              + "статус буфера: \(state). Продолжаю по статусу.")
                        logged = true
                    }
                    handled = true
                case .timedOut:
                    yumeLog("❌ GPU не ответила за \(Int(seconds)) с — \(what). "
                          + "Статус буфера: \(state), ошибка: \(cmd.error?.localizedDescription ?? "нет"). "
                          + "Аварийный выход, чтобы не перегрузить WindowServer.")
                    exit(70)
                case .keepWaiting:
                    Thread.sleep(forTimeInterval: 0.0005)
                    continue
                }
                if handled { break }
            }
        }
        let wallMs = Date().timeIntervalSince(start) * 1000
        let reported = (cmd.gpuEndTime - cmd.gpuStartTime) * 1000
        let ms = boundedGpuMs(reportedMs: reported, wallMs: wallMs)
        if reported > wallMs + 50 {
            yumeLog(String(format: "⚠️ драйвер сообщил %.0f мс GPU при %.0f мс стенного времени (%@) — показание отброшено",
                           reported, wallMs, what))
        }
        return ms
    }

    /// Состояние command buffer в терминах политики ожидания.
    static func bufferState(_ status: MTLCommandBufferStatus) -> GpuBufferState {
        switch status {
        case .completed: return .completed
        case .error: return .failed
        default: return .running
        }
    }

    static let shaderSource = MetalKernels.part0 + "\n" + MetalKernels.part1

    let matmulPS: MTLComputePipelineState
    let matmulTiledPS: MTLComputePipelineState
    let gradWAtomicPS: MTLComputePipelineState
    let gradWReducePS: MTLComputePipelineState
    let gradWPlainPS: MTLComputePipelineState
    let addPS: MTLComputePipelineState
    let addBiasPS: MTLComputePipelineState
    let mulPS: MTLComputePipelineState
    let subPS: MTLComputePipelineState
    let smulPS: MTLComputePipelineState
    let tanhPS: MTLComputePipelineState
    let tanhGradPS: MTLComputePipelineState
    let sumColsPS: MTLComputePipelineState
    let qsamplePS: MTLComputePipelineState
    let denoisePS: MTLComputePipelineState
    let adamPS: MTLComputePipelineState

    // Свёрточный путь
    let im2colPS: MTLComputePipelineState
    let im2col3dPS: MTLComputePipelineState
    let col2imPS: MTLComputePipelineState
    let col2im3dPS: MTLComputePipelineState
    let col2imS1PS: MTLComputePipelineState
    let col2im3dS1PS: MTLComputePipelineState
    let upsample2xPS: MTLComputePipelineState
    let upsample3dPS: MTLComputePipelineState
    let downsample2xPS: MTLComputePipelineState
    let downsample3dPS: MTLComputePipelineState
    let biasRowPS: MTLComputePipelineState
    let biasRow3dPS: MTLComputePipelineState
    let biasRowGradPS: MTLComputePipelineState
    let biasRowGradAtomicPS: MTLComputePipelineState
    let convBackInputS1PS: MTLComputePipelineState
    let reluPS: MTLComputePipelineState
    let reluGradPS: MTLComputePipelineState
    let sigmoidPS: MTLComputePipelineState
    let sigmoidGradPS: MTLComputePipelineState
    let addInPlacePS: MTLComputePipelineState
    let msePS: MTLComputePipelineState

    // Вариационный путь
    let fmaPS: MTLComputePipelineState
    let expPS: MTLComputePipelineState
    let addConstPS: MTLComputePipelineState
    let expHalfPS: MTLComputePipelineState
    let split2PS: MTLComputePipelineState
    let merge2PS: MTLComputePipelineState
    let klPS: MTLComputePipelineState
    let randnPS: MTLComputePipelineState
    let vaeGradPS: MTLComputePipelineState
    // 8-канальный путь: слой + композиция
    let layeredGradPS: MTLComputePipelineState

    private static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let l = try? device.makeLibrary(source: shaderSource, options: nil) {
            return l
        }
        yumeLog("⚠️ runtime-компиляция Metal недоступна — пробую внешний xcrun metal")
        // fallback: внешний компилятор не зависит от MTLCompilerService
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("yume_shaders_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let metalFile = dir.appendingPathComponent("shaders.metal")
        try Data(shaderSource.utf8).write(to: metalFile)
        let airFile = dir.appendingPathComponent("shaders.air")
        let libFile = dir.appendingPathComponent("shaders.metallib")
        try runExternal("/usr/bin/xcrun", ["-sdk", "macosx", "metal", "-c", metalFile.path, "-o", airFile.path])
        try runExternal("/usr/bin/xcrun", ["metallib", airFile.path, "-o", libFile.path])
        let lib = try device.makeLibrary(URL: libFile)
        yumeLog("✅ Шейдеры откомпилированы внешним xcrun metal")
        return lib
    }

    private static func runExternal(_ path: String, _ args: [String]) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        try p.run()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            throw NSError(domain: "Yume", code: 20,
                          userInfo: [NSLocalizedDescriptionKey:
                            "внешний \(path) завершился с кодом \(p.terminationStatus)"])
        }
    }

    convenience init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal недоступен на этой системе")
        }
        self.init(device: dev)
    }

    /// Движок на конкретном устройстве (например, на встроенной iGPU, чтобы
    /// переложить часть независимой работы со внешней видеокарты).
    init(device dev: MTLDevice) {
        self.device = dev
        self.deviceName = dev.name
        self.queue = dev.makeCommandQueue()!
        do {
            let lib = try Self.makeLibrary(device: dev)
            func ps(_ name: String) throws -> MTLComputePipelineState {
                guard let fn = lib.makeFunction(name: name) else {
                    fatalError("шейдер не найден: \(name)")
                }
                return try dev.makeComputePipelineState(function: fn)
            }
            self.matmulPS = try ps("matmul_kernel")
            self.matmulTiledPS = try ps("matmul_tiled_kernel")
            self.gradWAtomicPS = try ps("gradw_atomic_kernel")
            self.gradWReducePS = try ps("gradw_reduce_kernel")
            self.gradWPlainPS = try ps("gradw_plain_kernel")
            self.addPS = try ps("add_kernel")
            self.addBiasPS = try ps("addbias_kernel")
            self.mulPS = try ps("mul_kernel")
            self.subPS = try ps("sub_kernel")
            self.smulPS = try ps("smul_kernel")
            self.tanhPS = try ps("tanh_kernel")
            self.tanhGradPS = try ps("tanhgrad_kernel")
            self.sumColsPS = try ps("sumcols_kernel")
            self.qsamplePS = try ps("qsample_kernel")
            self.denoisePS = try ps("denoise_kernel")
            self.adamPS = try ps("adam_kernel")
            self.im2colPS = try ps("im2col_kernel")
            self.im2col3dPS = try ps("im2col3d_kernel")
            self.col2imPS = try ps("col2im_kernel")
            self.col2im3dPS = try ps("col2im3d_kernel")
            self.col2imS1PS = try ps("col2im_s1_kernel")
            self.col2im3dS1PS = try ps("col2im3d_s1_kernel")
            self.upsample2xPS = try ps("upsample2x_kernel")
            self.upsample3dPS = try ps("upsample3d_kernel")
            self.downsample2xPS = try ps("downsample2x_kernel")
            self.downsample3dPS = try ps("downsample3d_kernel")
            self.biasRowPS = try ps("biasrow_kernel")
            self.biasRow3dPS = try ps("biasrow3d_kernel")
            self.biasRowGradPS = try ps("biasrowgrad_kernel")
            self.biasRowGradAtomicPS = try ps("biasrowgrad_atomic_kernel")
            self.convBackInputS1PS = try ps("conv_back_input_s1_kernel")
            self.reluPS = try ps("relu_kernel")
            self.reluGradPS = try ps("relugrad_kernel")
            self.sigmoidPS = try ps("sigmoid_kernel")
            self.sigmoidGradPS = try ps("sigmoidgrad_kernel")
            self.addInPlacePS = try ps("addinplace_kernel")
            self.msePS = try ps("mse_reduce_kernel")
            self.fmaPS = try ps("fma_kernel")
            self.expPS = try ps("exp_kernel")
            self.addConstPS = try ps("addconst_kernel")
            self.expHalfPS = try ps("exphalf_kernel")
            self.split2PS = try ps("split2_kernel")
            self.merge2PS = try ps("merge2_kernel")
            self.klPS = try ps("kl_reduce_kernel")
            self.randnPS = try ps("randn_kernel")
            self.vaeGradPS = try ps("vae_grad_kernel")
            self.layeredGradPS = try ps("layered_grad_kernel")
        } catch {
            fatalError("ошибка компиляции Metal-шейдеров: \(error)")
        }
    }

    /// C = A (m×k) × B (k×n). Возвращает row-major [Float].
    func multiply(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
        // Форма входов проверяется явно: `makeBuffer(bytes:length:)` копирует
        // ровно m·k·4 байт, и с коротким массивом это было чтение за границей.
        precondition(a.count == m * k, "multiply: a.count \(a.count) != m·k = \(m * k)")
        precondition(b.count == k * n, "multiply: b.count \(b.count) != k·n = \(k * n)")
        guard m > 0, k > 0, n > 0,
              let abuf = device.makeBuffer(bytes: a, length: m * k * 4, options: []),
              let bbuf = device.makeBuffer(bytes: b, length: k * n * 4, options: []),
              let cbuf = device.makeBuffer(length: m * n * 4, options: []) else {
            assertionFailure("multiply: не удалось выделить буферы — возвращаю нули")
            return [Float](repeating: 0, count: m * n)
        }
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            return [Float](repeating: 0, count: m * n)
        }
        enc.setComputePipelineState(matmulPS)
        enc.setBuffer(abuf, offset: 0, index: 0)
        enc.setBuffer(bbuf, offset: 0, index: 1)
        enc.setBuffer(cbuf, offset: 0, index: 2)
        var mm = UInt32(m), kk = UInt32(k), nn = UInt32(n)
        enc.setBytes(&mm, length: 4, index: 3)
        enc.setBytes(&kk, length: 4, index: 4)
        enc.setBytes(&nn, length: 4, index: 5)
        var tA: UInt32 = 0, tB: UInt32 = 0
        enc.setBytes(&tA, length: 4, index: 6)
        enc.setBytes(&tB, length: 4, index: 7)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        // ровно m×n нитей (современный API, избегает лишних потоков)
        let threads = MTLSize(width: m, height: n, depth: 1)
        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
        enc.endEncoding()
        waitForCompletion(cmd, what: "CPU-умножение \(m)×\(k)×\(n)")
        let ptr = cbuf.contents().assumingMemoryBound(to: Float.self)
        var out = [Float](repeating: 0, count: m * n)
        for i in 0..<(m * n) { out[i] = ptr[i] }
        return out
    }
}
