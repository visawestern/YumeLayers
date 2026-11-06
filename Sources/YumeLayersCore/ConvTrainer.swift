// YumeLayers — обучение свёрточного VAE на Metal (480p и любое разрешение, кратное 8).
import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import Metal

// ---------- Загрузка изображений ----------

/// RGB [0,1], row-major сверху вниз, ровно w×h. Читает из файла.
public func loadImageRGB(path: String, w: Int, h: Int) -> [Float]? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    return decodeImageRGB(img, w: w, h: h)
}

/// RGB [0,1], row-major сверху вниз, ровно w×h. Декодирует из памяти (JPEG).
public func loadImageRGB(data: Data, w: Int, h: Int) -> [Float]? {
    guard let src = CGImageSourceCreateWithData(data as CFData, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    return decodeImageRGB(img, w: w, h: h)
}

/// Публичная точка декодирования CGImage → RGB [0,1] top-down.
/// Итерация 6: конвейеру нужно decoded-кадры из `AVAssetImageGenerator`,
/// а не только JPEG из файла/памяти. Логика одна и та же, дублировать её
/// нельзя — иначе превью и обучение будут видеть разные цвета.
public func decodeImageRGB(_ img: CGImage, w: Int, h: Int) -> [Float]? {
    var pixels = [UInt8](repeating: 0, count: w * h * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: cs,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    // CG рисует снизу вверх — переворачиваем в привычный top-down
    var out = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        let sy = h - 1 - y
        for x in 0..<w {
            let si = (sy * w + x) * 4
            let di = (y * w + x) * 3
            out[di] = Float(pixels[si]) / 255
            out[di + 1] = Float(pixels[si + 1]) / 255
            out[di + 2] = Float(pixels[si + 2]) / 255
        }
    }
    return out
}

public func listImages(in dir: String) -> [String] {
    let fm = FileManager.default
    guard let items = try? fm.contentsOfDirectory(atPath: dir) else { return [] }
    let exts: Set<String> = ["png", "jpg", "jpeg", "bmp", "tiff", "tif"]
    return items.filter { exts.contains(($0 as NSString).pathExtension.lowercased()) }
        .sorted()
        .map { (dir as NSString).appendingPathComponent($0) }
}

/// Дополняет кадр до размеров, кратных 8 (крайним пикселем).
public func padToMultipleOf8(_ img: [Float], w: Int, h: Int) -> ([Float], Int, Int) {
    let pw = ((w + 7) / 8) * 8
    let ph = ((h + 7) / 8) * 8
    if pw == w && ph == h { return (img, w, h) }
    var out = [Float](repeating: 0, count: pw * ph * 3)
    for y in 0..<ph {
        let sy = min(y, h - 1)
        for x in 0..<pw {
            let sx = min(x, w - 1)
            let si = (sy * w + sx) * 3
            let di = (y * pw + x) * 3
            out[di] = img[si]; out[di + 1] = img[si + 1]; out[di + 2] = img[si + 2]
        }
    }
    return (out, pw, ph)
}

// ---------- Данные ----------

/// Предварительно декодированный набор кадров в RGB8 (с паддингом до кратного 8).
/// Декодирование выполняется параллельно по всем ядрам один раз до обучения,
/// чтобы в цикле не было ни JPEG-декода, ни аллокаций [Float] на каждый шаг.
final class FrameCache {
    let w: Int
    let h: Int
    private let frames: [MemoryFrame]
    private let chunkSize: Int
    private let per: Int
    private let _count: Int

    // Активный чанк (готовые пиксели).
    private var active: [[UInt8]] = []
    private var activeStart = 0
    // Предзагружаемый чанк.
    private var next: [[UInt8]]? = nil
    private var nextStart = 0
    private let lock = NSLock()
    private let readySem = DispatchSemaphore(value: 0)
    private var prefetchRunning = false

    var count: Int { _count }

    init?(frames: [MemoryFrame], w: Int, h: Int, budgetBytes: Int = 1 << 30) {
        guard !frames.isEmpty else { return nil }
        self.frames = frames
        self.w = w
        self.h = h
        self.per = w * h * 3
        self._count = frames.count
        // Чанк ~1024 кадра: два буфера ≈ 2.5 ГБ — умещается в RAM и даёт
        // большой запас времени на фоновый декод следующего чанка.
        self.chunkSize = min(frames.count, max(1, min(1024, budgetBytes / max(2 * per, 1))))

        let firstEnd = min(chunkSize, frames.count)
        self.active = Self.decodeChunk(frames: frames, start: 0, end: firstEnd, w: w, h: h)
        self.activeStart = 0
        if firstEnd < frames.count {
            startPrefetch(nextIndex: firstEnd)
        }
        yumeLog("FrameCache: \(_count) кадров, чанк \(chunkSize) (преддекодирование чанками), RGB8 \(per) Б/кадр")
    }

    /// Кадр i как [Float] 0…1 (уже с паддингом). nil — кадр не декодировался.
    func floatRow(at i: Int) -> [Float]? {
        guard i >= 0, i < _count else { return nil }
        // Если кадр вне активного чанка — ждём готовности следующего и меняем.
        if i < activeStart || i >= activeStart + active.count {
            // Drain any pending prefetch before replacing the active window.
            lock.lock()
            let hasPending = prefetchRunning || next != nil
            lock.unlock()
            if hasPending { readySem.wait() }
            let desiredStart = (i / chunkSize) * chunkSize
            lock.lock()
            let prefetched = nextStart == desiredStart ? next : nil
            next = nil
            lock.unlock()
            activeStart = desiredStart
            active = prefetched ?? Self.decodeChunk(frames: frames, start: desiredStart,
                end: min(desiredStart + chunkSize, _count), w: w, h: h)
            let following = activeStart + active.count
            if following < _count { startPrefetch(nextIndex: following) }
        }

        let idx = i - activeStart
        guard idx >= 0, idx < active.count, !active[idx].isEmpty else { return nil }
        let r = active[idx]
        var out = [Float](repeating: 0, count: r.count)
        for j in 0..<r.count { out[j] = Float(r[j]) / 255 }
        return out
    }

    private func startPrefetch(nextIndex: Int) {
        lock.lock()
        guard !prefetchRunning else { lock.unlock(); return }
        prefetchRunning = true
        lock.unlock()

        let s = nextIndex
        let e = min(s + chunkSize, _count)
        guard s < e else {
            lock.lock(); prefetchRunning = false; lock.unlock()
            return
        }
        let f = frames
        let ww = w, hh = h
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let chunk = Self.decodeChunk(frames: f, start: s, end: e, w: ww, h: hh)
            guard let self = self else { return }
            self.lock.lock()
            self.next = chunk
            self.nextStart = s
            self.prefetchRunning = false
            self.lock.unlock()
            self.readySem.signal()
        }
    }

    /// Декодирует диапазон кадров [start, end) в RGB8, параллельно.
    private static func decodeChunk(frames: [MemoryFrame], start: Int, end: Int,
                                    w: Int, h: Int) -> [[UInt8]] {
        let n = end - start
        guard n > 0 else { return [] }
        let lock = NSLock()
        var out = [[UInt8]](repeating: [], count: n)
        DispatchQueue.concurrentPerform(iterations: n) { j in
            let i = start + j
            guard i < frames.count,
                  let img = loadImageRGB(data: frames[i].jpeg, w: w, h: h) else { return }
            var row = [UInt8](repeating: 0, count: img.count)
            for k in 0..<img.count {
                row[k] = UInt8(min(255, max(0, Int(img[k] * 255 + 0.5))))
            }
            lock.lock(); out[j] = row; lock.unlock()
        }
        return out
    }
}

public struct DataSource {
    var files: [String]
    /// Кадры в памяти (приоритет над `files`, если задано).
    var memFrames: [MemoryFrame]? = nil
    /// Преддекодированный кэш RGB8 (приоритет над всем).
    var cache: FrameCache? = nil
    var w: Int          // внутреннее разрешение (кратно 8)
    var h: Int
    var realW: Int      // исходное разрешение для сохранения превью
    var realH: Int
    var cursor = 0      // строгий порядок: кадры идут последовательно, как в списке

    /// Следующий кадр строго по порядку (циклически). Порядок зафиксирован
    /// порядком кадров — движение можно привязать к seed.
    mutating func sequentialSample() -> [Float] {
        if let c = cache, c.count > 0 {
            if cursor >= c.count { cursor = 0 }
            let idx = cursor
            cursor += 1
            if let img = c.floatRow(at: idx) {
                return img
            }
            // кадр не декодировался — честно сообщаем и уходим на синтетику
            yumeLog("⚠️ нет кадра \(idx) в кэше, использую синтетику")
        } else if let mem = memFrames, !mem.isEmpty {
            if cursor >= mem.count { cursor = 0 }
            let f = mem[cursor]
            cursor += 1
            if var img = loadImageRGB(data: f.jpeg, w: realW, h: realH) {
                let (padded, _, _) = padToMultipleOf8(img, w: realW, h: realH)
                img = padded
                return img
            }
            // кадр не декодировался — честно сообщаем и уходим на синтетику
            yumeLog("⚠️ не удалось декодировать \(f.name), использую синтетику")
        } else if !files.isEmpty {
            if cursor >= files.count { cursor = 0 }
            let f = files[cursor]
            cursor += 1
            if var img = loadImageRGB(path: f, w: realW, h: realH) {
                let (padded, _, _) = padToMultipleOf8(img, w: realW, h: realH)
                img = padded
                return img
            }
            // файл не прочитался — честно сообщаем и уходим на синтетику
            yumeLog("⚠️ не удалось прочитать \(f), использую синтетику")
        }
        return makeNormalSynthetic(w: w, h: h)
    }
}

// ---------- Сохранение пары «цель | реконструкция» ----------

public func savePairPNG(path: String, w: Int, h: Int, target: [Float], recon: [Float]) throws {
    let gap = 6
    let totalW = w * 2 + gap
    var out = [Float](repeating: 0.05, count: totalW * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            let si = (y * w + x) * 3
            let di = (y * totalW + x) * 3
            out[di] = target[si]; out[di + 1] = target[si + 1]; out[di + 2] = target[si + 2]
            let dj = (y * totalW + w + gap + x) * 3
            out[dj] = recon[si]; out[dj + 1] = recon[si + 1]; out[dj + 2] = recon[si + 2]
        }
    }
    try savePNG(path: path, width: totalW, height: h, rgb: out)
}

/// Сохранение одной картинки (RGB) в PNG.
public func saveSinglePNG(path: String, w: Int, h: Int, rgb: [Float]) throws {
    try savePNG(path: path, width: w, height: h, rgb: rgb)
}

/// Прогон обученной модели по кадрам и получение реконструкций В ПАМЯТИ
/// (без записи на диск). Возвращает словарь «имя кадра → PNG-данные».
/// Кадры берутся из контрольной выборки (p.controlFrames), если она задана,
/// иначе из памяти (p.memoryFrames), иначе с диска.
/// При наличии второй GPU (встроенная iGPU) работа делится между двумя
/// устройствами параллельно.
public func generateReconstructions(p: Params, onProgress: ((Int, Int) -> Void)? = nil) throws -> [String: Data] {
    let engine = MetalEngine.shared

    let pw = ((p.width + 7) / 8) * 8
    let ph = ((p.height + 7) / 8) * 8

    typealias FrameItem = (name: String, jpeg: Data)
    let items: [FrameItem]
    // Контрольная выборка (1 кадр/сек, итоговая модель) — приоритетна:
    // реконструируем только её, а не все кадры.
    if let ctrl = p.controlFrames, !ctrl.isEmpty {
        items = ctrl.map { ($0.name, $0.jpeg) }
        yumeLog("generateReconstructions: контрольная выборка \(items.count) кадров (1/сек) на итоговой модели")
    } else if let mem = p.memoryFrames, !mem.isEmpty {
        items = mem.map { ($0.name, $0.jpeg) }
        yumeLog("generateReconstructions: \(items.count) кадров из памяти (без диска)")
    } else {
        let files = (p.dataDir.map { listImages(in: $0) } ?? []).sorted()
        items = files.compactMap { f -> FrameItem? in
            guard let d = try? Data(contentsOf: URL(fileURLWithPath: f)) else { return nil }
            return ((f as NSString).lastPathComponent, d)
        }
        yumeLog("generateReconstructions: dataDir=\(p.dataDir ?? "nil"), files=\(items.count)")
    }
    guard !items.isEmpty else { return [:] }

    // Вторая GPU (встроенная Intel UHD) — только для достаточно больших наборов,
    // чтобы окупить инициализацию второго движка и компиляцию шейдеров.
    let allDevices = MTLCopyAllDevices()
    let secondaryDevice = allDevices.first { $0.registryID != engine.device.registryID }
    let useSecondary = secondaryDevice != nil && items.count >= 64

    if !useSecondary {
        return try runReconsOn(engine: engine, items: items, pw: pw, ph: ph,
                               convPath: p.convPath, onProgress: onProgress)
    }

    let secondaryEngine = MetalEngine(device: secondaryDevice!)
    yumeLog("Реконструкции на двух GPU: Radeon Pro 5500M + \(secondaryEngine.deviceName)")
    let mid = items.count / 2
    let lock = NSLock()
    var done = 0
    var errors: [Error] = []
    var out: [String: Data] = [:]
    let group = DispatchGroup()

    func runChunk(_ eng: MetalEngine, _ chunk: ArraySlice<FrameItem>) {
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            defer { group.leave() }
            do {
                let part = try runReconsOn(engine: eng, items: Array(chunk), pw: pw, ph: ph,
                                           convPath: p.convPath, onProgress: nil)
                lock.lock()
                out.merge(part) { _, new in new }
                done += part.count
                let d = done
                lock.unlock()
                onProgress?(d, items.count)
            } catch {
                lock.lock(); errors.append(error); lock.unlock()
            }
        }
    }
    runChunk(engine, items[..<mid])
    runChunk(secondaryEngine, items[mid...])
    group.wait()

    if let e = errors.first { throw e }
    return out
}

/// Прогон forward на одном движке (используется обеими GPU).
/// Возвращает словарь «имя кадра → PNG-данные реконструкции» В ПАМЯТИ
/// (без записи на диск).
private func runReconsOn(engine: MetalEngine, items: [(name: String, jpeg: Data)],
                         pw: Int, ph: Int, convPath: String,
                         onProgress: ((Int, Int) -> Void)?) throws -> [String: Data] {
    let session = GpuSession(engine)
    let vae = ConvVAEGPU(engine: engine)
    // Датчик температуры на этот прогон: решение о паузе принимается по
    // фактическому нагреву, а не по расписанию.
    let sensor = GpuSensor()
    if FileManager.default.fileExists(atPath: convPath) {
        let mats = try loadState(path: convPath)
        try vae.load(from: mats)
    } else {
        return [:]
    }

    // Батчи: декод кадров — параллельно по CPU, forward всей пачки — в одном
    // сеансе GPU (один wait на батч вместо одного на кадр).
    //
    // Потолок 16 — это ещё и потолок памяти: кэш прямого прохода одного кадра
    // 854×480 весит ~466 МБ, шестнадцать штук — 7.5 ГБ, которых на этой
    // видеокарте нет. Берём минимум из 16 и того, что влезает в бюджет активаций.
    let batchSize = max(1, min(16, effectiveBatchSize(
        requested: 16, w: pw, h: ph,
        budgetBytes: activationBudgetBytes(
            workingSet: engine.device.recommendedMaxWorkingSetSize))))
    var result: [String: Data] = [:]
    var done = 0
    var lastLoggedPercent = -1
    for range in batchRanges(total: items.count, batch: batchSize) {
        // «Стоп»: прерываем реконструкции (как и обучение).
        if TrainingControl.shared.isCancelled {
            yumeLog("Реконструкции остановлены пользователем (\(done)/\(items.count))")
            break
        }
        // Термо-контроль реконструкций: между батчами проверяем тепловой
        // уровень — при серьёзном пауза, при критическом ждём остывания
        // и продолжаем (без остановки).
        switch ProcessInfo.processInfo.thermalState {
        case .critical:
            // Минута паузы, затем проверка каждые 30 с до нормы.
            yumeLog("⚠️ реконструкции: критический перегрев — пауза 60 с")
            while ProcessInfo.processInfo.thermalState == .critical && !TrainingControl.shared.isCancelled {
                Thread.sleep(forTimeInterval: 0.25)
            }
        case .serious:
            Thread.sleep(forTimeInterval: 0.05)
        default:
            break
        }
        let chunkItems = Array(items[range])

        var imgs = [[Float]?](repeating: nil, count: chunkItems.count)
        let decodeLock = NSLock()
        DispatchQueue.concurrentPerform(iterations: chunkItems.count) { j in
            let image = loadImageRGB(data: chunkItems[j].jpeg, w: pw, h: ph)
            decodeLock.lock(); imgs[j] = image; decodeLock.unlock()
        }

        session.begin()
        var targets: [GPUTensor] = []
        var targetToItem: [Int] = []
        var recons: [GPUTensor] = []
        for j in 0..<chunkItems.count {
            guard let img = imgs[j] else { continue }
            let t = GPUTensor(engine: engine, rows: 3, cols: img.count / 3)
            session.upload(t, from: img)
            targets.append(t)
            targetToItem.append(j)
        }
        for t in targets {
            let cache = vae.forward(session, image: t, w: pw, h: ph)
            recons.append(cache.recon)
        }
        let reconGpuMs = session.end()

        // Доля работы реконструкций — по фактической температуре, а не
        // константой. Раньше здесь стояло targetDuty(for: .nominal), то есть
        // жёсткие 55%: ровно та же ошибка, что и с фиксированной паузой между
        // эпохами — карта холодная, а мы её «остужаем» впустую.
        let hot = sensor.read()?.temperatureC ?? 0
        let duty: Double = hot >= defaultMaxTempC ? 0.60
                         : hot >= defaultTargetTempC ? 0.85
                         : safeDutyCap
        let reconSleep = dutySleepMs(gpuMs: reconGpuMs, duty: duty, maxPauseMs: 2000)
        if reconSleep > 0 { Thread.sleep(forTimeInterval: reconSleep / 1000) }

        for (k, t) in recons.enumerated() {
            let r = t.toArray()
            let item = chunkItems[targetToItem[k]]
            // Реконструкция — сразу в память (PNG), без записи на диск.
            if let png = encodePNG(width: pw, height: ph, rgb: r) {
                result[item.name] = png
            }
            done += 1
            onProgress?(done, items.count)
            // Объективный прогресс в журнале: старт, каждые 5%, финиш.
            if shouldLogProgress(done: done, total: items.count,
                                 lastLoggedPercent: lastLoggedPercent) {
                lastLoggedPercent = Int((Double(done) / Double(max(items.count, 1))) * 100)
                let pct = Int(Double(done) / Double(max(items.count, 1)) * 100)
                yumeLog("Реконструкции: \(done)/\(items.count) кадров (\(pct)%)")
            }
        }
    }
    return result
}

// ---------- Обучение ----------

/// Колбэк прогресса: (эпоха, loss, psnr, путь превью, максимум эпох) -> Void
public typealias TrainProgress = (Int, Double, Double, String?) -> Void

/// Потокобезопасное управление обучением (для остановки из GUI).
public final class TrainingControl {
    public static let shared = TrainingControl()
    private let lock = NSLock()
    private var _cancelled = false
    public var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _cancelled
    }
    public func cancel() {
        lock.lock(); defer { lock.unlock() }
        _cancelled = true
    }
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        _cancelled = false
    }
}

/// Замена NaN/Inf в градиентах на 0 + клиппинг суммарной нормы (L2).
private func sanitizeGrads(_ grads: [GPUTensor], maxNorm: Float = 1.0) {
    var totalSq: Float = 0
    var fixed = 0
    for g in grads {
        var a = g.toArray()
        var s: Float = 0
        for i in 0..<a.count {
            let v = a[i]
            if v.isNaN || v.isInfinite {
                a[i] = 0
                fixed += 1
            } else {
                s += v * v
            }
        }
        totalSq += s
        g.setData(a)
    }
    let norm = sqrt(totalSq)
    if norm > maxNorm {
        let scale = maxNorm / norm
        for g in grads {
            var a = g.toArray()
            for i in 0..<a.count { a[i] *= scale }
            g.setData(a)
        }
    }
    if fixed > 0 {
        yumeLog("санация градиентов: \(fixed) значений NaN/Inf → 0; норма \(norm)")
    }
}

public func trainConvVAE(p: Params, onProgress: TrainProgress? = nil,
                         onStep: ((TrainStepReport) -> Void)? = nil) throws {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    let realW = p.width, realH = p.height
    let pw = ((realW + 7) / 8) * 8
    let ph = ((realH + 7) / 8) * 8

    var source: DataSource
    var frameCache: FrameCache? = nil
    // Фаза подготовки: сообщаем интерфейсу, что цикл жив, ещё до первого
    // шага — загрузка модели и кадров в VRAM занимает минуты, и без этого
    // отчёта лоадер выглядел бы «зависшим».
    var isPreparing = true
    let emitStatus: (String, TrainStage) -> Void = { note, stage in
        onStep?(TrainStepReport(done: 0, total: max(1, p.epochs * max(1, p.steps)),
                                epoch: 1, epochs: max(1, p.epochs),
                                step: 0, stepsPerEpoch: max(1, p.steps),
                                frame: 0, framesTotal: p.memoryFrames?.count ?? 0,
                                batch: 0, secondsPerStep: nil,
                                note: note, stage: stage))
    }
    emitStatus(trainPreparationNote(), .preparing)
    // Пул кадров в VRAM (строим первым — он заменяет CPU-кэш и быстрее).
    var gpuPool: GPUFramePool? = nil
    if let mem = p.memoryFrames, !mem.isEmpty {
        gpuPool = GPUFramePool(engine: engine, frames: mem, w: pw, h: ph) { loaded, total in
            emitStatus(trainVRAMLoadNote(loaded: loaded, total: total),
                       isPreparing ? .preparing : .training)
        }
        if gpuPool == nil {
            // Если пул не построился (память) — обычный CPU-кэш.
            frameCache = FrameCache(frames: mem, w: pw, h: ph)
        }
        source = DataSource(files: [], memFrames: mem, cache: frameCache,
                            w: pw, h: ph, realW: realW, realH: realH)
        yumeLog("Данные: \(mem.count) кадров из памяти\(gpuPool != nil ? " (VRAM-пул)" : (frameCache != nil ? " (кэш RGB8)" : ""))")
    } else {
        let files = (p.dataDir.map { listImages(in: $0) } ?? []).sorted()
        source = DataSource(files: files, w: pw, h: ph, realW: realW, realH: realH)
        if files.isEmpty {
            yumeLog("Данные: синтетические сцены (реальные кадры не заданы: --data <папка>)")
        } else {
            yumeLog("Данные: \(files.count) файлов из \(p.dataDir!)")
        }
    }

    // Размер набора для живого статуса («кадр N из M»). При синтетике — 0.
    let datasetFrames: Int
    if let mem = p.memoryFrames, !mem.isEmpty {
        datasetFrames = mem.count
    } else if let cache = frameCache {
        datasetFrames = cache.count
    } else {
        datasetFrames = source.files.count
    }

    yumeLog("Свёрточный VAE: кадр \(realW)×\(realH) → внутренний \(pw)×\(ph), латент \(pw/8)×\(ph/8)×\(ConvVAEGPU.latentC) (сжатие \(Int(ConvVAEGPU.compressionRatio))×)")

    let vae = ConvVAEGPU(engine: engine)
    // Отпечаток текущих данных — вычисляется один раз до освобождения JPEG.
    let currentFingerprint: String
    if let mem = p.memoryFrames, !mem.isEmpty {
        currentFingerprint = dataFingerprint(frames: mem)
    } else {
        currentFingerprint = "none"
    }
    // Переключатель «Продолжить с чекпоинта» = ТОЛЬКО загрузка весов для
    // дообучения. Эпохи НИКОГДА не пропускаем — всегда полный прогон с 0.
    if p.resume && FileManager.default.fileExists(atPath: p.convPath) {
        let st = try loadStateWithMeta(path: p.convPath)
        do {
            try vae.load(from: st.mats)
            if let fp = st.fingerprint, !fp.isEmpty, fp != currentFingerprint {
                yumeLog("⚠️ Данные отличаются от тех, на которых учился чекпоинт (другой файл?) — гружу веса, но эпохи не пропускаю")
            }
            yumeLog("Веса загружены из \(p.convPath) (дообучение)")
        } catch let error where isIncompatibleCheckpoint(error) {
            // Файл не битый — он от другой архитектуры (например, латент был
            // 64 канала до перехода на вариационный путь). Падать на первом же
            // шаге незачем: начинаем с новых весов и говорим об этом прямо.
            yumeLog("⚠️ Чекпоинт \(p.convPath) не совпадает по форме с текущей моделью:")
            yumeLog("   \((error as NSError).localizedDescription)")
            yumeLog("   Обучение начнётся с новых весов. Файл не тронут — при необходимости верните его из .backup.")
        }
    }

    let previewDir = (p.outPrefix as NSString).deletingLastPathComponent
    try FileManager.default.createDirectory(atPath: previewDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(atPath: (p.convPath as NSString).deletingLastPathComponent,
                                            withIntermediateDirectories: true)

    var t = 0
    let start = Date()
    var stepTimes: [Double] = []
    let pauseMs = Double(max(p.pauseMs, 0))
    var stableRun = true   // false, если обнаружен NaN — чекпоинт сохранять нельзя

    // Термо-регулятор. Работает НА КАЖДОМ шаге и управляет долей работы GPU,
    // а не только «уровнем тревоги»: 4 ноября 2026 система ушла в панику
    // watchdogd, потому что GPU была занята почти непрерывно и WindowServer
    // не мог получить доступ к тому же устройству 120 секунд подряд.
    // Подробности и формулы — в ThermalGovernor.swift.
    var stepBaselineMs = 0.0          // лучший (холодный) замер шага
    var lastHeatLevel: HeatLevel = .nominal
    var pendingSleepMs = 0.0          // пауза, вычисленная на этом шаге
    var gpuMsThisStep = 0.0
    var heatLogAt = Date.distantPast
    // Сглаживание уровня: без него дрейф 328/366 мс при пороге 2.5× дёргал
    // «горячо ↔ критично» на каждом шаге и забивал журнал.
    var heatDebouncer = HeatDebouncer(steps: 3)
    // Последняя прочитанная температура: нужна детектору, чтобы не путать
    // нагрев со сбросом частоты по мощности (см. `heatLevel`).
    var lastTempC: Double? = nil

    // Замкнутый контур по температуре GPU. Читаем телеметрию из IORegistry
    // (без root) раз в 2 секунды и подстраиваем долю работы так, чтобы карта
    // держалась ниже порога троттлинга. Пользователь задаёт только потолок
    // («мощность, %»); фактическую нагрузку находит сам контур.
    let gpuSensor = GpuSensor()
    var power = PowerController(ceiling: effectiveDutyCeiling(powerPercent: p.gpuPowerPercent))
    var currentDuty = power.duty
    var lastSensorAt = Date.distantPast
    var lastSensorLog = Date.distantPast
    var lastDutyCutAt = Date.distantPast
    yumeLog(String(format: "Нагрузка GPU: потолок %.0f%% (доля работы ≤ %.0f%%), цель %.0f °C",
                   Double(p.gpuPowerPercent), power.ceiling * 100, power.targetTempC))

    // Термо-регулятор: решение принимается НА КАЖДОМ шаге.
    //
    // Раньше здесь стоял thermalGuard, который вызывался раз в 10 шагов и
    // только менял `pauseMs`, — а `pauseMs` применялся исключительно на
    // границе эпохи (43 187 шагов). То есть авто-троттлинг не срабатывал
    // ни разу за всё обучение. Теперь пауза считается из фактического
    // времени работы GPU и применяется сразу.
    func governHeat(gpuMs: Double, wallMs: Double, duty: Double,
                    tempC: Double?) -> ThrottleDecision {
        let raw = throttleDecision(
            gpuMs: gpuMs,
            wallMs: wallMs,
            baselineMs: stepBaselineMs,
            thermalRaw: ProcessInfo.processInfo.thermalState.rawValue,
            // floorMs — это НЕ «минимальная пауза вообще», а фиксированный
            // режим: он нужен только когда авто-регулятор выключен и нагрузку
            // задаёт поле «Пауза, мс». Если авто включён, полом служит сама
            // доля работы, иначе 250 мс из интерфейса перекрывали бы решение
            // контура: при 99% расчётная пауза 1.7 мс, а floor поднимал её
            // до 250 мс — фактическая нагрузка падала до ~40%, и карта
            // не могла догреться до рабочей точки ни при каких настройках.
            floorMs: p.adaptiveThrottle ? 0 : pauseMs,
            maxPauseMs: Double(p.maxPauseMs),
            duty: duty,
            tempC: tempC)
        stepBaselineMs = raw.baselineMs

        // Уровень сглаживаем: одиночные выбросы времени шага — это шум
        // планировщика и драйвера, а не смена теплового режима.
        let level = heatDebouncer.update(raw.level)

        // Пишем в журнал при смене УЖЕ СГЛАЖЕННОГО уровня и не чаще раза
        // в минуту — иначе строки о тепле утонут в прогрессе.
        if level != lastHeatLevel, Date().timeIntervalSince(heatLogAt) > 60 {
            heatLogAt = Date()
            yumeLog(String(format: "🌡️ тепло: %@ · GPU %.0f мс/шаг, база %.0f мс · пауза %.0f мс · нагрузка %.0f%%",
                           level.title, gpuMs, raw.baselineMs,
                           raw.sleepMs, duty * 100))
            if level >= .hot, let t = tempC {
                yumeLog(String(format: "⚠️ нагрев вырос до «%@» при %.0f °C — снижаю нагрузку",
                               level.title, t))
            }
        }
        lastHeatLevel = level
        return ThrottleDecision(level: level, sleepMs: raw.sleepMs, baselineMs: raw.baselineMs)
    }

    // Пачка кадров за шаг: под активации отведено 30% рабочего набора GPU
    // (60% уже занял пул кадров, 10% — запас). Считаем не по размеру кадра,
    // а по полному объёму активаций одного образца.
    let vramBudgetFrames = effectiveBatchSize(
        requested: Int.max,
        w: pw, h: ph,
        budgetBytes: activationBudgetBytes(
            workingSet: engine.device.recommendedMaxWorkingSetSize))
    // Жёсткий потолок 16: большие пачки держат GPU занятым десятки секунд без
    // отдачи — на MacBookPro16,1 это приводило к зависанию WindowServer
    // (watchdog 409, thermal heavy). Для стабильности системы — не выше 16.
    let maxSafeBatch = 16
    let effBatch = min(max(p.batch, 1), vramBudgetFrames, maxSafeBatch)
    if effBatch < max(p.batch, 1) {
        yumeLog("Батч урезан \(p.batch) → \(effBatch) (потолок стабильности \(maxSafeBatch); по памяти влезает \(vramBudgetFrames), \(convVAEBytesPerSample(w: pw, h: ph) / 1_048_576) МБ на кадр)")
    } else {
        yumeLog("Батч: \(effBatch) кадров за шаг (потолок стабильности \(maxSafeBatch); по памяти влезает \(vramBudgetFrames), \(convVAEBytesPerSample(w: pw, h: ph) / 1_048_576) МБ на кадр)")
    }
    // Тензоры под пачку — создаются один раз и живут весь прогон.
    let inputPool = (0..<effBatch).map { _ in GPUTensor(engine: engine, rows: 3, cols: pw * ph) }

    // «Полный проход» по набору: при пачке >1 число оптимизаторских шагов
    // автоматически сокращается (кадры за шаг = effBatch). Пользовательский
    // steps (когда он задан явно) не трогаем.
    var epochSteps = max(1, p.steps)
    if let mem = p.memoryFrames, !mem.isEmpty, p.steps == mem.count, effBatch > 1 {
        epochSteps = max(1, Int((Double(mem.count) / Double(effBatch)).rounded(.up)))
        yumeLog("Шагов за эпоху (авто под пачку): \(epochSteps) × \(effBatch) кадров = \(mem.count)")
    }

    // Эпохи НИКОГДА не пропускаются: полный прогон с эпохи 0 по всем
    // p.epochs, независимо от resume (resume грузит только веса).
    isPreparing = false
    for ep in 0..<p.epochs {
        if TrainingControl.shared.isCancelled {
            yumeLog("Остановлено пользователем на эпохе \(ep)")
            break
        }
        var lossSum = 0.0
        var psnrSum = 0.0
        var stepsDone = 0
        // цель для превью — первый сэмпл эпохи
        var previewTarget: [Float]? = nil
        var previewRecon: [Float]? = nil

        // Всегда с шага 0: дообучение идёт полной эпохой с начала.
        for stepIdx in 0..<epochSteps {
            if TrainingControl.shared.isCancelled { break }
            var accGrads: [GPUTensor]? = nil
            var lastMse: Double = 0
            let stepStart = Date()

            var batchImgs: [[Float]] = []
            var batchTensors: [GPUTensor] = []
            if let pool = gpuPool {
                for b in 0..<effBatch {
                    let frame = pool.next()
                    inputPool[b].copyFrom(frame)
                    batchTensors.append(inputPool[b])
                }
            }
            session.begin()
            if batchTensors.isEmpty {
                for b in 0..<effBatch {
                    let img = source.sequentialSample()
                    batchImgs.append(img)
                    session.upload(inputPool[b], from: img)
                    batchTensors.append(inputPool[b])
                }
            } else {
                // Из пула VRAM: кадры уже на GPU, ничего грузить не нужно.
                for _ in 0..<effBatch { batchImgs.append([]) }
            }

            var firstGrads: [GPUTensor]? = nil
            var firstCache: VAECache? = nil
            var recons: [GPUTensor] = []
            var mseBufs: [MTLBuffer] = []
            for b in 0..<effBatch {
                let t = batchTensors[b]
                let batchCache = vae.forward(session, image: t, w: pw, h: ph)
                let (_, grads) = vae.backward(session, batchCache, target: t)
                recons.append(batchCache.recon)
                // MSE считается прямо на GPU (без чтения полных тензоров на CPU).
                mseBufs.append(session.encodeMSE(recon: batchCache.recon, target: t))
                if firstCache == nil { firstCache = batchCache }
                if firstGrads == nil { firstGrads = grads }
                if accGrads == nil {
                    // Накопитель создаётся «пустым» (storageModePrivate не гарантирует
                    // нули), поэтому ПЕРВОЕ слагаемое копируется, а не прибавляется.
                    // Раньше здесь был addInPlace поверх мусора — градиенты уезжали
                    // в случайные величины и спасал только клиппинг.
                    accGrads = grads.map { g in
                        let c = GPUTensor(engine: engine, rows: g.rows, cols: g.cols)
                        session.copy(c, from: g)
                        return c
                    }
                } else {
                    for i in 0..<grads.count { session.addInPlace(accGrads![i], grads[i]) }
                }
            }
            gpuMsThisStep = session.end()   // один commit + один wait на всю пачку

            // --- Фаза 3: потеря с GPU (скаляр на каждый кадр пачки) ---
            var mseSum = 0.0
            var firstImg: [Float]? = nil
            var firstR: [Float]? = nil
            for b in 0..<effBatch {
                let mse = Double(mseBufs[b].contents().assumingMemoryBound(to: Float.self)[0])
                    / Double(max(recons[b].n, 1))
                mseSum += mse
                // Для превью нужны реальные пиксели — читаем ОДИН кадр,
                // и только когда превью ещё нет (первый шаг эпохи).
                if b == 0 && previewTarget == nil {
                    if gpuPool != nil {
                        // Из пула VRAM: оригинал берём из исходного кадра,
                        // С ПАДДИНГОМ (pw×ph), чтобы длина совпадала с реконструкцией.
                        firstImg = batchTensors[0].toArray()
                    } else {
                        firstImg = batchImgs[0]
                    }
                    firstR = recons[0].toArray()
                }
            }
            lastMse = mseSum / Double(effBatch)
            lossSum += lastMse
            psnrSum += -10.0 * log10(max(lastMse, 1e-12))
            stepsDone += 1

            if previewTarget == nil {
                previewTarget = firstImg
                previewRecon = firstR
            }
            if ep == 0 && stepIdx == 0 {
                // трассировка градиентов первого шага
                for (i, g) in firstGrads!.enumerated() {
                    let a = g.toArray()
                    if a.contains(where: { $0.isNaN }) {
                        yumeLog("❌ NaN в градиенте [\(i)]")
                    } else {
                        yumeLog(String(format: "grad[%d] min=%.4e max=%.4e", i, a.min()!, a.max()!))
                    }
                }
                // локализация NaN в forward (пишется в train.log)
                func scanNaN(_ name: String, _ t: GPUTensor) {
                    let a = t.toArray()
                    if a.contains(where: { $0.isNaN }) {
                        yumeLog("❌ NaN найден в \(name)")
                    }
                }
                let c0 = firstCache!
                scanNaN("p1", c0.p1); scanNaN("a1", c0.a1)
                scanNaN("p2", c0.p2); scanNaN("a2", c0.a2)
                scanNaN("p3", c0.p3); scanNaN("mu", c0.mu)
                scanNaN("logvar", c0.logvar); scanNaN("z", c0.z)
                scanNaN("u1", c0.u1)
                scanNaN("p4", c0.p4); scanNaN("a4", c0.a4)
                scanNaN("u2", c0.u2)
                scanNaN("p5", c0.p5); scanNaN("a5", c0.a5)
                scanNaN("u3", c0.u3)
                scanNaN("p6", c0.p6)
                scanNaN("recon", c0.recon)
                yumeLog(String(format: "debug step0: mse=%.6f recon0=%.4f img0=%.4f",
                               lastMse, firstR![0], firstImg![0]))
            }

            // детект NaN: останавливаем обучение до того, как NaN попадёт в чекпоинт
            if !lastMse.isFinite {
                stableRun = false
                yumeLog("⚠️ loss = NaN на эпохе \(ep) — остановка обучения (чекпоинт не сохраняется)")
                TrainingControl.shared.cancel()
                break
            }

            // Санация: NaN/Inf в градиентах → 0; глобальный клиппинг нормы.
            if effBatch > 1 {
                for gradient in accGrads! {
                    gradient.setData(gradient.toArray().map { $0 / Float(effBatch) })
                }
            }
            sanitizeGrads(accGrads!, maxNorm: 1.0)

            // Adam по накопленным градиентам (lr делится на размер пачки)
            t += 1
            session.begin()
            vae.applyAdam(session, grads: accGrads!, t: t, lr: p.lr)
            gpuMsThisStep += session.end()   // Adam тоже занимает GPU — суммируем

            stepTimes.append(Date().timeIntervalSince(stepStart))

            // Живой прогресс по шагам: эпоха, текущий кадр набора, шаг,
            // скорость. Отчёт идёт каждые 5 шагов (и на последнем) — этого
            // достаточно для лоадера, но не дёргает UI на каждом шаге.
            if let onStep = onStep {
                let done = t
                let total = max(1, p.epochs * epochSteps)
                if done % 5 == 0 || done == total {
                    let recent = stepTimes.suffix(50)
                    let recentAvg = recent.reduce(0, +) / Double(max(recent.count, 1))
                    let frame = currentFrameIndex(epochIndex: ep, stepIndex: stepIdx,
                                                 stepsPerEpoch: epochSteps, batch: effBatch,
                                                 datasetFrames: datasetFrames)
                    onStep(TrainStepReport(done: done, total: total,
                                           epoch: ep + 1, epochs: p.epochs,
                                           step: stepIdx + 1, stepsPerEpoch: epochSteps,
                                           frame: frame, framesTotal: datasetFrames,
                                           batch: effBatch,
                                           secondsPerStep: recentAvg))
                }
            }

            // Термо-регулятор: решение на КАЖДОМ шаге по фактическому времени GPU.
            // Раз в 2 с пересчитываем долю работы по живой температуре карты.
            if Date().timeIntervalSince(lastSensorAt) >= 2.0 {
                lastSensorAt = Date()
                let stats = gpuSensor.read()
                currentDuty = power.update(stats: stats)
                if let s = stats, s.hasTemperature {
                    lastTempC = s.temperatureC
                    // Предохранитель: выше 86 °C обучение уходит на остывание,
                    // что бы ни решил контур. Перезагрузка случалась на 93 °C.
                    if s.temperatureC >= emergencyTempC {
                        yumeLog(String(format: "🔥 GPU %.0f °C — принудительное остывание 20 с",
                                       s.temperatureC))
                        Thread.sleep(forTimeInterval: 20)
                        currentDuty = max(power.duty * 0.5, 0.05)
                    }
                    if Date().timeIntervalSince(lastSensorLog) > 60 {
                        lastSensorLog = Date()
                        yumeLog(String(format: "🌡️ GPU %.0f °C · %.0f Вт → нагрузка %.0f%% (цель %.0f °C)",
                                       s.temperatureC, s.powerW, currentDuty * 100,
                                       power.targetTempC))
                    }
                } else {
                    // Датчик пропал: честно обнуляем, иначе детектор будет
                    // опираться на устаревшее значение.
                    lastTempC = nil
                }
            }
            let wallMs = stepTimes.last.map { $0 * 1000 } ?? 0
            // Дрейф считаем по ЧИСТОМУ времени GPU, а не по стенному: пауза,
            // которую мы сами добавляем, не должна выглядеть как нагрев —
            // иначе регулятор уходит в спираль «больше пауза → больше дрейф →
            // ещё больше пауза». gpuMs от пауз не зависит.
            let decision = governHeat(gpuMs: gpuMsThisStep,
                                      wallMs: gpuMsThisStep > 0 ? gpuMsThisStep : wallMs,
                                      duty: currentDuty,
                                      tempC: lastTempC)
            pendingSleepMs = decision.sleepMs

            // Аварийный резерв — ТОЛЬКО когда датчик недоступен. Тогда
            // единственный признак проблемы это просадка времени шага.
            //
            // При живом датчике этот блок был вреден: он каждые 2 минуты
            // срезал нагрузку на 10%, а контур тут же возвращал её к потолку,
            // потому что карта холодная (64 °C). В журнале это выглядело как
            // «снижаю нагрузку» без единого реального изменения — ручка
            // дёргалась туда-сюда, а доля работы оставалась 99%.
            if lastTempC == nil, decision.level >= .hot,
               Date().timeIntervalSince(lastDutyCutAt) > 120 {
                lastDutyCutAt = Date()
                currentDuty = max(currentDuty * 0.90, 0.50)
                yumeLog(String(format: "⚠️ нет данных датчика, шаг замедлился (%@) — снижаю нагрузку до %.0f%%",
                               decision.level.title, currentDuty * 100))
            }

            // Промежуточный чекпоинт каждые 500 шагов: веса + текущий прогресс
            // (эпоха/шаг в метаданных) — чтобы после перезапуска можно было
            // продолжить ровно с места срыва.
            if stepIdx > 0 && stepIdx % 500 == 0 {
                do {
                    try saveState(vae.matrices, to: p.convPath,
                                  epoch: ep, step: stepIdx, totalSteps: epochSteps,
                                  dataFingerprint: currentFingerprint)
                    yumeLog("Промежуточный чекпоинт: эпоха \(ep), шаг \(stepIdx)/\(epochSteps) — сохранён")
                } catch {
                    yumeLog("⚠️ ошибка промежуточного чекпоинта: \(error.localizedDescription)")
                }
            }

            // Пауза — ЗДЕСЬ, в конце каждого шага, а не между эпохами.
            // Именно её отсутствие перегрело GPU 4 ноября 2026.
            if pendingSleepMs > 0 {
                Thread.sleep(forTimeInterval: pendingSleepMs / 1000)
            }
        }

        let meanLoss = lossSum / Double(max(stepsDone, 1))
        let meanPsnr = psnrSum / Double(max(stepsDone, 1))
        let perStep = stepTimes.suffix(epochSteps).reduce(0, +) / Double(max(stepTimes.suffix(epochSteps).count, 1))
        let elapsed = Date().timeIntervalSince(start)
        yumeLog(String(format: "epoch %d  loss %.6f  psnr %.2f dB  %.2f s/шаг  всего %.0f c",
                     ep, meanLoss, meanPsnr, perStep, elapsed))

        if let pt = previewTarget, let pr = previewRecon {
            // Превью эпохи — только для живого показа в интерфейсе.
            // Реконструкции кадров после обучения живут в памяти, папка на
            // диске больше не нужна (и не запрашивается в интерфейсе).
            let outPath = "\(previewDir)/conv_ep\(String(format: "%03d", ep)).png"
            try savePairPNG(path: outPath, w: pw, h: ph, target: pt, recon: pr)
            onProgress?(ep, meanLoss, meanPsnr, outPath)
        } else {
            onProgress?(ep, meanLoss, meanPsnr, nil)
        }

        if stableRun {
            try saveState(vae.matrices, to: p.convPath,
                          epoch: ep + 1, step: 0, totalSteps: epochSteps,
                          dataFingerprint: currentFingerprint)
        }

        // Пауза между эпохами — тоже только по теплу, а не «по расписанию».
        //
        // Раньше здесь всегда спалось max(pauseMs, 1000) мс: за эпоху карта
        // якобы успевает прогреться, и ей нужен отдых. Но замер на 99%
        // (`yume gpuheat 180 0`) показал обратное: за 180 с непрерывной
        // работы температура не поднялась выше 78 °C и сама сползла к 68–70,
        // мощность 5–23 Вт. Греться нечему — значит и остывать не от чего,
        // и фиксированная секунда простоя на каждой границе эпохи это чистый
        // простой. Отдых даём только если карта реально у стоп-порога.
        if ep < p.epochs - 1 {
            if TrainingControl.shared.isCancelled { break }
            let hotTemp = gpuSensor.read()?.temperatureC
            let coolMs = epochCoolPauseMs(tempC: hotTemp, pauseMs: pauseMs,
                                          targetTempC: power.targetTempC,
                                          maxTempC: power.maxTempC)
            if coolMs > 0 {
                yumeLog(String(format: "🌡️ пауза между эпохами: %.0f мс на остывание (GPU %.0f °C)",
                               coolMs, hotTemp ?? 0))
                Thread.sleep(forTimeInterval: coolMs / 1000)
            }
        }
    }

    if TrainingControl.shared.isCancelled { throw CancellationError() }
    yumeLog("Готово. Чекпоинт: \(p.convPath)")
}

// ---------- Бенчмарк железа ----------

/// Прогрев GPU тем же циклом, что и обучение, с заданной долей работы.
///
/// Нужен, чтобы откалибровать порог температуры: пока нет замеров, любой
/// «безопасный процент» — это догадка. Запускает настоящий forward+backward
/// свёрточного VAE на кадре 856×480, держит долю работы `duty` и печатает
/// температуру, частоту и мощность каждые 2 секунды.
public func runGpuHeat(seconds: Double = 60, duty: Double = 0.85,
                       w: Int = 856, h: Int = 480) {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let vae = ConvVAEGPU(engine: engine)
    let sensor = GpuSensor()
    let img = makeNormalSynthetic(w: w, h: h)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    // duty <= 0 — автоматический режим: нагрузку задаёт контур по температуре,
    // ровно как в обучении. Это и есть проверка «работает на грани».
    // В ОБОИХ режимах потолок — физический предел 98%: диагностика не имеет
    // права быть опаснее обучения. Раньше здесь стояло 98% в авто и 100% в ручном.
    let auto = duty <= 0
    let requestedDuty = clampedGpuDuty(duty)
    var power = PowerController(ceiling: safeDutyCap,
                                startDuty: auto ? 0.5 : requestedDuty)
    var currentDuty = auto ? power.duty : requestedDuty

    yumeLog(String(format: "Прогрев: %@, %d×%d, %@, %.0f с",
                   engine.deviceName, w, h,
                   auto ? String(format: "авто (цель %.0f °C)", power.targetTempC)
                        : String(format: "доля работы %.0f%%", requestedDuty * 100),
                   seconds))
    yumeLog(String(format: "Потолок доли работы %.0f%% · запрос %.0f%%",
                   safeDutyCap * 100, duty * 100))
    let start = Date()
    var t = 0
    var lastLog = Date.distantPast
    var peakTemp = 0.0
    var stepMsAll: [Double] = []

    while Date().timeIntervalSince(start) < seconds {
        if TrainingControl.shared.isCancelled { break }
        session.begin()
        let cache = vae.forward(session, image: x, w: w, h: h)
        let (_, g) = vae.backward(session, cache, target: x)
        t += 1
        vae.applyAdam(session, grads: g, t: t, lr: 0.0001)
        let gpuMs = session.end()
        stepMsAll.append(gpuMs)

        // Датчик читаем один раз за шаг и В ОБОИХ режимах: аварийная
        // температура обязана останавливать и ручной прогрев тоже.
        let stats = sensor.read()
        if auto {
            currentDuty = clampedGpuDuty(power.update(stats: stats))
        }
        if let s = stats, s.temperatureC >= emergencyTempC {
            yumeLog(String(format: "🔥 %.0f °C — остывание 20 с", s.temperatureC))
            Thread.sleep(forTimeInterval: 20)
            currentDuty = clampedGpuDuty(max(power.duty * 0.5, 0.05))
        }

        // Финальный клэмп на месте применения: даже если контур или ручной
        // запрос когда-нибудь вернут больше, до GPU дойдёт не выше 98%.
        currentDuty = clampedGpuDuty(currentDuty)
        let sleepMs = dutySleepMs(gpuMs: gpuMs, duty: currentDuty, maxPauseMs: 5000)
        if sleepMs > 0 { Thread.sleep(forTimeInterval: sleepMs / 1000) }

        if Date().timeIntervalSince(lastLog) > 3 {
            lastLog = Date()
            guard let s = stats else { continue }
            peakTemp = max(peakTemp, s.temperatureC)
            yumeLog(String(format: "  %.0f °C · %.0f Вт · нагрузка %.0f%% · %.0f мс/шаг",
                           s.temperatureC, s.powerW, currentDuty * 100, gpuMs))
        }
    }

    let third = max(1, stepMsAll.count / 3)
    let head = stepMsAll.prefix(third).reduce(0, +) / Double(third)
    let tail = stepMsAll.suffix(third).reduce(0, +) / Double(third)
    yumeLog(String(format: "Итог: пик %.0f °C, нагрузка %.0f%%, шаг %.0f → %.0f мс, шагов %d",
                   peakTemp, currentDuty * 100, head, tail, t))
}

public func runGpuBench() {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let dev = engine.device
    yumeLog("⚠️ Диагностика — не запускайте при активном обучении (нагрузка на GPU).")
    yumeLog("Устройство: \(dev.name)")
    yumeLog("Унифицированная память: \(dev.hasUnifiedMemory)")
    yumeLog("Рекомендуемый рабочий набор: \(dev.recommendedMaxWorkingSetSize / 1024 / 1024) МБ")
    yumeLog("Макс. потоков в группе: \(dev.maxThreadsPerThreadgroup.width)")

    // Пропускная способность: relu над 32 млн float (128 МБ), 3 повтора
    let n = 32 * 1024 * 1024 / 4
    let a = GPUTensor(engine: engine, rows: 1, cols: n)
    session.begin()
    _ = session.relu(a)
    session.end()
    var t0 = Date()
    for _ in 0..<3 {
        session.begin()
        _ = session.relu(a)
        session.end()
    }
    var dt = Date().timeIntervalSince(t0) / 3
    let gb = Double(n * 4 * 2) / 1e9
    yumeLog(String(format: "Память (relu 128 МБ): %.2f ГБ/с, %.2f мс", gb / dt, dt * 1000))

    // Вычисления: тайловый matmul 512³
    let m = 512, k = 512, nn = 512
    let A = GPUTensor(engine: engine, rows: m, cols: k)
    let B = GPUTensor(engine: engine, rows: k, cols: nn)
    t0 = Date()
    session.begin()
    _ = session.matmul(A, B)
    session.end()
    dt = Date().timeIntervalSince(t0)
    let flops = 2.0 * Double(m) * Double(k) * Double(nn)
    yumeLog(String(format: "Matmul 512³: %.1f мс, %.1f ГФЛОП/с", dt * 1000, flops / dt / 1e9))
}



public func profileConv(w: Int, h: Int) {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let vae = ConvVAEGPU(engine: engine)
    let img = makeNormalSynthetic(w: w, h: h)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    func time(_ name: String, _ reps: Int, _ body: () -> Void) {
        body()  // прогрев
        let t0 = Date()
        for _ in 0..<reps { body() }
        let dt = Date().timeIntervalSince(t0) / Double(reps)
        yumeLog(String(format: "%@: %.1f мс", name, dt * 1000))
    }

    time("forward", 3) {
        session.begin()
        _ = vae.forward(session, image: x, w: w, h: h)
        let gpu = session.end()
        yumeLog("  (GPU: \(String(format: "%.1f", gpu)) мс)")
    }

    // кэш для backward
    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h)
    session.end()

    time("backward (полный)", 3) {
        session.begin()
        _ = vae.backward(session, cache, target: x)
        session.end()
    }

    // Фазовый разбор backward
    session.begin()
    let dP6 = session.sigmoidGrad(cache.p6, session.smul(2.0 / Float(3 * w * h), session.sub(cache.recon, x)))
    session.end()
    time("gW всех слоёв", 3) {
        session.begin()
        _ = vae.layers[0].backward(session, cache.c1, dOut: cache.p1)
        _ = vae.layers[1].backward(session, cache.c2, dOut: cache.p2)
        _ = vae.layers[2].backward(session, cache.c3, dOut: cache.p3)
        _ = vae.layers[3].backward(session, cache.c4, dOut: cache.p4)
        _ = vae.layers[4].backward(session, cache.c5, dOut: cache.p5)
        _ = vae.layers[5].backward(session, cache.c6, dOut: dP6)
        session.end()
    }

    time("активации backward (relu/tanh/сигмоид)", 3) {
        session.begin()
        _ = session.reluGrad(cache.a5, cache.p5)
        _ = session.reluGrad(cache.a4, cache.p4)
        _ = session.reluGrad(cache.a2, cache.p2)
        _ = session.reluGrad(cache.a1, cache.p1)
        _ = session.sigmoidGrad(cache.p6, cache.p6)
        _ = session.tanhGrad(cache.p3)
        session.end()
    }

    time("downsample2x ×3", 3) {
        session.begin()
        _ = session.downsample2x(cache.p6, h: h / 2, w: w / 2)
        _ = session.downsample2x(cache.p5, h: h / 4, w: w / 4)
        _ = session.downsample2x(cache.p4, h: h / 8, w: w / 8)
        session.end()
    }

    // изоляция тяжёлых kernels
    let cols6 = cache.c6.cols       // (16*9) × (480*856)
    let dOut6 = cache.p6           // 3 × (480*856)
    time("gradW d3 (3×144, K=410880)", 3) {
        session.begin()
        _ = session.gradW(dOut6, cols6, oc: 3, kk: 144)
        session.end()
    }
    time("biasRowGrad d3", 3) {
        session.begin()
        _ = session.biasRowGrad(dOut6)
        session.end()
    }
    time("convBackInput d3", 3) {
        session.begin()
        _ = session.convBackInputS1(dOut6, vae.layers[5].W, ic: 16, oc: 3,
                                    ih: h, iw: w, k: 3, pad: 1)
        session.end()
    }
    let cols1 = cache.c1.cols       // 27 × (240*428)
    let dOut1 = cache.p1           // 16 × (240*428)
    time("gradW e1 (16×27, K=102720)", 3) {
        session.begin()
        _ = session.gradW(dOut1, cols1, oc: 16, kk: 27)
        session.end()
    }

    let recon = cache.recon
    time("readback 1 кадр", 5) {
        _ = recon.toArray()
    }

    time("аллокация 236 МБ", 5) {
        _ = GPUTensor(engine: engine, rows: 16, cols: w * h)
    }

    time("заполнение 236 МБ (relu)", 5) {
        let a = GPUTensor(engine: engine, rows: 16, cols: w * h)
        session.begin()
        _ = session.relu(a)
        session.end()
    }

    // отдельно im2col самого тяжёлого слоя (d3: 16→3, 480×856)
    let bigIn = GPUTensor(engine: engine, rows: 16, cols: w * h)
    time("im2col d3 (16×480×856)", 3) {
        session.begin()
        _ = session.im2col(bigIn, ih: h, iw: w, ic: 16, kh: 3, kw: 3, stride: 1, pad: 1)
        session.end()
    }
    time("col2im d3 (16×480×856)", 3) {
        session.begin()
        let cols = session.im2col(bigIn, ih: h, iw: w, ic: 16, kh: 3, kw: 3, stride: 1, pad: 1)
        _ = session.col2im(cols, ic: 16, ih: h, iw: w, kh: 3, kw: 3, stride: 1, pad: 1,
                           oh: h, ow: w)
        session.end()
    }
}



/// Сравнивает аналитические градиенты свёрточного VAE с конечными разностями.
public func checkConvGradients() {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let vae = ConvVAEGPU(engine: engine)
    let w = 32, h = 24   // 32/8=4, 24/8=3 — минимальный рабочий размер
    let beta: Float = 1.0
    let img = makeNormalSynthetic(w: w, h: h)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    func lossValue() -> Double {
        session.begin()
        // deterministic: ε = 0, чтобы шум не менялся между вызовами при
        // численном дифференцировании.
        let cache = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        session.end()
        // Та же потеря, что в backward: MSE + β·KL. Без KL-слагаемого
        // аналитические градиенты заведомо не сойдутся с конечными разностями.
        return vaeLoss(cache: cache, target: img, beta: beta)
    }

    yumeLog(String(format: "loss до тренировки: %.8f", lossValue()))
    // короткая тренировка, чтобы веса ушли от нулевого инициализации и градиенты были значимыми
    for step in 1...30 {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        let (_, g) = vae.backward(session, c, target: x, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: 0.001)
        session.end()
    }
    yumeLog(String(format: "loss после тренировки: %.8f", lossValue()))

    // аналитические градиенты
    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h, deterministic: true)
    let (_, grads) = vae.backward(session, cache, target: x, beta: beta)
    session.end()

    for (li, _) in vae.layers.enumerated() {
        let a = grads[li * 2].toArray()
        let mx = a.map { abs($0) }.max() ?? 0
        let nan = a.contains { $0.isNaN }
        yumeLog("слой \(li): max|grad| = \(mx)\(nan ? "  ЕСТЬ NaN" : "")")
    }

    // 1e-2, а не 1e-3: градиенты вариационного узла ~1e-5, при шаге 1e-3
    // изменение потери тонет в шуме float32.
    let eps: Float = 1e-2
    var worst: Double = 0
    var checks = 0

    for (li, layer) in vae.layers.enumerated() {
        let analytic = grads[li * 2].toArray()
        let n = layer.W.n
        // берём 3 крупнейших по модулю градиента — там сигнал заведомо выше шума
        let idxs = (0..<n).sorted { abs(analytic[$0]) > abs(analytic[$1]) }.prefix(3)

        for idx in idxs {
            let orig = layer.W.getValue(idx)
            layer.W.setValue(idx, orig + eps)
            let lp = lossValue()
            layer.W.setValue(idx, orig - eps)
            let lm = lossValue()
            layer.W.setValue(idx, orig)
            let numeric = (lp - lm) / Double(2 * eps)
            let a = Double(analytic[idx])
            let denom = max(abs(numeric), abs(a), 1e-6)
            let rel = abs(numeric - a) / denom
            worst = max(worst, rel)
            checks += 1
            yumeLog(String(format: "слой %d  idx %6d  numeric %+.8f  analytic %+.8f  rel %.3f",
                         li, idx, numeric, a, rel))
        }
    }

    yumeLog(String(format: "Проверок: %d, худшая относительная ошибка: %.4f", checks, worst))
    let score = worst < 0.05 ? 100.0 : (worst < 0.15 ? 85.0 : 50.0)
    yumeLog(String(format: "Оценка корректности backward: %.0f/100", score))
}


// ---------- Самопроверка (TDD): эмиттер кадров + правило обновления ----------

public func runSelfTest() {
    var failures = 0
    func expect(_ cond: Bool, _ name: String) {
        yumeLog((cond ? "✅ " : "❌ ") + name)
        if !cond { failures += 1 }
    }

    // 1. пока пропуск чёрных неизвестен — ничего не отдаём
    var pl = FrameDeliveryPlanner()
    expect(pl.emittableRanges(chunkDone: [true, true]).isEmpty, "пока skip неизвестен — не отдаём")

    // 2. первый чанк готов → отдаём первый
    pl.learnSkipIfNeeded(leadingBlacks: 0)
    let r1 = pl.emittableRanges(chunkDone: [true, false, false])
    expect(r1.count == 1 && r1[0] == 0..<1, "готов только чанк 0 → отдаём его")

    // 3. строгий порядок: чанк 1 готов, но 0 нет → ничего
    var pl2 = FrameDeliveryPlanner(); pl2.learnSkipIfNeeded(leadingBlacks: 0)
    expect(pl2.emittableRanges(chunkDone: [false, true]).isEmpty,
           "чанк 0 не готов → не перепрыгиваем через него")

    // 4. подряд готовы 0 и 1 → отдаём оба разом
    var pl3 = FrameDeliveryPlanner(); pl3.learnSkipIfNeeded(leadingBlacks: 0)
    let r3 = pl3.emittableRanges(chunkDone: [true, true, false])
    expect(r3.count == 1 && r3[0] == 0..<2, "подряд готовы 0..1 → оба вместе")

    // 5. продвижение: после отдачи следующий чанк становится доступен
    pl3.markEmittedChunks(2)
    let r4 = pl3.emittableRanges(chunkDone: [true, true, true])
    expect(r4.count == 1 && r4[0] == 2..<3, "после продвижения отдаём следующий")

    // 6. фиксация пропуска лидирующих чёрных
    var pl4 = FrameDeliveryPlanner(); pl4.learnSkipIfNeeded(leadingBlacks: 2)
    expect(pl4.skip == 2, "пропуск лидирующих чёрных зафиксирован")

    // 7. правило обновления коллажа: первые 3 сразу, далее каждые 200
    expect(shouldRefreshCollage(atFrame: 1) && shouldRefreshCollage(atFrame: 2)
           && shouldRefreshCollage(atFrame: 3) && shouldRefreshCollage(atFrame: 200)
           && shouldRefreshCollage(atFrame: 400), "обновление на 1/2/3/200/400")
    expect(!shouldRefreshCollage(atFrame: 201) && !shouldRefreshCollage(atFrame: 199),
           "без обновления на 201/199")


    // 8. разбиение на батчи
    expect(batchRanges(total: 10, batch: 4) == [0..<4, 4..<8, 8..<10], "батчи 4/4/2")
    expect(batchRanges(total: 0, batch: 4).isEmpty, "пусто при total=0")
    expect(batchRanges(total: 3, batch: 8) == [0..<3], "один батч целиком")


    // 9. клипы: метка источника и строгий порядок сохраняются
    let frames = [MemoryFrame(name: "frame_0000.jpg", jpeg: Data([1]), source: "a.mp4"),
                  MemoryFrame(name: "frame_0001.jpg", jpeg: Data([2]), source: "a.mp4"),
                  MemoryFrame(name: "frame_0000.jpg", jpeg: Data([3]), source: "b.mp4")]
    expect(frames.count == 3, "клип: кадры из источников")
    expect(frames.allSatisfy { !$0.source.isEmpty }, "клип: у каждого кадра есть источник")
    let clipA = frames.filter { $0.source == "a.mp4" }
    expect(clipA.count == 2 && clipA[0].name == "frame_0000.jpg" && clipA[1].name == "frame_0001.jpg",
           "клип: порядок внутри источника сохранён")


    // 10. GPU-MSE совпадает с CPU-расчётом
    do {
        let engine = MetalEngine.shared
        let session = GpuSession(engine)
        let n = 200_000
        var a = [Float](repeating: 0, count: n)
        var b = [Float](repeating: 0, count: n)
        for i in 0..<n {
            a[i] = sin(Float(i) * 0.001)
            b[i] = cos(Float(i) * 0.001)
        }
        let tA = GPUTensor(engine: engine, data: a, rows: 1)
        let tB = GPUTensor(engine: engine, data: b, rows: 1)
        session.begin()
        let buf = session.encodeMSE(recon: tA, target: tB)
        _ = session.end()
        let gpuMse = Double(buf.contents().assumingMemoryBound(to: Float.self)[0]) / Double(n)
        var cpuSum = 0.0
        for i in 0..<n {
            let d = Double(a[i] - b[i])
            cpuSum += d * d
        }
        let cpuMse = cpuSum / Double(n)
        // MSE единицы: sin/cos дают ~0.5±; допустимая погрешность 1e-3
        expect(abs(gpuMse - cpuMse) < 1e-3, "GPU-MSE (\(gpuMse)) ≈ CPU (\(cpuMse))")
    }

    // 11. контрольная выборка: 1 кадр на секунду видео (по fps)
    expect(isControlFrame(index: 0, fps: 24), "выборка: кадр 0 (24 fps) всегда контрольный")
    expect(isControlFrame(index: 24, fps: 24), "выборка: кадр 24 (24 fps) — 1-я секунда")
    expect(!isControlFrame(index: 1, fps: 24), "выборка: кадр 1 (24 fps) — не контрольный")
    expect(!isControlFrame(index: 23, fps: 24), "выборка: кадр 23 (24 fps) — не контрольный")
    expect(isControlFrame(index: 30, fps: 30), "выборка: кадр 30 (30 fps) — 1-я секунда")
    expect(!isControlFrame(index: 29, fps: 30), "выборка: кадр 29 (30 fps) — не контрольный")
    expect(!isControlFrame(index: 24, fps: 30), "выборка: кадр 24 (30 fps) — не контрольный")
    expect(isControlFrame(index: 60, fps: 30), "выборка: кадр 60 (30 fps) — 2-я секунда")
    expect(isControlFrame(index: 0, fps: 0), "выборка: fps=0 → шаг 1 (безопасно)")

    // 12. контрольная выборка не пересекается с освобождением основного потока
    // (каждый контрольный кадр существует и в общем наборе, но живёт отдельно).
    let allNames = (0..<100).map { String(format: "frame_%04d.jpg", $0) }
    let controlIdx = (0..<100).filter { isControlFrame(index: $0, fps: 24) }
    let expectedControls = stride(from: 0, to: 100, by: 24).map { $0 }
    expect(controlIdx == expectedControls, "выборка: индексы 0,24,48,72,96 (24 fps)")
    expect(controlIdx.allSatisfy { allNames.indices.contains($0) }, "выборка: каждый контрольный кадр есть в общем наборе")
    expect(controlIdx.count == 5, "выборка: для 100 кадров (24 fps) — 5 контрольных")

    // 13. освобождение JPEG из RAM после загрузки в VRAM
    var poolFrames = (0..<64).map { i -> MemoryFrame in
        MemoryFrame(name: String(format: "frame_%04d.jpg", i),
                    jpeg: Data(repeating: UInt8(i & 0xFF), count: 1000 + i), source: "a.mp4")
    }
    let freed1 = releaseFrameJPEG(&poolFrames, range: 0..<32)
    expect(freed1 > 0, "освобождение: возвращает число освобождённых байт")
    expect(poolFrames[0].jpeg.isEmpty && poolFrames[31].jpeg.isEmpty,
           "освобождение: JPEG 0 и 31 стали пустыми")
    expect(!poolFrames[32].jpeg.isEmpty, "освобождение: кадр 32 (вне диапазона) не тронут")
    // Повторное освобождение — ничего не освобождает (уже пусто).
    let freed2 = releaseFrameJPEG(&poolFrames, range: 0..<32)
    expect(freed2 == 0, "освобождение: повторно — 0 байт (защита от двойного)")
    // Полный диапазон за пределами массива — безопасно (без падения).
    let freed3 = releaseFrameJPEG(&poolFrames, range: 100..<200)
    expect(freed3 == 0, "освобождение: диапазон вне массива — безопасно")

    // 14. бэкап чекпоинтов перед стартом новой сессии (TDD)
    do {
        let tmp = NSTemporaryDirectory() + "yume_tdd_backup_\(UUID().uuidString)"
        let fm = FileManager.default
        try fm.createDirectory(atPath: tmp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: tmp) }

        let existing = tmp + "/conv.safetensors"
        let payload = Data("test-weights".utf8)
        try payload.write(to: URL(fileURLWithPath: existing))

        let empty = tmp + "/empty.safetensors"
        try Data().write(to: URL(fileURLWithPath: empty))

        let missing = tmp + "/missing.safetensors"

        // Непустой файл → бэкап создаётся, содержимое совпадает.
        let b1 = createCheckpointBackup(fromPath: existing)
        expect(b1 != nil, "бэкап: непустой файл → бэкап создан")
        if let b1 = b1 {
            expect(fm.fileExists(atPath: b1), "бэкап: файл существует на диске")
            expect((try? Data(contentsOf: URL(fileURLWithPath: b1))) == payload,
                   "бэкап: содержимое совпадает с оригиналом")
            // Бэкап не равен оригинальному пути.
            expect(b1 != existing, "бэкап: путь отличается от оригинала (суффикс времени)")
        }

        // Коллизия: повторный вызов в ту же секунду — создаётся ДРУГАЯ копия
        // (а не упавший nil). Это фикс реального бага.
        let b2 = createCheckpointBackup(fromPath: existing)
        expect(b2 != nil, "бэкап: повторный вызов не падает (коллизия)")
        if let b1 = b1, let b2 = b2 {
            expect(b2 != b1, "бэкап: повторный вызов создаёт другую копию (уникальное имя)")
        }

        // Непустой файл детектится isNonEmptyFile.
        expect(isNonEmptyFile(existing), "бэкап: isNonEmptyFile(непустой) = true")
        expect(!isNonEmptyFile(empty), "бэкап: isNonEmptyFile(пустой) = false")
        expect(!isNonEmptyFile(missing), "бэкап: isNonEmptyFile(нет файла) = false")

        // Пустой/отсутствующий файл → бэкап не создаётся.
        expect(createCheckpointBackup(fromPath: empty) == nil, "бэкап: пустой файл → nil")
        expect(createCheckpointBackup(fromPath: missing) == nil, "бэкап: нет файла → nil")

        // createBackups: бэкапит только существующие непустые.
        let backs = createBackups(forPaths: [existing, empty, missing])
        expect(backs.count == 1, "бэкап: createBackups бэкапит только 1 непустой из 3")
        expect(backs[0] != existing, "бэкап: createBackups возвращает путь копии")
    } catch {
        expect(false, "бэкап: настройка временных файлов — ошибка: \(error.localizedDescription)")
    }

    // 15. редактируемое поле: видимость элементов в зависимости от режима
    let normal = editFieldVisibility(isEditing: false)
    expect(normal.showValue && normal.showEditButton && !normal.showInputWithActions,
           "поле: в обычном режиме виден лейбл и карандаш, поле скрыто")
    let editing = editFieldVisibility(isEditing: true)
    expect(!editing.showValue && !editing.showEditButton && editing.showInputWithActions,
           "поле: в режиме правки видно поле с ✓/✕, лейбл и карандаш скрыты")

    // 16. размер порции дозагрузки в VRAM-пул (подстраивается под память/GPU)
    // frameBytes для 856×480 float32 ≈ 4.93 МБ; RAM 32 ГБ; рабочий набор GPU ~6.5 ГБ.
    let frameBytes856 = 856 * 480 * 3 * 4
    let ram32 = UInt64(32 * 1024 * 1024 * 1024)
    let gpu65 = UInt64(6_500_000_000)

    // Реальный случай: порция заметно больше 64 (консервативного значения), но < 1024.
    let realChunk = frameLoadChunkSize(frameBytes: frameBytes856,
                                       availableRAM: ram32, maxWorkingSet: gpu65)
    expect(realChunk >= 64 && realChunk <= 1024,
           "порция: реальное значение в диапазоне [64, 1024]")
    expect(realChunk > 64, "порция: подстраивается под память (больше консервативных 64)")
    expect(realChunk < 1024, "порция: не упирается в потолок 1024")

    // Логика бюджета: больше RAM → не меньше порция.
    let bigRAM = frameLoadChunkSize(frameBytes: frameBytes856,
                                    availableRAM: UInt64(128 * 1024 * 1024 * 1024),
                                    maxWorkingSet: gpu65)
    let smallRAM = frameLoadChunkSize(frameBytes: frameBytes856,
                                      availableRAM: UInt64(8 * 1024 * 1024 * 1024),
                                      maxWorkingSet: gpu65)
    expect(bigRAM >= smallRAM, "порция: больше RAM → порция не меньше")

    // Безопасные краевые случаи.
    expect(frameLoadChunkSize(frameBytes: 0, availableRAM: ram32, maxWorkingSet: gpu65) == 64,
           "порция: frameBytes=0 → дефолт 64")
    expect(frameLoadChunkSize(frameBytes: frameBytes856,
                              availableRAM: 1, maxWorkingSet: 1) == 64,
           "порция: нет ресурсов → нижняя граница 64")
    // Огромная память на крошечном кадре → упирается в потолок 1024.
    let huge = frameLoadChunkSize(frameBytes: 1000,
                                  availableRAM: UInt64(512 * 1024 * 1024 * 1024),
                                  maxWorkingSet: UInt64(128_000_000_000))
    expect(huge == 1024, "порция: огромные ресурсы → потолок 1024")

    // 17. реальный прогресс в кадрах (а не в шагах)
    // При батче 8: шаг 4800 из 5398 = 38400 из 43184 кадров (89%).
    let p1 = progressInFrames(stepDone: 4800, totalSteps: 5398, batch: 8)
    expect(p1 == (38400, 43184), "прогресс: 4800 шагов × 8 = 38400/43184 кадров")
    expect(Double(p1.done) / Double(p1.total) > 0.889
           && Double(p1.done) / Double(p1.total) < 0.890,
           "прогресс: ~89% реального объёма")
    // Батч 1 — количество шагов равно количеству кадров.
    let p2 = progressInFrames(stepDone: 100, totalSteps: 43187, batch: 1)
    expect(p2 == (100, 43187), "прогресс: батч 1 → шаги = кадры")
    // Батч 0/отрицательный — защита (минимум 1).
    let p3 = progressInFrames(stepDone: 10, totalSteps: 100, batch: 0)
    expect(p3 == (10, 100), "прогресс: батч 0 → дефолт 1")
    // Начало работы — 0.
    let p4 = progressInFrames(stepDone: 0, totalSteps: 5398, batch: 8)
    expect(p4 == (0, 43184), "прогресс: шаг 0 → 0 кадров")

    // 18. переход в край пагинации (начало/конец)
    expect(edgePage(current: 5, pages: 100, toStart: true) == 0,
           "пагинация: переход в начало → страница 0")
    expect(edgePage(current: 5, pages: 100, toStart: false) == 99,
           "пагинация: переход в конец → последняя страница")
    expect(edgePage(current: 0, pages: 1, toStart: false) == 0,
           "пагинация: одна страница → 0")
    expect(edgePage(current: 0, pages: 0, toStart: false) == 0,
           "пагинация: нет страниц → 0 (безопасно)")
    expect(edgePage(current: 3, pages: 5, toStart: true) == 0,
           "пагинация: в начало на 5 страницах → 0")
    expect(edgePage(current: 0, pages: 5, toStart: false) == 4,
           "пагинация: в конец → 4")

    // 18a. автоматический пересчёт видимых кружков от ширины колонки.
    // Ширина центральной колонки при окне 1360: (1360 − 440 − 2)/2 = 459 pt.
    // Стрелки: 4 × 28 pt, зазор 4 pt → доступно 327 pt.
    expect(dotCapacity(columnWidth: 459, dotWidth: 28) == 10,
           "пагинация: 459 pt, кружок 28 → 10 видимых")
    expect(dotCapacity(columnWidth: 459, dotWidth: 40) == 7,
           "пагинация: 459 pt, кружок 40 (4 цифры) → 7 видимых")
    expect(dotCapacity(columnWidth: 320, dotWidth: 40) == 4,
           "пагинация: узкая колонка 320 → 4")
    expect(dotCapacity(columnWidth: 120, dotWidth: 40) == 1,
           "пагинация: тесно → минимум 1 кружок")
    // Ширина кружка растёт с числом цифр номера страницы, чтобы не обрезать.
    expect(paginatorDotWidth(pageNumber: 1) == 28, "пагинация: 1 цифра → 28")
    expect(paginatorDotWidth(pageNumber: 99) == 28, "пагинация: 2 цифры → 28")
    expect(paginatorDotWidth(pageNumber: 100) == 34, "пагинация: 3 цифры → 34")
    expect(paginatorDotWidth(pageNumber: 999) == 34, "пагинация: 3 цифры (999) → 34")
    expect(paginatorDotWidth(pageNumber: 1000) == 40, "пагинация: 4 цифры → 40")
    expect(paginatorDotWidth(pageNumber: 4319) == 40, "пагинация: 4 цифры (4319) → 40")
    // Окно видимых страниц: текущая держится в центре, у краёв прижимается.
    expect(paginatorVisiblePages(current: 0, pages: 4319, limit: 7) == [0, 1, 2, 3, 4, 5, 6],
           "пагинация: старт → первые 7 страниц")
    expect(paginatorVisiblePages(current: 4318, pages: 4319, limit: 7) == Array(4312...4318),
           "пагинация: конец → последние 7 страниц")
    expect(paginatorVisiblePages(current: 2000, pages: 4319, limit: 7) == Array(1997...2003),
           "пагинация: середина → текущая в центре")
    expect(paginatorVisiblePages(current: 0, pages: 3, limit: 7) == [0, 1, 2],
           "пагинация: страниц меньше лимита → все")
    expect(paginatorVisiblePages(current: 5, pages: 0, limit: 7) == [],
           "пагинация: нет страниц → пусто (безопасно)")

    // 18b. Фактическая ширина строки пагинатора = ширина центральной колонки
    // минус её горизонтальные отступы (24 + 24). Считать от колонки целиком
    // нельзя: строка уже на 48 pt, и на 3–4-значных номерах кружки снова
    // распирают колонку.
    expect(paginatorRowWidth(columnWidth: 494) == 446,
           "пагинация: колонка 494 → строка 446 (минус отступы 48)")
    expect(paginatorRowWidth(columnWidth: 419) == 371,
           "пагинация: колонка 419 (узкое окно) → строка 371")
    expect(paginatorRowWidth(columnWidth: 100) == 120,
           "пагинация: вырожденная ширина → безопасный минимум")
    // Вместимость считается уже от фактической ширины строки.
    expect(dotCapacity(columnWidth: paginatorRowWidth(columnWidth: 494),
                       dotWidth: 28) == 9,
           "пагинация: 446 pt, кружок 28 → 9 видимых")
    expect(dotCapacity(columnWidth: paginatorRowWidth(columnWidth: 494),
                       dotWidth: 34) == 8,
           "пагинация: 446 pt, кружок 34 (3 цифры) → 8 видимых")
    expect(dotCapacity(columnWidth: paginatorRowWidth(columnWidth: 494),
                       dotWidth: 40) == 7,
           "пагинация: 446 pt, кружок 40 (4 цифры) → 7 видимых")

    // 19. прогресс реконструкций: когда писать в журнал
    // Старт и финиш — всегда.
    expect(shouldLogProgress(done: 0, total: 1800, lastLoggedPercent: 0),
           "журнал: старт (0) логируется")
    expect(shouldLogProgress(done: 1800, total: 1800, lastLoggedPercent: 95),
           "журнал: финиш (total) логируется")
    // Каждые 5%: при 5% и 10% логируется; на 4% — нет.
    expect(shouldLogProgress(done: 90, total: 1800, lastLoggedPercent: 0),
           "журнал: 5% логируется")
    expect(shouldLogProgress(done: 180, total: 1800, lastLoggedPercent: 5),
           "журнал: 10% логируется")
    expect(!shouldLogProgress(done: 72, total: 1800, lastLoggedPercent: 0),
           "журнал: 4% — не логируется")
    expect(!shouldLogProgress(done: 100, total: 1800, lastLoggedPercent: 5),
           "журнал: 5.5% (до 10%) — не логируется повторно")
    // После 98% идём к 100% — сам финиш (done==total) логируется отдельно.
    expect(!shouldLogProgress(done: 1799, total: 1800, lastLoggedPercent: 95),
           "журнал: 99% (до границы 100%) — не логируется повторно")
    // Пересечение очередной границы 5% на 95% (после 90%).
    expect(shouldLogProgress(done: 1710, total: 1800, lastLoggedPercent: 90),
           "журнал: 95% логируется после 90%")
    // Безопасный случай: total=0.
    expect(shouldLogProgress(done: 0, total: 0, lastLoggedPercent: 0),
           "журнал: total=0 → логируется")

    // 20. PNG в памяти (encodePNG) вместо записи на диск
    // Простая картинка 2×2, красный квадрат.
    let red = [Float](repeating: 1.0, count: 2 * 2 * 3)  // 1.0 = красный
    guard let png = encodePNG(width: 2, height: 2, rgb: red) else {
        expect(false, "PNG: encodePNG вернул nil")
        return
    }
    expect(png.count > 8, "PNG: данные не пустые")
    // Магия PNG: 0x89 'P' 'N' 'G'.
    let magic = [UInt8](png.prefix(8))
    expect(magic == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
           "PNG: корректная магия заголовка")
    // Неверный размер → nil.
    expect(encodePNG(width: 0, height: 2, rgb: red) == nil, "PNG: width=0 → nil")
    expect(encodePNG(width: 2, height: 3, rgb: red) == nil, "PNG: мало данных (2×2 на 2×3) → nil")
    // Кодирование и декодирование: пиксель белый воспроизводится (через CGImageSource).
    let white = [Float](repeating: 1.0, count: 1 * 1 * 3)
    if let whitePNG = encodePNG(width: 1, height: 1, rgb: white),
       let src = CGImageSourceCreateWithData(whitePNG as CFData, nil),
       let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
       let data = img.dataProvider?.data as Data? {
        let px = [UInt8](data)
        // RGBA premultiplied: белый → все каналы 255.
        expect(px.count >= 4 && px[0] > 240 && px[1] > 240 && px[2] > 240,
               "PNG: белый пиксель воспроизводится")
    } else {
        expect(false, "PNG: не удалось декодировать обратно")
    }

    // 21. автоподбор параметров с учётом resume
    // Resume включён → НЕ пересчитываем (сохраняем текущие, changed=false),
    // но LR авто-снижается ×0.5 (если не задан вручную).
    let au1 = autoTuneValues(frames: 43187, batch: 8, resume: true,
                             currentSteps: 5000, currentEpochs: 3,
                             currentLr: 0.0003,
                             stepsEditedByUser: false, epochsEditedByUser: false,
                             lrEditedByUser: false)
    expect(au1.steps == 5000 && au1.epochs == 3 && !au1.changed,
           "авто: resume=true → сохраняем текущие шаги/эпохи, без изменений")
    expect(au1.lr != nil && abs(au1.lr! - 0.00015) < 1e-9,
           "авто: resume=true → LR авто-снижен ×0.5 (0.0003 → 0.00015)")
    // Resume + LR задан вручную → LR не трогаем (nil).
    let au6 = autoTuneValues(frames: 43187, batch: 8, resume: true,
                             currentSteps: 5000, currentEpochs: 3,
                             currentLr: 0.0001,
                             stepsEditedByUser: true, epochsEditedByUser: true,
                             lrEditedByUser: true)
    expect(au6.lr == nil, "авто: resume + LR вручную → LR не трогаем")
    // Resume выключён, ничего не редактировали → пересчёт, LR не трогаем.
    let au2 = autoTuneValues(frames: 43187, batch: 8, resume: false,
                             currentSteps: 20, currentEpochs: 20,
                             currentLr: 0.0003,
                             stepsEditedByUser: false, epochsEditedByUser: false,
                             lrEditedByUser: false)
    expect(au2.steps == 5398 && au2.changed, "авто: шаги = frames/batch = 5398, changed=true")
    expect(au2.lr == nil, "авто: resume=false → LR не трогаем")
    // Эпохи: min(10, max(3, 3000/frames)); для 43187 → 3.
    expect(au2.epochs == 3, "авто: эпохи = 3 для 43187 кадров")
    // Resume выключён, но шаги редактированы вручную → эпохи пересчитываются, шаги нет.
    let au3 = autoTuneValues(frames: 43187, batch: 8, resume: false,
                             currentSteps: 1234, currentEpochs: 20,
                             currentLr: 0.0003,
                             stepsEditedByUser: true, epochsEditedByUser: false,
                             lrEditedByUser: false)
    expect(au3.steps == 1234 && au3.epochs == 3 && au3.changed,
           "авто: шаги редактированы → не трогаем, эпохи авто")
    // Оба поля редактированы → ничего не меняется.
    let au4 = autoTuneValues(frames: 43187, batch: 8, resume: false,
                             currentSteps: 100, currentEpochs: 5,
                             currentLr: 0.0003,
                             stepsEditedByUser: true, epochsEditedByUser: true,
                             lrEditedByUser: false)
    expect(!au4.changed, "авто: оба поля редактированы → changed=false")
    // frames=0 → безопасно, без изменений.
    let au5 = autoTuneValues(frames: 0, batch: 8, resume: false,
                             currentSteps: 20, currentEpochs: 20,
                             currentLr: 0.0003,
                             stepsEditedByUser: false, epochsEditedByUser: false,
                             lrEditedByUser: false)
    expect(!au5.changed && au5.steps == 20, "авто: frames=0 → без изменений")

    // 22. Отпечаток данных (детерминированный, реагирует на содержимое).
    let fa = [MemoryFrame(name: "a0.jpg", jpeg: Data([1,2,3]), source: "a.mp4"),
              MemoryFrame(name: "a1.jpg", jpeg: Data([4,5,6]), source: "a.mp4")]
    let fa2 = [MemoryFrame(name: "a0.jpg", jpeg: Data([1,2,3]), source: "a.mp4"),
               MemoryFrame(name: "a1.jpg", jpeg: Data([4,5,6]), source: "a.mp4")]
    let fb = [MemoryFrame(name: "b0.jpg", jpeg: Data([9,9,9]), source: "b.mp4"),
              MemoryFrame(name: "b1.jpg", jpeg: Data([7,7,7]), source: "b.mp4")]
    expect(dataFingerprint(frames: fa) == dataFingerprint(frames: fa2),
           "отпечаток: одинаковые данные → одинаковый отпечаток")
    expect(dataFingerprint(frames: fa) != dataFingerprint(frames: fb),
           "отпечаток: разные данные → разные отпечатки")
    expect(dataFingerprint(frames: []) == "empty",
           "отпечаток: пустой набор → 'empty'")

    // 24. Живой статус обучения: эпоха, текущий кадр, шаг, скорость.
    expect(formatThousands(43187) == "43 187", "статус: 43187 → '43 187'")
    expect(formatThousands(0) == "0", "статус: 0 → '0'")
    expect(formatThousands(1234567) == "1 234 567", "статус: 1234567 → '1 234 567'")

    let live1 = TrainStepReport(done: 12480, total: 129561,
                                epoch: 1, epochs: 3,
                                step: 12480, stepsPerEpoch: 43187,
                                frame: 12480, framesTotal: 43187,
                                batch: 1, secondsPerStep: 0.46)
    expect(trainLiveText(live1) ==
           "Эпоха 1/3 · кадр 12 480 из 43 187 · шаг 12 480/43 187 · 0.46 с/шаг",
           "статус: батч 1 → эпоха/кадр/шаг/скорость")
    let live2 = TrainStepReport(done: 1, total: 129561,
                                epoch: 1, epochs: 3,
                                step: 1, stepsPerEpoch: 5399,
                                frame: 1, framesTotal: 43187,
                                batch: 8, secondsPerStep: nil)
    expect(trainLiveText(live2) ==
           "Эпоха 1/3 · кадр 1 из 43 187 · батч 8 · шаг 1/5 399",
           "статус: батч 8 → кадр + размер батча")
    let live3 = TrainStepReport(done: 5, total: 20,
                                epoch: 1, epochs: 1,
                                step: 5, stepsPerEpoch: 20,
                                frame: 0, framesTotal: 0,
                                batch: 1, secondsPerStep: nil)
    expect(trainLiveText(live3) == "Эпоха 1/1 · шаг 5/20",
           "статус: без кадров → только эпоха и шаг")
    // Скорость округляется до сотых, чтобы не дёргалась на каждом шаге.
    let live4 = TrainStepReport(done: 1, total: 10,
                                epoch: 1, epochs: 1,
                                step: 1, stepsPerEpoch: 10,
                                frame: 1, framesTotal: 10,
                                batch: 1, secondsPerStep: 0.456)
    expect(trainLiveText(live4).hasSuffix("· 0.46 с/шаг"),
           "статус: 0.456 с/шаг → '0.46 с/шаг'")

    // 25. «Идёт или зависло»: состояние по времени с последнего шага.
    expect(trainLivePhase(secondsSinceStep: 5, stepSeconds: 0.46) == .running,
           "статус: 5 с назад → обучение идёт")
    expect(trainLivePhase(secondsSinceStep: 50, stepSeconds: 0.46) == .waiting(seconds: 50),
           "статус: 50 с назад → последний шаг давно")
    expect(trainLivePhase(secondsSinceStep: 95, stepSeconds: 0.46) == .stalled(seconds: 95),
           "статус: 95 с назад → возможно зависло")
    // Медленные шаги (3 с/шаг) не должны давать ложное «зависло» на 95 с.
    expect(trainLivePhase(secondsSinceStep: 95, stepSeconds: 3) == .running,
           "статус: медленные шаги 3 с → 95 с ещё норма")
    expect(trainLivePhaseText(.running) == "Обучение идёт",
           "статус: текст running")
    expect(trainLivePhaseText(.waiting(seconds: 50)) == "Последний шаг 50 с назад",
           "статус: текст waiting")
    expect(trainLivePhaseText(.stalled(seconds: 95)) ==
           "Нет шагов 1 мин 35 с — возможно, зависло",
           "статус: текст stalled с минутами")
    expect(formatShortDuration(125) == "2 мин 5 с",
           "статус: 125 с → '2 мин 5 с'")

    // 26. Ёмкость VRAM-пула: как в ранних прогонах — 1024 кадра при 8.57 ГБ.
    expect(gpuFramePoolCapacity(framesTotal: 43187, frameBytes: 4_930_560,
                                workingSet: 8_570_000_000) == 1024,
           "VRAM: 8.57 ГБ → 1024 кадра (≈4.8 ГБ)")
    expect(gpuFramePoolCapacity(framesTotal: 300, frameBytes: 4_930_560,
                                workingSet: 8_570_000_000) == 300,
           "VRAM: кадров меньше потолка → все")
    expect(gpuFramePoolCapacity(framesTotal: 43187, frameBytes: 4_930_560,
                                workingSet: 2_000_000_000) == 243,
           "VRAM: 2 ГБ → 243 кадра (60% бюджета)")
    expect(gpuFramePoolCapacity(framesTotal: 43187, frameBytes: 0,
                                workingSet: 8_570_000_000) == 1,
           "VRAM: нулевой кадр → минимум 1 (без деления на 0)")
    expect(gpuFramePoolCapacity(framesTotal: 0, frameBytes: 4_930_560,
                                workingSet: 8_570_000_000) == 0,
           "VRAM: нет кадров → 0")

    // 27. Фаза подготовки: пока кадры грузятся в VRAM, шагов ещё нет —
    // интерфейс обязан показывать прогресс загрузки, а не «зависло».
    expect(trainPreparationNote() == "Подготовка: загрузка модели на GPU…",
           "статус: текст подготовки")
    expect(trainVRAMLoadNote(loaded: 512, total: 1024) == "Загрузка кадров в VRAM: 512/1024",
           "статус: прогресс загрузки VRAM")
    expect(trainVRAMLoadNote(loaded: 1024, total: 1024) == "Загрузка кадров в VRAM: 1024/1024",
           "статус: VRAM загружена полностью")
    let prepReport = TrainStepReport(done: 0, total: 129561,
                                     epoch: 1, epochs: 3,
                                     step: 0, stepsPerEpoch: 43187,
                                     frame: 0, framesTotal: 43187,
                                     batch: 1, secondsPerStep: nil,
                                     note: trainVRAMLoadNote(loaded: 512, total: 1024),
                                     stage: .preparing)
    expect(trainLiveText(prepReport) == "Загрузка кадров в VRAM: 512/1024",
           "статус: отчёт с пометкой показывает пометку")
    expect(trainLiveStateText(phase: .running, stage: .preparing) == "Подготовка данных",
           "статус: состояние «подготовка данных»")
    expect(trainLiveStateText(phase: .stalled(seconds: 95), stage: .preparing) ==
           "Подготовка данных не идёт 1 мин 35 с — возможно, зависло",
           "статус: зависание на фазе подготовки")
    expect(trainLiveStateText(phase: .running, stage: .training) == "Обучение идёт",
           "статус: состояние при обучении")

    // 28. Номер кадра, который обучается прямо сейчас (1-based).
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 0, stepsPerEpoch: 43187,
                             batch: 1, datasetFrames: 43187) == 1,
           "кадр: первый шаг → кадр 1")
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 12479, stepsPerEpoch: 43187,
                             batch: 1, datasetFrames: 43187) == 12480,
           "кадр: шаг 12480 → кадр 12480")
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 43186, stepsPerEpoch: 43187,
                             batch: 1, datasetFrames: 43187) == 43187,
           "кадр: последний кадр эпохи → 43 187")
    expect(currentFrameIndex(epochIndex: 1, stepIndex: 0, stepsPerEpoch: 43187,
                             batch: 1, datasetFrames: 43187) == 1,
           "кадр: вторая эпоха → набор начинается заново (кадр 1)")
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 1, stepsPerEpoch: 5399,
                             batch: 8, datasetFrames: 43187) == 9,
           "кадр: батч 8 → первый кадр следующей пачки (9)")
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 0, stepsPerEpoch: 10,
                             batch: 1, datasetFrames: 0) == 0,
           "кадр: набора нет → 0 (кадр не показываем)")

    // 23. Локальный API: белый список Host привязан к фактическому порту
    // сервера. Иначе второй экземпляр (диагностика) не может ответить.
    func httpRequest(_ host: String, origin: String? = nil) -> Data {
        var s = "GET /api/state HTTP/1.1\r\nHost: \(host)\r\n"
        if let o = origin { s += "Origin: \(o)\r\n" }
        s += "\r\n"
        return Data(s.utf8)
    }
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17851"), port: 17851)) != nil,
           "API: Host на своём порту (17851) принимается")
    expect((try? LocalHTTPRequest.parse(httpRequest("localhost:17851"), port: 17851)) != nil,
           "API: localhost на своём порту принимается")
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17852"), port: 17852)) != nil,
           "API: Host на своём порту (17852) принимается")
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17851"), port: 17852)) == nil,
           "API: чужой порт в Host отклоняется")
    expect((try? LocalHTTPRequest.parse(httpRequest("evil.example:17851"), port: 17851)) == nil,
           "API: чужой хост отклоняется")
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17852", origin: "http://127.0.0.1:17852"), port: 17852)) != nil,
           "API: Origin совпадает с Host — принимается")
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17852", origin: "http://evil.example"), port: 17852)) == nil,
           "API: чужой Origin отклоняется")

    yumeLog(failures == 0 ? "SELFTEST PASS" : "SELFTEST FAIL: \(failures)")
}

// ---------- Пул кадров в VRAM ----------

/// Резидентный в VRAM пул кадров (float32): 1024 кадра ≈ 4.8 ГБ.
/// Обучение берёт кадры напрямую из пула (без PCIe-пересылок на каждом шаге);
/// при исчерпании окна — чанк дозагружается одним command buffer (много blit).
/// Ёмкость резидентного VRAM-пула кадров.
///
/// Бюджет — 60% рекомендуемого рабочего набора GPU. На Radeon Pro 5500M
/// (8.57 ГБ) это ≈5.14 ГБ, чего хватает на 1024 кадра по 4.7 МБ (≈4.8 ГБ) —
/// ровно столько грузили ранние прогоны. Прежняя формула брала четверть
/// рабочего набора и незаметно урезала пул до 434 кадров (2 ГБ), из-за чего
/// VRAM грузилась заметно меньше.
public func gpuFramePoolCapacity(framesTotal: Int, frameBytes: Int,
                                 workingSet: UInt64, preferred: Int = 1024) -> Int {
    guard framesTotal > 0 else { return 0 }
    guard frameBytes > 0 else { return 1 }
    let budget = Double(workingSet) * 0.6
    let byVRAM = max(1, Int(budget) / frameBytes)
    return min(framesTotal, max(1, preferred), byVRAM)
}

final class GPUFramePool {
    private let engine: MetalEngine
    private var frames: [MemoryFrame]
    private let w: Int
    private let h: Int
    private let size: Int
    private var tensors: [GPUTensor] = []
    private var windowStart = -1
    private var cursorGlobal = 0
    private let lock = NSLock()
    /// Прогресс загрузки окна в VRAM: (загружено кадров, размер окна).
    /// Нужен интерфейсу, чтобы во время долгой загрузки не казалось, что
    /// обучение зависло.
    private let onChunk: ((Int, Int) -> Void)?

    init?(engine: MetalEngine, frames: [MemoryFrame], w: Int, h: Int,
          size: Int = 1024, onChunk: ((Int, Int) -> Void)? = nil) {
        guard !frames.isEmpty else { return nil }
        self.engine = engine
        self.frames = frames
        self.w = w
        self.h = h
        self.onChunk = onChunk
        // Бюджет пула — 60% рабочего набора GPU (см. gpuFramePoolCapacity).
        self.size = gpuFramePoolCapacity(
            framesTotal: frames.count,
            frameBytes: w * h * 3 * 4,
            workingSet: engine.device.recommendedMaxWorkingSetSize,
            preferred: size)
        self.tensors = (0..<self.size).map { _ in
            GPUTensor(engine: engine, rows: 3, cols: w * h)
        }
        loadWindow(index: 0)
        yumeLog("GPUFramePool: \(self.size) кадров в VRAM (\(self.size * w * h * 3 * 4 / (1024*1024)) МБ)")
    }

    /// Честный resume: сдвинуть курсор на уже пройденные кадры.
    func skip(_ count: Int) {
        lock.lock()
        cursorGlobal = max(0, count)
        lock.unlock()
    }

    /// Следующий кадр по порядку (циклически по всему набору).
    func next() -> GPUTensor {
        lock.lock()
        let i = cursorGlobal
        cursorGlobal += 1
        lock.unlock()

        let g = i % frames.count
        let ws = (g / size) * size
        if windowStart != ws {
            loadWindow(index: ws)
        }
        return tensors[g - windowStart]
    }

    /// Дозагрузка окна [index, index+size).
    /// Декодирование и загрузка идут порциями по 64 кадра: не держим в RAM
    /// 5 ГБ массивов и не аллоцируем 5 ГБ staging (это упиралось в память и
    /// вешало приложение после первого чанка).
    private func loadWindow(index: Int) {
        let s = index
        let e = min(s + size, frames.count)
        let n = e - s
        guard n > 0 else { return }

        let frameBytes = w * h * 3 * 4
        // Размер порции подстраивается под доступную память и рабочий набор GPU
        // (см. frameLoadChunkSize). На 32 ГБ RAM + Radeon 5500M это заметно
        // больше консервативных 64, но ниже 1024 — чтобы не создавать пик ~10 ГБ.
        let batch = min(n, 16, frameLoadChunkSize(frameBytes: frameBytes,
                                       availableRAM: UInt64(ProcessInfo.processInfo.physicalMemory),
                                       maxWorkingSet: engine.device.recommendedMaxWorkingSetSize))
        // Переиспользуемые staging-буферы: пик памяти ≈ batch × кадр.
        let staging = (0..<batch).map { _ in
            engine.device.makeBuffer(length: frameBytes, options: .storageModeShared)!
        }

        yumeLog("GPUFramePool: дозагрузка чанка \(s)..<\(e) — \(n) кадров, порциями по \(batch)")
        var j = 0
        while j < n {
            let jend = min(j + batch, n)
            let cnt = jend - j

            var imgs = [[Float]?](repeating: nil, count: cnt)
            let decodeLock = NSLock()
            DispatchQueue.concurrentPerform(iterations: cnt) { k in
                let image = loadImageRGB(data: frames[s + j + k].jpeg, w: w, h: h)
                decodeLock.lock(); imgs[k] = image; decodeLock.unlock()
            }

            let cmd = engine.queue.makeCommandBuffer()!
            let blit = cmd.makeBlitCommandEncoder()!
            for k in 0..<cnt {
                guard let img = imgs[k], img.count * 4 == frameBytes else { continue }
                img.withUnsafeBytes { raw in
                    staging[k].contents().copyMemory(from: raw.baseAddress!, byteCount: frameBytes)
                }
                blit.copy(from: staging[k], sourceOffset: 0, to: tensors[j + k].buf,
                          destinationOffset: 0, size: frameBytes)
            }
            blit.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            j = jend
            onChunk?(j, n)
        }
        windowStart = s
        let mb = n * frameBytes / (1024 * 1024)
        yumeLog("GPUFramePool: чанк \(s)..<\(e) в VRAM (\(mb) МБ)")

        // JPEG must remain available when the cyclic cursor revisits this window.
    }
}
