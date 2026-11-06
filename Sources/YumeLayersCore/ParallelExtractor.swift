// YumeLayers Core — извлечение кадров через ffmpeg ПАРАЛЛЕЛЬНО.
// Видео режется на K непересекающихся временных сегментов (K ≈ число
// ядер CPU, по умолчанию до 10). Каждый сегмент декодируется отдельным
// процессом ffmpeg одновременно; кадры читаются в память и сжимаются в
// JPEG (≈85%), затем собираются в строгом порядке.
//
// Потокобезопасность: один экземпляр используется из одного потока;
// внутренние параллельные задачи синхронизированы через DispatchGroup.

import Foundation
import AVFoundation

/// Потокобезопасный счётчик: рабочие берут следующий чанк по очереди.
private final class AtomicCounter {
    private let lock = NSLock()
    private var value = 0

    func increment() -> Int {
        lock.lock(); defer { lock.unlock() }
        let current = value
        value += 1
        return current
    }
}

/// Потокобезопасный буфер сегментов: сегменты пишут кадры параллельно,
/// а сборщик забирает их строго по порядку, как только сегмент готов.
private final class SegmentBuffer {
    private let condition = NSCondition()
    private var frames: [[Data]]
    private var finished: [Bool]
    private var readCounts: [Int]
    private var positions: [Int]
    private var failed = false
    private var cancelled = false
    private let target: Int

    init(count: Int, target: Int) {
        self.target = max(1, target)
        frames = Array(repeating: [], count: count)
        finished = Array(repeating: false, count: count)
        readCounts = Array(repeating: 0, count: count)
        positions = Array(repeating: 0, count: count)
    }

    /// Добавляет кадр сразу после разбора — сборщик может отдать его,
    /// не дожидаясь конца сегмента.
    func append(_ frame: Data, at index: Int) {
        condition.lock()
        frames[index].append(frame)
        readCounts[index] += 1
        condition.broadcast()
        condition.unlock()
    }

    func finish(at index: Int) {
        condition.lock()
        finished[index] = true
        condition.broadcast()
        condition.unlock()
    }

    func markFailed() {
        condition.lock(); failed = true; condition.broadcast(); condition.unlock()
    }

    var isCancelled: Bool {
        condition.lock(); defer { condition.unlock() }
        return cancelled
    }

    /// Ждёт готовности сегмента `index`. Возвращает false при ошибке/отмене.
    func waitFor(_ index: Int, shouldStop: () -> Bool) -> Bool {
        condition.lock()
        while !finished[index] && !failed && !cancelled {
            if shouldStop() { condition.unlock(); return false }
            condition.wait(until: Date().addingTimeInterval(0.1))
        }
        let ok = finished[index] && !failed && !cancelled
        condition.unlock()
        return ok
    }

    /// Следующий кадр сегмента. Ждёт появления кадра, пока сегмент не
    /// закончился. Возвращает `(кадр, сегментЗакончен)`.
    func next(_ index: Int, shouldStop: () -> Bool) -> (Data?, Bool) {
        condition.lock()
        while positions[index] >= frames[index].count && !finished[index] {
            if failed || cancelled || shouldStop() {
                condition.unlock()
                return (nil, true)
            }
            condition.wait(until: Date().addingTimeInterval(0.05))
        }
        if positions[index] < frames[index].count {
            let frame = frames[index][positions[index]]
            frames[index][positions[index]] = Data()  // сразу освобождаем память
            positions[index] += 1
            condition.unlock()
            return (frame, false)
        }
        let done = finished[index] || failed || cancelled
        condition.unlock()
        return (nil, done)
    }

    var hasFailed: Bool {
        condition.lock(); defer { condition.unlock() }
        return failed
    }
}

final class ParallelExtractor {
    struct Options {
        /// Максимальное число кадров (0 — все кадры исходника).
        var maxFrames: Int = 0
        /// Смещение глобальной нумерации кадров.
        var startIndex: Int = 0
        /// Целевое разрешение кадров. 0 — без перемасштабирования
        /// (кадры исходного размера). Задаётся только при необходимости.
        var targetWidth: Int = 0
        var targetHeight: Int = 0
        /// Число параллельных сегментов (ядер). 0 — авто (до 10).
        var cores: Int = 0
        /// Частота выборки кадров (кадров/с). 0 — как в исходном видео.
        var framesPerSecond: Double = 0
        init(maxFrames: Int = 0, startIndex: Int = 0,
             targetWidth: Int = 0, targetHeight: Int = 0,
             cores: Int = 0, framesPerSecond: Double = 0) {
            self.maxFrames = maxFrames
            self.startIndex = startIndex
            self.targetWidth = targetWidth
            self.targetHeight = targetHeight
            self.cores = cores
            self.framesPerSecond = framesPerSecond
        }
    }

    /// Один кусок ролика: непрерывный диапазон кадров.
    /// Чанки идут строго по порядку и покрывают ролик без дыр и перекрытий.
    struct Chunk {
        /// Номер первого кадра чанка (глобальный, внутри ролика).
        let startFrame: Int
        /// Сколько кадров извлекает этот чанк.
        let frameCount: Int
    }

    /// План извлечения: сколько кадров, с какой частотой и на каких
    /// чанках режется ролик. Чистая функция — проверяется тестами.
    struct Plan {
        let duration: Double
        let sourceFPS: Double
        let allFrames: Int
        let targetFrames: Int
        let chunks: [Chunk]

        /// Сколько чанков (совместимость с ранними версиями плана).
        var segments: Int { chunks.count }

        /// Частота, с которой ffmpeg отдаёт кадры. Без пользовательского
        /// лимита — ровно как в исходнике; с лимитом — равномерная выборка.
        var outputFPS: Double {
            guard duration > 0 else { return sourceFPS }
            if targetFrames >= allFrames { return sourceFPS }
            return Double(targetFrames) / duration
        }

        /// Временной диапазон чанка: границы привязаны к номерам кадров,
        /// чтобы стыки соседних чанков были точными — каждый кадр ролика
        /// извлекается ровно один раз, без сдвигов на стыках.
        func timeRange(of chunk: Chunk) -> (start: Double, end: Double) {
            let fps = max(outputFPS, 1e-6)
            let s = Double(chunk.startFrame) / fps
            let e = Double(chunk.startFrame + chunk.frameCount) / fps
            return (s, e)
        }
    }

    static func makePlan(duration: Double, sourceFPS: Double,
                         maxFrames: Int, cores: Int) -> Plan {
        let fps = sourceFPS.isFinite && sourceFPS > 0 ? sourceFPS : 24
        let allFrames = max(1, Int(duration * fps))
        let effectiveMax = maxFrames > 0 ? maxFrames : allFrames
        let target = max(1, min(allFrames, effectiveMax))
        let chunks = chunkPlan(target: target)
        return Plan(duration: duration, sourceFPS: fps, allFrames: allFrames,
                    targetFrames: target, chunks: chunks)
    }

    /// План чанков: маленький первый чанк — первые кадры попадают в
    /// интерфейс почти сразу, а не после декодирования всего ролика.
    /// Дальше чанки растут до базового размера: большой ролик не порождает
    /// тысячи процессов ffmpeg, а буфер «на лету» ограничен сверху.
    /// Чистая функция — проверяется тестами.
    static func chunkPlan(target: Int) -> [Chunk] {
        guard target > 0 else { return [] }
        var chunks: [Chunk] = []
        var frame = 0
        var size = min(60, target)
        while frame < target {
            let n = min(size, target - frame)
            chunks.append(Chunk(startFrame: frame, frameCount: n))
            frame += n
            size = min(size * 2, 1920)
        }
        return chunks
    }

    /// Качество JPEG в шкале ffmpeg (2 — лучшее, 31 — худшее).
    /// 5 соответствует ≈85%: меньше кадров в RAM без заметной потери.
    static let jpegQuality = 5

    static var ffmpegPath: String? {
        let candidates = [
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "/usr/bin/ffmpeg",
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    private let url: URL
    private let options: Options
    private let lock = NSLock()
    private var processes: [Process] = []
    private var _cancelled = false
    /// Последнее отданное значение прогресса: UI не должен видеть откат
    /// назад, когда таймер чтения и сборщик кадров сообщают наперегонки.
    private var lastReportedProgress = -1.0

    init(url: URL, options: Options = Options()) {
        self.url = url
        self.options = options
    }

    func cancel() {
        lock.lock(); _cancelled = true
        for p in processes { if p.isRunning { p.terminate() } }
        lock.unlock()
    }

    private var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _cancelled
    }

    /// Монотонный прогресс: значения только растут, повторы отбрасываются.
    /// Нужен, потому что прогресс сообщают два источника — таймер чтения
    /// из ffmpeg и сборщик, отдающий кадры в UI.
    private func reportProgress(_ p: Double, progress: (Double) -> Void) {
        lock.lock()
        let clamped = min(max(p, 0), 1)
        guard clamped > lastReportedProgress else { lock.unlock(); return }
        lastReportedProgress = clamped
        lock.unlock()
        progress(clamped)
    }

    /// Останавливает ffmpeg-процессы, не помечая прогон отменённым.
    private func terminateAll() {
        lock.lock()
        for p in processes where p.isRunning { p.terminate() }
        lock.unlock()
    }

    /// Запускает параллельное извлечение. onFrame вызывается строго по
    /// порядку для каждого кадра; progress — 0...1; completion — по
    /// завершении (или отмене/ошибке).
    func run(
        onFrame: @escaping (_ index: Int, _ total: Int, _ ordinal: Int, _ jpeg: Data) -> Void,
        progress: @escaping (Double) -> Void,
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        guard let ffmpeg = Self.ffmpegPath else {
            completion(.failure(FrameExtractError.unreadableAsset)); return
        }
        let asset = AVURLAsset(url: url)
        guard let time = try? loadAssetDuration(asset) else {
            completion(.failure(FrameExtractError.unreadableAsset)); return
        }
        let duration = CMTimeGetSeconds(time)
        guard duration.isFinite, duration > 0.01 else {
            completion(.failure(FrameExtractError.unreadableAsset)); return
        }
        let nominalFPS: Double
        if let track = try? loadVideoTracks(of: asset).first,
           let rate = try? loadNominalFrameRate(of: track), rate > 0 {
            nominalFPS = Double(rate)
        } else {
            nominalFPS = 24
        }
        let requestedFPS = options.framesPerSecond > 0 ? options.framesPerSecond : nominalFPS
        let cores = options.cores > 0 ? options.cores
            : min(max(ProcessInfo.processInfo.activeProcessorCount, 1), 10)
        let plan = Self.makePlan(duration: duration, sourceFPS: requestedFPS,
                                 maxFrames: options.maxFrames, cores: cores)

        // Полное извлечение (все кадры) — ОДИН процесс ffmpeg со всеми
        // ядрами: он и так нагружает CPU целиком, и параллельные чанки не
        // ускоряют, а добавляют накладные на запуск процессов. Параллельные
        // чанки выгодны только при выборке с лимитом — там кадров заметно
        // меньше, чем в исходнике.
        if plan.targetFrames >= plan.allFrames {
            runSinglePass(plan: plan, ffmpeg: ffmpeg,
                          onFrame: onFrame, progress: progress,
                          completion: completion)
            return
        }

        let buffer = SegmentBuffer(count: plan.chunks.count, target: plan.targetFrames)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "yume.parallel", attributes: .concurrent)

        // Параллельно работают не больше `cores` процессов ffmpeg: каждый
        // берёт следующий неотданный чанк. Чанки маленькие, поэтому ГДЕ-ТО
        // В СЕРЕДИНЕ ролика извлечение не замирает — готовые чанки
        // отдаются по мере готовности, в строгом порядке.
        let nextChunk = AtomicCounter()
        let workers = min(max(cores, 1), plan.chunks.count)

        func decodeChunk(_ c: Int) {
            let chunk = plan.chunks[c]
            let range = plan.timeRange(of: chunk)
            guard !isCancelled else { return }

            // Видеофильтр: исходный fps + (опционально) масштаб.
            // Фильтр fps обязателен: он синхронизирует выходную частоту с
            // временными границами чанка, иначе стыки «плывут» и кадры
            // теряются на границах чанков.
            var vf = String(format: "fps=%.4f", plan.outputFPS)
            if options.targetWidth > 0 && options.targetHeight > 0 {
                let scaleBar = "scale=\(options.targetWidth):\(options.targetHeight):force_original_aspect_ratio=decrease:flags=lanczos"
                vf = "\(vf),\(scaleBar)"
            }

            let args = [
                "-hide_banner", "-loglevel", "error",
                "-nostdin",
                // Seek с запасом в полкадра в прошлое и точностью до микросекунды:
                // округление «вверх» до 0.1 мс отбрасывало кадр ровно на границе
                // чанка (29.97 fps и т.п.), а дубли не страшны — предыдущий чанк
                // ограничен своим -frames:v и до этой точки не доходит.
                "-ss", String(format: "%.6f", max(0, range.start - 0.5 / plan.outputFPS)),
                "-i", url.path,
                "-vf", vf,
                "-frames:v", "\(chunk.frameCount)",
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "-q:v", "\(Self.jpegQuality)",
                "-vsync", "0",
                "-",
            ]
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: ffmpeg)
            proc.arguments = args
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice
            lock.lock()
            processes.append(proc)
            lock.unlock()
            do {
                try proc.run()
            } catch {
                buffer.markFailed()
                return
            }
            var gotFrames = 0
            Self.readJpegs(handle: pipe.fileHandleForReading,
                           maxFrames: chunk.frameCount,
                           onCancel: { [self] in isCancelled },
                           onFrame: { jpeg in
                               gotFrames += 1
                               buffer.append(jpeg, at: c)
                           })
            proc.waitUntilExit()
            // Ошибка декодирования внутри чанка (битые данные, сбой seek)
            // раньше оставляла чанк «законченным» с нулём кадров — недобор
            // никто не замечал, completion уходил с успехом.
            if proc.terminationStatus != 0, !isCancelled {
                if gotFrames == 0 {
                    yumeLog("⚠️ ffmpeg: чанк \(c) не выдал ни одного кадра (код \(proc.terminationStatus))")
                    buffer.markFailed()
                    return
                }
                yumeLog("⚠️ ffmpeg: чанк \(c) завершился с кодом \(proc.terminationStatus), получено \(gotFrames)/\(chunk.frameCount) кадров")
            }
            buffer.finish(at: c)
        }

        for _ in 0..<workers {
            group.enter()
            queue.async { [self] in
                defer { group.leave() }
                while true {
                    let c = nextChunk.increment()
                    if c >= plan.chunks.count { break }
                    if isCancelled { break }
                    decodeChunk(c)
                    if buffer.hasFailed { break }
                }
            }
        }

        // Сборка в строгом порядке: кадры первого чанка отдаются сразу, как
        // только он готов (обычно за доли секунды), дальше — каждый следующий
        // чанк по мере готовности. Прогресс считаем ТОЛЬКО по фактически
        // отданным кадрам, чтобы UI не прыгал вперёд «прочитанного»,
        // но ещё не показанного.
        let progressStride = max(1, plan.targetFrames / 100)

        DispatchQueue(label: "yume.emit").async { [self] in
            var idx = 0
            var failure: Error?
            emitLoop: for c in 0..<plan.chunks.count {
                if idx >= plan.targetFrames { break }
                while true {
                    if isCancelled {
                        failure = FrameExtractError.cancelled
                        break emitLoop
                    }
                    let (frame, done) = buffer.next(c, shouldStop: { isCancelled })
                    if let frame {
                        if idx >= plan.targetFrames { break emitLoop }
                        onFrame(idx, plan.targetFrames, idx + options.startIndex, frame)
                        idx += 1
                        if idx % progressStride == 0 {
                            reportProgress(Double(idx) / Double(plan.targetFrames),
                                           progress: progress)
                        }
                        continue
                    }
                    if done { break }
                }
                if buffer.hasFailed {
                    failure = FrameExtractError.unreadableAsset
                    break
                }
            }

            if let failure {
                terminateAll()
                completion(.failure(failure))
                return
            }
            guard idx > 0 else {
                completion(.failure(FrameExtractError.unreadableAsset))
                return
            }
            reportProgress(1.0, progress: progress)
            completion(.success(()))
        }
    }

    /// Однопроходное извлечение ВСЕХ кадров одним процессом ffmpeg.
    /// Кадры идут MJPEG-потоком через pipe и разбираются в памяти строго
    /// по порядку; прогресс — по фактически отданным кадрам.
    private func runSinglePass(
        plan: Plan,
        ffmpeg: String,
        onFrame: @escaping (_ index: Int, _ total: Int, _ ordinal: Int, _ jpeg: Data) -> Void,
        progress: @escaping (Double) -> Void,
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        var vf = String(format: "fps=%.4f", plan.outputFPS)
        if options.targetWidth > 0 && options.targetHeight > 0 {
            let scaleBar = "scale=\(options.targetWidth):\(options.targetHeight):force_original_aspect_ratio=decrease:flags=lanczos"
            vf = "\(vf),\(scaleBar)"
        }
        let args = [
            "-hide_banner", "-loglevel", "error",
            "-nostdin",
            "-threads", "12",
            "-i", url.path,
            "-vf", vf,
            // Жёсткий предел вывода: без него VFR-источник (screen recording
            // с заниженным номинальным fps) выдаёт заметно больше кадров, чем
            // plan.targetFrames; читатель останавливается на лимите, хвост
            // переполняет буфер пайпа, ffmpeg блокируется на записи —
            // waitUntilExit висит вечно.
            "-frames:v", "\(plan.targetFrames)",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "\(Self.jpegQuality)",
            "-vsync", "0",
            "-",
        ]
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: ffmpeg)
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        lock.lock()
        processes.append(proc)
        lock.unlock()
        do {
            try proc.run()
        } catch {
            completion(.failure(error))
            return
        }

        let progressStride = max(1, plan.targetFrames / 100)
        DispatchQueue(label: "yume.emit").async { [self] in
            var idx = 0
            var failure: Error?
            Self.readJpegs(handle: pipe.fileHandleForReading,
                           maxFrames: plan.targetFrames,
                           onCancel: { [self] in isCancelled },
                           onFrame: { [self] jpeg in
                               if isCancelled { return }
                               onFrame(idx, plan.targetFrames, idx + options.startIndex, jpeg)
                               idx += 1
                               if idx % progressStride == 0 {
                                   reportProgress(Double(idx) / Double(plan.targetFrames),
                                                  progress: progress)
                               }
                           })
            proc.waitUntilExit()
            if isCancelled {
                failure = FrameExtractError.cancelled
            } else if idx < 1 {
                failure = FrameExtractError.unreadableAsset
            }
            if let failure {
                completion(.failure(failure))
                return
            }
            reportProgress(1.0, progress: progress)
            completion(.success(()))
        }
    }

    /// Читает поток MJPEG и разбирает его на отдельные JPEG-кадры.
    /// Границы кадра — SOI (FFD8) и EOI (FFD9); буфер обрезается порциями,
    /// поэтому разбор не вырождается в квадратичный на больших сегментах.
    private static func readJpegs(handle: FileHandle,
                                  maxFrames: Int,
                                  onCancel: @escaping () -> Bool,
                                  onFrame: @escaping (Data) -> Void) {
        var count = 0
        var buffer = Data()
        var scan = 0
        // Защита от битого потока без SOI: без маркера буфер рос до конца
        // процесса (гигабайты RAM). Кадр в максимуме ~10 МБ — 64 МБ хватает
        // с многократным запасом.
        let maxBufferBytes = 64 * 1024 * 1024
        while count < maxFrames {
            if onCancel() { break }
            let chunk = handle.readData(ofLength: 65536)
            if chunk.isEmpty { break }
            buffer.append(chunk)
            while count < maxFrames {
                guard let start = findMarker(0xFFD8, in: buffer, from: scan) else {
                    scan = max(0, buffer.count - 1)
                    break
                }
                guard let end = findMarker(0xFFD9, in: buffer, from: start + 2) else {
                    scan = start
                    break
                }
                let frame = buffer.subdata(in: start..<(end + 2))
                if frame.count > 100 {
                    onFrame(frame)
                    count += 1
                }
                buffer.removeSubrange(0..<(end + 2))
                scan = 0
            }
            if buffer.count > maxBufferBytes { break }
        }
    }

    /// Ищет 2-байтовый маркер в Data, начиная с `from`, без копирования
    /// буфера в массив.
    private static func findMarker(_ marker: UInt16, in data: Data, from: Int) -> Int? {
        guard from + 1 < data.count else { return nil }
        let hi = UInt8(marker >> 8)
        let lo = UInt8(marker & 0xFF)
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int? in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return nil
            }
            var i = from
            while i + 1 < raw.count {
                if base[i] == hi && base[i + 1] == lo { return i }
                i += 1
            }
            return nil
        }
    }
}
