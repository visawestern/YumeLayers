// YumeLayers — резидентный VRAM-пул кадров (выделено из ConvTrainer.swift).
//
// Отдельный модуль для соблюдения лимита 600 строк/файл.
import Foundation
import Metal

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
    private let realW: Int
    private let realH: Int
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
          realW: Int? = nil, realH: Int? = nil,
          size: Int = 1024, onChunk: ((Int, Int) -> Void)? = nil) {
        guard !frames.isEmpty else { return nil }
        self.engine = engine
        self.frames = frames
        self.w = w
        self.h = h
        // Истинный размер кадра (до паддинга до кратного 8). Декод всегда
        // идёт в него, затем padToMultipleOf8: раньше кадр 854×480
        // растягивался интерполяцией в 856×480, и одна и та же выборка
        // выглядела по-разному в пуле VRAM и на CPU-пути (продление края).
        self.realW = realW ?? w
        self.realH = realH ?? h
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
    ///
    /// Вся операция — под замком: раньше `windowStart` читался после
    /// `lock.unlock()`, и параллельный `next()` мог между этими шагами
    /// перезагрузить окно — индекс `tensors[g - windowStart]` становился
    /// отрицательным (крэш), а две параллельные `loadWindow` интерливирали
    /// запись тензоров.
    func next() -> GPUTensor {
        lock.lock(); defer { lock.unlock() }
        let i = cursorGlobal
        cursorGlobal += 1

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
        // (без `!`: при OOM пользователь получал падение на разыменовании nil
        // без единого слова о том, чего не хватило)
        let staging = (0..<batch).map { _ in engine.takeStaging(frameBytes) }

        yumeLog("GPUFramePool: дозагрузка чанка \(s)..<\(e) — \(n) кадров, порциями по \(batch)")
        var j = 0
        while j < n {
            let jend = min(j + batch, n)
            let cnt = jend - j

            var imgs = [[Float]?](repeating: nil, count: cnt)
            let decodeLock = NSLock()
            DispatchQueue.concurrentPerform(iterations: cnt) { k in
                // Канонический путь: декод в истинном размере, затем
                // паддинг до кратного 8 (не растяжение интерполяцией).
                let image = loadImageRGB(data: frames[s + j + k].jpeg, w: realW, h: realH)
                    .map { padToMultipleOf8($0, w: realW, h: realH).0 }
                decodeLock.lock(); imgs[k] = image; decodeLock.unlock()
            }

            guard let cmd = engine.queue.makeCommandBuffer(),
                  let blit = cmd.makeBlitCommandEncoder() else {
                fatalError(allocationFailureMessage(
                    bytes: 0,
                    availableBytes: Int(engine.device.recommendedMaxWorkingSetSize)
                        - engine.device.currentAllocatedSize,
                    what: "command buffer дозагрузки окна пула"))
            }
            for k in 0..<cnt {
                if let img = imgs[k], img.count * 4 == frameBytes {
                    // Кадр в памяти — HWC; сеть ждёт CHW. Без этой конвертации
                    // каналы отправлялись перепутанными (см. hwcToChw).
                    let chw = hwcToChw(img)
                    chw.withUnsafeBytes { raw in
                        staging[k].contents().copyMemory(from: raw.baseAddress!, byteCount: frameBytes)
                    }
                } else {
                    // Битый кадр: раньше слот пула оставался с байтами ПРЕДЫДУЩЕГО
                    // окна — обучение молча шло по «фантому». Теперь это явный
                    // чёрный кадр + запись в журнал.
                    yumeLog("⚠️ GPUFramePool: кадр \(frames[s + j + k].name) не декодирован — чёрная заглушка")
                    memset(staging[k].contents(), 0, frameBytes)
                }
                blit.copy(from: staging[k], sourceOffset: 0, to: tensors[j + k].buf,
                          destinationOffset: 0, size: frameBytes)
            }
            blit.endEncoding()
            engine.waitForCompletion(cmd, what: "дозагрузка окна кадров \(s)..<\(e)")
            // Staging больше не нужен — возвращаем в пул (после завершения GPU).
            for b in staging { engine.putStaging(b) }
            j = jend
            onChunk?(j, n)
        }
        windowStart = s
        let mb = n * frameBytes / (1024 * 1024)
        yumeLog("GPUFramePool: чанк \(s)..<\(e) в VRAM (\(mb) МБ)")

        // JPEG must remain available when the cyclic cursor revisits this window.
    }
}
