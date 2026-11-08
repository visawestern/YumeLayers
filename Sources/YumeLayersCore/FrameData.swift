// YumeLayers — данные кадров: кэш и источник.
//
// Выделено из `ConvTrainer.swift` для соблюдения лимита 600 строк/файл.
import Foundation

/// Преддекодированный кэш кадров в RGB8, с фоновым преддекодированием
/// следующего чанка (чтобы обучение не ждало декод).
final class FrameCache {
    let w: Int
    let h: Int
    private let realW: Int
    private let realH: Int
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

    init?(frames: [MemoryFrame], w: Int, h: Int, realW: Int? = nil, realH: Int? = nil,
          budgetBytes: Int = 1 << 30) {
        guard !frames.isEmpty else { return nil }
        self.frames = frames
        self.w = w
        self.h = h
        // Канонический декод: истинный размер + padToMultipleOf8 (см. GPUFramePool).
        self.realW = realW ?? w
        self.realH = realH ?? h
        self.per = w * h * 3
        self._count = frames.count
        // Чанк ~1024 кадра: два буфера ≈ 2.5 ГБ — умещается в RAM и даёт
        // большой запас времени на фоновый декод следующего чанка.
        self.chunkSize = min(frames.count, max(1, min(1024, budgetBytes / max(2 * per, 1))))

        let firstEnd = min(chunkSize, frames.count)
        self.active = Self.decodeChunk(frames: frames, start: 0, end: firstEnd,
                                       w: w, h: h, realW: self.realW, realH: self.realH)
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
                end: min(desiredStart + chunkSize, _count), w: w, h: h,
                realW: realW, realH: realH)
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
        let ww = w, hh = h, rw = realW, rh = realH
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let chunk = Self.decodeChunk(frames: f, start: s, end: e, w: ww, h: hh,
                                         realW: rw, realH: rh)
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
                                    w: Int, h: Int, realW: Int, realH: Int) -> [[UInt8]] {
        let n = end - start
        guard n > 0 else { return [] }
        let lock = NSLock()
        var out = [[UInt8]](repeating: [], count: n)
        DispatchQueue.concurrentPerform(iterations: n) { j in
            let i = start + j
            // Декод в истинном размере + паддинг (канонический путь).
            guard i < frames.count,
                  let img = loadImageRGB(data: frames[i].jpeg, w: realW, h: realH)
                      .map({ padToMultipleOf8($0, w: realW, h: realH).0 }) else { return }
            var row = [UInt8](repeating: 0, count: img.count)
            for k in 0..<img.count {
                row[k] = UInt8(min(255, max(0, Int(img[k] * 255 + 0.5))))
            }
            lock.lock(); out[j] = row; lock.unlock()
        }
        return out
    }
}

/// Источник кадров: последовательная выборка (кэш → память → файлы → синтетика).
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
