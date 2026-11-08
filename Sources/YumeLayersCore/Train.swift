// YumeLayers Core — синтетические сцены и хранилище весов.
import Foundation

/// Генератор синтетических сцен: фон-градиент + фигуры + «свет» (одна линия).
public func makeSyntheticScene<R: RandomNumberGenerator>(w: Int, h: Int, rng: inout R) -> [Float] {
    var img = [Float](repeating: 0, count: w * h * 3)
    let r = Float.random(in: 0.3...0.9, using: &rng)
    let g = Float.random(in: 0.3...0.9, using: &rng)
    let b = Float.random(in: 0.3...0.9, using: &rng)
    for y in 0..<h {
        let t = Float(y) / Float(max(h - 1, 1))
        for x in 0..<w {
            let i = (y * w + x) * 3
            img[i] = r * (1 - t); img[i+1] = g * (1 - t); img[i+2] = b * t
            // фигура: круг в центре
            let dx = Float(x) / Float(w) - 0.5
            let dy = Float(y) / Float(h) - 0.5
            if dx*dx + dy*dy < 0.18 {
                img[i] = 1.0; img[i+1] = 0.3; img[i+2] = 0.3
            }
            // «свет»: полоса сверху
            if y < h / 8 { img[i] = min(1.0, img[i] * 1.5) }
        }
    }
    return img
}

public func makeSyntheticScene<R: RandomNumberGenerator>(size: Int, rng: inout R) -> [Float] {
    return makeSyntheticScene(w: size, h: size, rng: &rng)
}

/// Состояние оптимизатора Adam для честного продолжения обучения.
///
/// Без него продолжение обучения РАЗРУШАЕТ модель. Adam при старте имеет
/// нулевые моменты m = v = 0 и счётчик t = 1, а его коррекция смещения даёт
/// первый шаг величиной ±lr по КАЖДОМУ весу (знак градиента), независимо от
/// величины градиента. Для обучения с нуля это нормально, для продолжения
/// обученной модели — это удар, который стирает всё, что было выучено.
public struct CheckpointOptimizerState {
    /// Счётчик шагов Adam — по нему ядро считает коррекцию смещения.
    public let step: Int
    /// Моменты Adam: для каждой матрицы весов (в том же порядке, что `mats`)
    /// сначала `m`, потом `v`.
    public let moments: [Matrix]
    public init(step: Int, moments: [Matrix]) {
        self.step = step
        self.moments = moments
    }
}

/// Хранилище весов в общепринятом бинарном формате **safetensors**
/// (магия "<safeten" + 8 байт длины JSON-заголовка + JSON + сырые float32).
/// Без легаси-JSON: только бинарный формат.
public func saveState(_ mats: [Matrix], to path: String, epoch: Int? = nil,
                      step: Int? = nil, totalSteps: Int? = nil,
                      dataFingerprint: String? = nil,
                      optimizer: CheckpointOptimizerState? = nil,
                      modelWidth: Int? = nil) throws {
    var dict: [String: Any] = [:]
    var offset = 0
    var order: [String] = []
    for (i, m) in mats.enumerated() {
        guard m.data.allSatisfy({ $0.isFinite }) else {
            throw NSError(domain: "Yume", code: 7,
                          userInfo: [NSLocalizedDescriptionKey:
                            "NaN/Inf в весах p\(i) — обучение нестабильно, чекпоинт не сохранён"])
        }
        let name = "p\(i)"
        order.append(name)
        let bytes = m.data.count * 4
        dict[name] = ["dtype": "F32",
                      "shape": [m.rows, m.cols],
                      "data_offsets": [offset, offset + bytes]]
        offset += bytes
    }
    var meta: [String: Any] = ["format": "YumeLayers VAE", "order": order]
    if let e = epoch { meta["epoch"] = e }
    if let s = step { meta["step"] = s }
    if let t = totalSteps { meta["totalSteps"] = t }
    // Отпечаток обучающих данных: позволяет на resume понять, что чекпоинт
    // был обучен на ТЕХ ЖЕ кадрах, а не на другом файле.
    if let fp = dataFingerprint { meta["dataFingerprint"] = fp }
    if let w = modelWidth { meta["modelWidth"] = w }
    // Моменты Adam — отдельными тензорами "adam.mN"/"adam.vN", чтобы старые
    // чекпоинты (без них) продолжали читаться как есть.
    var momentData: [[Float]] = []
    if let opt = optimizer {
        guard opt.moments.count % 2 == 0 else {
            throw NSError(domain: "Yume", code: 12,
                          userInfo: [NSLocalizedDescriptionKey:
                            "состояние оптимизатора повреждено: моменты должны идти парами m/v"])
        }
        for (i, m) in opt.moments.enumerated() {
            guard m.data.allSatisfy({ $0.isFinite }) else {
                throw NSError(domain: "Yume", code: 12,
                              userInfo: [NSLocalizedDescriptionKey:
                                "NaN/Inf в моментах оптимизатора (#\(i)) — чекпоинт не сохранён"])
            }
            let pair = i / 2
            let name = (i % 2 == 0) ? "adam.m\(pair)" : "adam.v\(pair)"
            let bytes = m.data.count * 4
            dict[name] = ["dtype": "F32",
                          "shape": [m.rows, m.cols],
                          "data_offsets": [offset, offset + bytes]]
            offset += bytes
            momentData.append(m.data)
        }
        meta["optimizerStep"] = opt.step
    }
    dict["__metadata__"] = meta
    let header = try JSONSerialization.data(withJSONObject: dict)

    var out = Data()
    out.append(Data("<safeten".utf8))                 // 8 байт магии
    var headerLen = UInt64(header.count)
    withUnsafeBytes(of: &headerLen) { out.append(contentsOf: $0) }
    out.append(header)
    for m in mats {
        m.data.withUnsafeBytes { out.append(contentsOf: $0) }
    }
    for m in momentData {
        m.withUnsafeBytes { out.append(contentsOf: $0) }
    }
    try out.write(to: URL(fileURLWithPath: path), options: .atomic)
}

/// Чекпоинт: веса + позиция обучения (эпоха/шаг) + отпечаток данных
/// для честного resume.
public struct CheckpointState {
    public let mats: [Matrix]
    public let epoch: Int
    public let step: Int
    /// Отпечаток кадров, на которых обучались эти веса (nil, если чекпоинт
    /// старого формата — тогда безопасно считать, что данные неизвестны).
    public let fingerprint: String?
    /// Состояние оптимизатора Adam (nil — чекпоинт старого формата, без него).
    public let optimizer: CheckpointOptimizerState?
    /// Кратность ширины свёрточных каналов модели (1 — базовая).
    public let modelWidth: Int
    public init(mats: [Matrix], epoch: Int, step: Int, fingerprint: String? = nil,
                optimizer: CheckpointOptimizerState? = nil, modelWidth: Int = 1) {
        self.mats = mats
        self.epoch = epoch
        self.step = step
        self.fingerprint = fingerprint
        self.optimizer = optimizer
        self.modelWidth = modelWidth
    }
}

public func loadState(path: String) throws -> [Matrix] {
    try loadStateWithMeta(path: path).mats
}

/// Разбор заголовка safetensors: магия + длина JSON + объект заголовка.
private func parseCheckpointHeader(_ data: Data) throws -> (headerLen: Int, obj: [String: Any]) {
    guard data.count >= 16, data.prefix(8).elementsEqual(Data("<safeten".utf8)) else {
        throw NSError(domain: "Yume", code: 9,
                      userInfo: [NSLocalizedDescriptionKey: "чекпоинт не в формате safetensors"])
    }
    let rawHeaderLen = data.withUnsafeBytes {
        UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self))
    }
    guard rawHeaderLen <= UInt64(data.count - 16) else {
        throw NSError(domain: "Yume", code: 9,
                      userInfo: [NSLocalizedDescriptionKey: "чекпоинт повреждён (длина)"])
    }
    let headerLen = Int(rawHeaderLen)
    guard let obj = try? JSONSerialization.jsonObject(
        with: data.subdata(in: 16..<(16 + headerLen))) as? [String: Any] else {
        throw NSError(domain: "Yume", code: 9,
                      userInfo: [NSLocalizedDescriptionKey: "чекпоинт повреждён (заголовок)"])
    }
    return (headerLen, obj)
}

/// Один тензор в заголовке safetensors.
private struct CheckpointEntry {
    let name: String
    let rows: Int
    let cols: Int
    let start: Int
    let end: Int
}

/// Сборка записей тензоров из заголовка, отсортированных по смещению.
private func parseCheckpointEntries(_ obj: [String: Any],
                                   baseOffset: Int) throws -> [CheckpointEntry] {
    var entries: [CheckpointEntry] = []
    for (k, v) in obj where k != "__metadata__" {
        guard let d = v as? [String: Any],
              let shape = d["shape"] as? [NSNumber], shape.count == 2,
              let off = d["data_offsets"] as? [NSNumber], off.count == 2,
              d["dtype"] as? String == "F32" else {
            throw NSError(domain: "Yume", code: 9)
        }
        entries.append(CheckpointEntry(name: k,
                                       rows: Int(shape[0].intValue),
                                       cols: Int(shape[1].intValue),
                                       start: Int(off[0].intValue),
                                       end: Int(off[1].intValue)))
    }
    entries.sort { $0.start < $1.start }
    return entries
}

/// Декод матрицы из секции данных по записи.
private func decodeCheckpointMatrix(_ e: CheckpointEntry,
                                   data: Data,
                                   baseOffset: Int) throws -> Matrix {
    let (n, overflow) = e.rows.multipliedReportingOverflow(by: e.cols)
    guard e.rows > 0, e.cols > 0, !overflow, n <= (data.count - baseOffset) / 4,
          e.start >= 0, e.end >= e.start, e.end <= data.count - baseOffset,
          e.end - e.start == n * 4 else {
        throw NSError(domain: "Yume", code: 9,
                      userInfo: [NSLocalizedDescriptionKey: "чекпоинт повреждён (данные)"])
    }
    let lo = baseOffset + e.start
    let floats = data.withUnsafeBytes { buf -> [Float] in
        (0..<n).map { buf.loadUnaligned(fromByteOffset: lo + $0 * 4, as: Float.self) }
    }
    guard floats.allSatisfy({ $0.isFinite }) else { throw NSError(domain: "Yume", code: 9) }
    return Matrix(floats, rows: e.rows)
}

/// Восстановление состояния оптимизатора из моментных записей adam.mN/adam.vN.
private func parseOptimizerMoments(_ momentEntries: [CheckpointEntry],
                                  data: Data,
                                  baseOffset: Int,
                                  step: Int) throws -> CheckpointOptimizerState? {
    guard !momentEntries.isEmpty else { return nil }
    var pairM: [Int: Matrix] = [:]
    var pairV: [Int: Matrix] = [:]
    for e in momentEntries {
        let matrix = try decodeCheckpointMatrix(e, data: data, baseOffset: baseOffset)
        if e.name.hasPrefix("adam.m") {
            guard let idx = Int(e.name.dropFirst("adam.m".count)) else {
                throw NSError(domain: "Yume", code: 12,
                              userInfo: [NSLocalizedDescriptionKey: "неверное имя момента оптимизатора: \(e.name)"])
            }
            pairM[idx] = matrix
        } else if e.name.hasPrefix("adam.v") {
            guard let idx = Int(e.name.dropFirst("adam.v".count)) else {
                throw NSError(domain: "Yume", code: 12,
                              userInfo: [NSLocalizedDescriptionKey: "неверное имя момента оптимизатора: \(e.name)"])
            }
            pairV[idx] = matrix
        } else {
            throw NSError(domain: "Yume", code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "неверное имя момента оптимизатора: \(e.name)"])
        }
    }
    var flat: [Matrix] = []
    for idx in pairM.keys.sorted() {
        guard let m = pairM[idx], let v = pairV[idx] else {
            throw NSError(domain: "Yume", code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "состояние оптимизатора повреждено: пара \(idx) неполная"])
        }
        flat.append(m)
        flat.append(v)
    }
    return CheckpointOptimizerState(step: step, moments: flat)
}

public func loadStateWithMeta(path: String) throws -> CheckpointState {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let (headerLen, obj) = try parseCheckpointHeader(data)
    let baseOffset = 16 + headerLen

    var metaEpoch = 0
    var metaStep = 0
    var metaFingerprint: String? = nil
    var metaOptimizerStep = 0
    var metaModelWidth = 1
    if let meta = obj["__metadata__"] as? [String: Any] {
        metaEpoch = (meta["epoch"] as? NSNumber)?.intValue ?? 0
        metaStep = (meta["step"] as? NSNumber)?.intValue ?? 0
        metaFingerprint = meta["dataFingerprint"] as? String
        metaOptimizerStep = (meta["optimizerStep"] as? NSNumber)?.intValue ?? 0
        metaModelWidth = max(1, (meta["modelWidth"] as? NSNumber)?.intValue ?? 1)
    }

    let entries = try parseCheckpointEntries(obj, baseOffset: baseOffset)
    var weightEntries: [CheckpointEntry] = []
    var momentEntries: [CheckpointEntry] = []
    for e in entries {
        if e.name.hasPrefix("adam.") { momentEntries.append(e) } else { weightEntries.append(e) }
    }

    var result: [Matrix] = []
    for e in weightEntries {
        result.append(try decodeCheckpointMatrix(e, data: data, baseOffset: baseOffset))
    }
    let optimizer = try parseOptimizerMoments(momentEntries, data: data,
                                              baseOffset: baseOffset,
                                              step: metaOptimizerStep)
    return CheckpointState(mats: result, epoch: metaEpoch, step: metaStep,
                           fingerprint: metaFingerprint, optimizer: optimizer,
                           modelWidth: metaModelWidth)
}
