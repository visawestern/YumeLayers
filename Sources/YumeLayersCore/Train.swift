// YumeLayers Core — синтетические сцены и хранилище весов.
import Foundation

/// Генератор синтетических сцен: фон-градиент + фигуры + «свет» (одна линия).
public func makeSyntheticScene<R: RandomNumberGenerator>(w: Int, h: Int, rng: inout R) -> [Float] {
    let size = min(w, h)
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

/// Хранилище весов в общепринятом бинарном формате **safetensors**
/// (магия "<safeten" + 8 байт длины JSON-заголовка + JSON + сырые float32).
/// Без легаси-JSON: только бинарный формат.
public func saveState(_ mats: [Matrix], to path: String, epoch: Int? = nil,
                      step: Int? = nil, totalSteps: Int? = nil,
                      dataFingerprint: String? = nil) throws {
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
    public init(mats: [Matrix], epoch: Int, step: Int, fingerprint: String? = nil) {
        self.mats = mats
        self.epoch = epoch
        self.step = step
        self.fingerprint = fingerprint
    }
}

public func loadState(path: String) throws -> [Matrix] {
    try loadStateWithMeta(path: path).mats
}

public func loadStateWithMeta(path: String) throws -> CheckpointState {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard data.count >= 16, data.prefix(8).elementsEqual(Data("<safeten".utf8)) else {
        throw NSError(domain: "Yume", code: 9,
                      userInfo: [NSLocalizedDescriptionKey:
                        "чекпоинт не в формате safetensors"])
    }
    let rawHeaderLen = data.withUnsafeBytes {
        UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self))
    }
    guard rawHeaderLen <= UInt64(data.count - 16) else {
        throw NSError(domain: "Yume", code: 9, userInfo: [NSLocalizedDescriptionKey: "чекпоинт повреждён (длина)"])
    }
    let headerLen = Int(rawHeaderLen)
    guard
          let obj = try? JSONSerialization.jsonObject(with: data.subdata(in: 16..<(16 + headerLen)))
              as? [String: Any] else {
        throw NSError(domain: "Yume", code: 9,
                      userInfo: [NSLocalizedDescriptionKey: "чекпоинт повреждён (заголовок)"])
    }

    // Метаданные (эпоха/шаг/отпечаток данных), если есть — для продолжения
    // с места срыва и проверки, что данные те же.
    var metaEpoch = 0
    var metaStep = 0
    var metaFingerprint: String? = nil
    if let meta = obj["__metadata__"] as? [String: Any] {
        metaEpoch = (meta["epoch"] as? NSNumber)?.intValue ?? 0
        metaStep = (meta["step"] as? NSNumber)?.intValue ?? 0
        metaFingerprint = meta["dataFingerprint"] as? String
    }

    struct Entry {
        let name: String
        let rows: Int
        let cols: Int
        let start: Int
        let end: Int
    }
    // Сдвиги в safetensors — относительно начала секции данных (после магии
    // 8 Б + длины заголовка 8 Б + самого заголовка).
    let baseOffset = 16 + headerLen
    var entries: [Entry] = []
    for (k, v) in obj where k != "__metadata__" {
        guard let d = v as? [String: Any],
              let shape = d["shape"] as? [NSNumber], shape.count == 2,
              let off = d["data_offsets"] as? [NSNumber], off.count == 2,
              d["dtype"] as? String == "F32" else {
            throw NSError(domain: "Yume", code: 9)
        }
        entries.append(Entry(name: k,
                             rows: Int(shape[0].intValue),
                             cols: Int(shape[1].intValue),
                             start: Int(off[0].intValue),
                             end: Int(off[1].intValue)))
    }
    entries.sort { $0.start < $1.start }

    var result: [Matrix] = []
    for e in entries {
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
        result.append(Matrix(floats, rows: e.rows))
    }
    return CheckpointState(mats: result, epoch: metaEpoch, step: metaStep,
                           fingerprint: metaFingerprint)
}
