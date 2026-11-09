// YumeLayers — Фаза 3: чекпоинт диффузии в формате safetensors.
// Формат идентичен ядру: 8-байтовый little-endian размер заголовка, JSON,
// затем сырые float32. Добавлены метаданные архитектуры денойзера и
// расписания шума.
import Foundation

/// Метаданные чекпоинта диффузии.
public struct DiffusionCheckpointMeta: Equatable, Sendable {
    public let dimIn: Int
    public let dimEmb: Int
    public let hidden: Int
    public let timesteps: Int
    public let betaStart: Double
    public let betaEnd: Double
    public let archVersion: Int

    public init(dimIn: Int, dimEmb: Int, hidden: Int, timesteps: Int,
                betaStart: Double, betaEnd: Double, archVersion: Int = 4) {
        self.dimIn = dimIn
        self.dimEmb = dimEmb
        self.hidden = hidden
        self.timesteps = timesteps
        self.betaStart = betaStart
        self.betaEnd = betaEnd
        self.archVersion = archVersion
    }
}

/// Сохранение чекпоинта диффузии (матрицы денойзера + метаданные).
func saveDiffusionCheckpoint(net: DenoiserMLPGPU,
                             schedule: DiffusionSchedule,
                             to path: String) throws {
    let mats = net.matrices
    var dict: [String: Any] = [:]
    var offset = 0
    var order: [String] = []
    for (i, m) in mats.enumerated() {
        guard m.data.allSatisfy({ $0.isFinite }) else {
            throw NSError(domain: "YumeDiff", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "NaN/Inf в весах денойзера p\(i)"])
        }
        let name = "p\(i)"
        order.append(name)
        let bytes = m.data.count * 4
        dict[name] = ["dtype": "F32",
                      "shape": [m.rows, m.cols],
                      "data_offsets": [offset, offset + bytes]]
        offset += bytes
    }
    dict["format"] = "YumeLayers Diffusion"
    dict["order"] = order
    dict["dimIn"] = net.dimIn
    dict["dimEmb"] = net.dimEmb
    dict["hidden"] = net.hidden
    dict["timesteps"] = schedule.timesteps
    dict["betaStart"] = schedule.betas.first ?? 0
    dict["betaEnd"] = schedule.betas.last ?? 0
    dict["archVersion"] = 4
    let header = try JSONSerialization.data(withJSONObject: dict)
    var headerLen = UInt64(header.count).littleEndian
    var file = Data()
    file.append(Data(bytes: &headerLen, count: 8))
    file.append(header)
    for (i, m) in mats.enumerated() {
        _ = i
        let bytes = m.data.map { $0.bitPattern }
        var block = Data()
        for b in bytes {
            var v = b.littleEndian
            block.append(Data(bytes: &v, count: 4))
        }
        file.append(block)
    }
    try file.write(to: URL(fileURLWithPath: path))
}

/// Результат чтения чекпоинта диффузии.
public struct DiffusionCheckpoint {
    public let mats: [Matrix]
    public let meta: DiffusionCheckpointMeta
}

/// Чтение чекпоинта диффузии. nil при повреждённых данных.
public func loadDiffusionCheckpoint(from path: String) throws -> DiffusionCheckpoint? {
    let url = URL(fileURLWithPath: path)
    guard let file = try? Data(contentsOf: url), file.count >= 8 else { return nil }
    let headerLen = file.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self) }
    let headerSize = Int(headerLen.littleEndian)
    guard headerSize > 0, 8 + headerSize <= file.count else { return nil }
    let headerData = file.subdata(in: 8..<(8 + headerSize))
    guard let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any],
          (obj["format"] as? String) == "YumeLayers Diffusion",
          let order = obj["order"] as? [String] else { return nil }
    var mats: [Matrix] = []
    for name in order {
        guard let info = obj[name] as? [String: Any],
              let shape = info["shape"] as? [Int], shape.count == 2,
              let offs = info["data_offsets"] as? [Int], offs.count == 2 else { return nil }
        let rows = shape[0], cols = shape[1]
        let start = 8 + headerSize + offs[0]
        let end = 8 + headerSize + offs[1]
        guard end <= file.count, end > start else { return nil }
        let raw = file.subdata(in: start..<end)
        let count = rows * cols
        guard raw.count == count * 4 else { return nil }
        var data = [Float](repeating: 0, count: count)
        _ = raw.withUnsafeBytes { src in
            data.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: src)
            }
        }
        mats.append(Matrix(data, rows: rows))
    }
    guard let dimIn = obj["dimIn"] as? Int,
          let dimEmb = obj["dimEmb"] as? Int,
          let hidden = obj["hidden"] as? Int,
          let timesteps = obj["timesteps"] as? Int,
          let betaStart = obj["betaStart"] as? Double,
          let betaEnd = obj["betaEnd"] as? Double else { return nil }
    let v = (obj["archVersion"] as? Int) ?? 1
    return DiffusionCheckpoint(mats: mats,
                               meta: DiffusionCheckpointMeta(dimIn: dimIn,
                                                             dimEmb: dimEmb,
                                                             hidden: hidden,
                                                             timesteps: timesteps,
                                                             betaStart: betaStart,
                                                             betaEnd: betaEnd,
                                                             archVersion: v))
}
