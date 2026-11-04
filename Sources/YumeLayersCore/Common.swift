// YumeLayers Core — общие определения: параметры, PNG, синтетика.
import Foundation
import ImageIO
import UniformTypeIdentifiers
import CoreGraphics

// ---------- Чтение/запись изображений через ImageIO (нет внешних зависимостей) ----------

/// Кодирует RGB-массив (Float 0…1) в PNG-данные В ПАМЯТИ (без записи на диск).
/// Возвращает Data или nil при ошибке. Чистая, тестируемая функция.
public func encodePNG(width: Int, height: Int, rgb: [Float]) -> Data? {
    guard width > 0, height > 0, rgb.count >= width * height * 3 else { return nil }
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let si = (y * width + x) * 3
            let di = (y * width + x) * 4
            pixels[di] = UInt8(min(255, max(0, rgb[si] * 255)))
            pixels[di+1] = UInt8(min(255, max(0, rgb[si+1] * 255)))
            pixels[di+2] = UInt8(min(255, max(0, rgb[si+2] * 255)))
            pixels[di+3] = 255
        }
    }
    let cs = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                        bytesPerRow: width * 4, space: cs,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    guard let img = ctx.makeImage() else { return nil }
    let data = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
    CGImageDestinationAddImage(dest, img, nil)
    guard CGImageDestinationFinalize(dest) else { return nil }
    return data as Data
}

public func savePNG(path: String, width: Int, height: Int, rgb: [Float]) throws {
    guard let data = encodePNG(width: width, height: height, rgb: rgb) else {
        throw NSError(domain: "Yume", code: 2)
    }
    try data.write(to: URL(fileURLWithPath: path))
}

/// Шина журнала: core-код пишет через yumeLog, UI подписывается и показывает
/// те же строки в журнале приложения (одна точка вывода отладки).
public final class YumeLogBus {
    public static let shared = YumeLogBus()
    private let lock = NSLock()
    private var sinks: [(String) -> Void] = []
    private init() {}

    public func subscribe(_ sink: @escaping (String) -> Void) {
        lock.lock(); sinks.append(sink); lock.unlock()
    }

    func publish(_ s: String) {
        lock.lock(); let list = sinks; lock.unlock()
        for sink in list { sink(s) }
    }
}

/// Лог в консоль + файл ~/YumeLayers/train.log + журнал UI (через YumeLogBus).
public func yumeLog(_ s: String) {
    let fmt = DateFormatter()
    fmt.dateFormat = "HH:mm:ss"
    let stamped = "[\(fmt.string(from: Date()))] \(s)"
    print(stamped)
    fflush(stdout)
    let line = stamped + "\n"
    let url = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("YumeLayers", isDirectory: true)
        .appendingPathComponent("train.log")
    if let handle = try? FileHandle(forWritingTo: url) {
        handle.seekToEndOfFile()
        handle.write(Data(line.utf8))
        try? handle.close()
    } else {
        try? line.write(to: url, atomically: true, encoding: .utf8)
    }
    YumeLogBus.shared.publish(stamped)
}

/// Синтетическая сцена для проверок и обучения без реальных кадров.
///
/// RNG зафиксирован: иначе `convcheck`/`convprofile` давали разные числа от
/// прогона к прогону, и «стало лучше / стало хуже» нельзя было сравнить.
/// `makeSyntheticScene(rng:)` остаётся доступным для случайных сцен.
public func makeNormalSynthetic(size: Int) -> [Float] {
    var rng = SplitMix64(seed: 0xC0FFEE_1234_5678)
    return makeSyntheticScene(size: size, rng: &rng)
}

public func makeNormalSynthetic(w: Int, h: Int) -> [Float] {
    var rng = SplitMix64(seed: 0xC0FFEE_1234_5678)
    return makeSyntheticScene(w: w, h: h, rng: &rng)
}

// ---------- Параметры ----------
public struct Params {
    public init() {
        // Базовый каталог пользователя — чтобы приложение работало из Finder и из терминала
        let base = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("YumeLayers").path
        vaePath = base + "/checkpoints/vae.safetensors"
        diffPath = base + "/checkpoints/diff.safetensors"
        convPath = base + "/checkpoints/conv.safetensors"
        outPrefix = base + "/outputs/gen"
    }
    public var imageSize = 32
    public var width = 854
    public var height = 480
    public var latentDim = 16
    public var epochs = 20
    public var steps = 20
    public var batch = 1
    public var lr: Float = 0.0003
    public var noiseSteps = 80
    public var vaePath: String
    public var diffPath: String
    public var convPath: String
    public var dataDir: String? = nil
    /// Кадры в памяти (JPEG-байты), если извлечение было без записи на диск.
    /// Приоритетнее `dataDir` — если задано, обучение/реконструкции читают из памяти.
    public var memoryFrames: [MemoryFrame]? = nil
    /// Контрольная выборка для визуальной оценки качества: 1 кадр на секунду
    /// видео (каждый round(fps)-й). Держится в памяти отдельно и не освобождается
    /// по ходу обучения. После обучения по ней идёт реконструкция (итоговая модель).
    public var controlFrames: [MemoryFrame]? = nil
    public var resume = false
    public var pauseMs = 250
    public var maxPauseMs = 2000
    public var adaptiveThrottle = true
    /// Потолок нагрузки GPU в процентах от безопасного максимума.
    /// 100 — работать на пределе, сколько карта держит по температуре.
    /// Фактическую скважность всегда задаёт контур: он держит 85 °C
    /// и не даёт уйти выше 89 °C (перезагрузка случалась на 93 °C).
    public var gpuPowerPercent = 100
    public var outPrefix: String
}

/// Кадр, извлечённый в память (без записи на диск).
/// `name` повторяет старое имя файла (frame_XXXX.jpg), чтобы реконструкции
/// с тем же номером можно было сопоставлять по имени.
/// `source` — метка источника (имя видеофайла): кадры одного источника
/// образуют клип движения со строгим порядком.
public struct MemoryFrame {
    public let name: String
    public var jpeg: Data
    public let source: String
    public init(name: String, jpeg: Data, source: String = "") {
        self.name = name
        self.jpeg = jpeg
        self.source = source
    }
}
