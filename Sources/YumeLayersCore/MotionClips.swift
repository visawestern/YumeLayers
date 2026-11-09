// YumeLayers — фундамент Фазы 3: клипы движения.
//
// Для обучения диффузии в латентном пространстве нужны временные окна:
// последовательные кадры из ОДНОГО источника со строгим порядком. Здесь —
// чистые функции планирования клипов и сериализации манифеста (JSON).
import Foundation

/// Ссылка на извлечённый кадр: источник (имя видеофайла) + индекс внутри него.
public struct ExtractedFrameRef: Equatable, Sendable {
    public let source: String
    public let index: Int
    public init(source: String, index: Int) {
        self.source = source
        self.index = index
    }
}

/// Непрерывное окно кадров одного источника (клип движения).
public struct MotionClip: Equatable, Sendable {
    public let source: String
    public let startIndex: Int
    public let frameCount: Int

    public init(source: String, startIndex: Int, frameCount: Int) {
        self.source = source
        self.startIndex = startIndex
        self.frameCount = frameCount
    }

    /// Индексы кадров внутри источника, строго по возрастанию.
    public var frameIndices: [Int] {
        (0..<frameCount).map { startIndex + $0 }
    }
}

/// Разбивает последовательность извлечённых кадров на окна-клипы:
/// - окна никогда не пересекают границу источника;
/// - внутри серии одного источника окна идут с шагом `stride`
///   (по умолчанию 1 — полное перекрытие);
/// - если `window <= 0`, берётся вся серия одним клипом.
public func motionClips(from refs: [ExtractedFrameRef],
                        window: Int,
                        stride: Int = 1) -> [MotionClip] {
    let step = max(1, stride)
    var clips: [MotionClip] = []
    var start = 0
    while start < refs.count {
        let source = refs[start].source
        var end = start + 1
        while end < refs.count, refs[end].source == source {
            end += 1
        }
        let runLen = end - start
        let w = window > 0 ? min(window, runLen) : runLen
        let base = refs[start].index
        var s = 0
        while s + w <= runLen {
            clips.append(MotionClip(source: source,
                                    startIndex: base + s,
                                    frameCount: w))
            s += step
        }
        start = end
    }
    return clips
}

/// JSON-манифест клипов для диска.
/// Формат: {"version":1,"clips":[{"source":...,"startIndex":...,"frameCount":...}]}
public func motionClipManifestJSON(clips: [MotionClip]) throws -> Data {
    var arr: [[String: Any]] = []
    for c in clips {
        arr.append([
            "source": c.source,
            "startIndex": c.startIndex,
            "frameCount": c.frameCount,
        ])
    }
    let root: [String: Any] = ["version": 1, "clips": arr]
    return try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
}

/// Обратное чтение манифеста. `nil` при повреждённых данных.
public func motionClips(fromManifest data: Data) throws -> [MotionClip]? {
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let arr = root["clips"] as? [[String: Any]] else {
        return nil
    }
    var clips: [MotionClip] = []
    for item in arr {
        guard let source = item["source"] as? String,
              let startIndex = item["startIndex"] as? Int,
              let frameCount = item["frameCount"] as? Int,
              startIndex >= 0, frameCount > 0 else {
            return nil
        }
        clips.append(MotionClip(source: source, startIndex: startIndex, frameCount: frameCount))
    }
    return clips
}
