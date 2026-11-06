// YumeLayers Core — проверка (тест) текущих кадров в памяти.
// Чистая функция: вход — кадры и лимит, выход — отчёт проверки.
// Декодирование JPEG инъекцией (decode), чтобы функцию можно было
// тестировать без ImageIO и реальных JPEG.

import Foundation

/// Отчёт проверки кадров.
public struct FrameValidationReport {
    /// Сколько кадров запрошено лимитом (0 — все).
    public let requested: Int
    /// Сколько кадров фактически проверено (min(limit, count)).
    public let checked: Int
    /// Сколько кадров прошли проверку.
    public let passed: Int
    /// Сколько кадров не прошли.
    public let failed: Int
    /// Текстовые описания проблем (по одной на проблемный кадр).
    public let issues: [String]

    public init(requested: Int, checked: Int, passed: Int, failed: Int,
                issues: [String]) {
        self.requested = requested
        self.checked = checked
        self.passed = passed
        self.failed = failed
        self.issues = issues
    }
}

/// Проверяет кадры из памяти:
/// - у каждого кадра есть JPEG-сигнатура (SOI FFD8) и непустые данные;
/// - JPEG декодируется (decode);
/// - имя не пустое и не дублируется.
/// Возвращает отчёт; проверяется не больше `limit` кадров (0 — все).
/// Чистая функция — покрыта тестами.
public func frameValidation(frames: [MemoryFrame], limit: Int,
                            decode: (Data) -> Bool) -> FrameValidationReport {
    let requested = limit > 0 ? limit : frames.count
    let n = min(requested, frames.count)
    var issues: [String] = []
    var seenNames = Set<String>()
    for i in 0..<n {
        let f = frames[i]
        var problems: [String] = []
        let hasSOI = f.jpeg.count >= 4 && f.jpeg[0] == 0xFF && f.jpeg[1] == 0xD8
        if !hasSOI {
            problems.append("JPEG-сигнатура")
        } else if !decode(f.jpeg) {
            problems.append("декодирование")
        }
        if f.name.isEmpty {
            problems.append("пустое имя")
        } else if seenNames.contains(f.name) {
            problems.append("дубликат имени \(f.name)")
        }
        seenNames.insert(f.name)
        if !problems.isEmpty {
            issues.append("кадр \(i) (\(f.name)): " + problems.joined(separator: ", "))
        }
    }
    return FrameValidationReport(requested: requested,
                                 checked: n,
                                 passed: n - issues.count,
                                 failed: issues.count,
                                 issues: issues)
}
