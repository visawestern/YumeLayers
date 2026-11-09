// YumeLayers — тесты фундамента Фазы 3: клипы движения
// (метка источника + строгий порядок внутри клипа) и манифест.
import Foundation

func testMotionClipPlanner() {
    section("Фаза 3: планировщик клипов движения (окна по источникам)")

    // Пустой вход → нет клипов.
    expectEqual(motionClips(from: [], window: 4).count, 0,
                "пустой список → 0 клипов")

    // Один источник, 10 кадров, окно 4, шаг 2 → 4 окна: 0-3, 2-5, 4-7, 6-9.
    let refs = (0..<10).map { ExtractedFrameRef(source: "v1.mp4", index: $0) }
    let clips = motionClips(from: refs, window: 4, stride: 2)
    expectEqual(clips.count, 4, "10 кадров, окно 4, шаг 2 → 4 клипа")
    expect(clips.allSatisfy { $0.source == "v1.mp4" }, "все клипы из v1.mp4")
    expectEqual(clips[0].startIndex, 0, "клип 0: start=0")
    expectEqual(clips[0].frameIndices, [0, 1, 2, 3], "клип 0: кадры 0..3")
    expectEqual(clips[1].startIndex, 2, "клип 1: start=2 (шаг 2)")
    expectEqual(clips[3].frameIndices, [6, 7, 8, 9], "клип 3: кадры 6..9")

    // Окно больше длины серии → один клип по размеру серии.
    let short = motionClips(from: Array(refs.prefix(3)), window: 8)
    expectEqual(short.count, 1, "3 кадра < окна → 1 клип")
    expectEqual(short[0].frameIndices, [0, 1, 2], "клип покрывает все 3 кадра")

    // Смена источника → окна НЕ пересекают границу источника.
    // Индексы даём глобально уникальными (a: 0..4, b: 5..9).
    let mixed = (0..<5).map { ExtractedFrameRef(source: "a.mp4", index: $0) }
        + (5..<10).map { ExtractedFrameRef(source: "b.mp4", index: $0) }
    let mixedClips = motionClips(from: mixed, window: 3, stride: 1)
    expect(mixedClips.allSatisfy { clip in
        if clip.source == "a.mp4" {
            return clip.frameIndices.allSatisfy { $0 < 5 }
        } else {
            return clip.frameIndices.allSatisfy { $0 >= 5 }
        }
    }, "ни один клип не пересекает границу a/b")
    expect(mixedClips.contains { $0.source == "a.mp4" && $0.startIndex == 2 },
           "серия a: последнее окно начинается с 2 (5−3)")
    expect(mixedClips.contains { $0.source == "b.mp4" && $0.startIndex == 5 },
           "серия b: первое окно начинается с 5")

    // Детерминизм: два прогона дают идентичные планы.
    let again = motionClips(from: mixed, window: 3, stride: 1)
    expectEqual(again, mixedClips, "план детерминирован")

    // step=0 → безопасный дефолт 1.
    let zeroStride = motionClips(from: refs, window: 4, stride: 0)
    expectEqual(zeroStride.count, 7, "stride=0 → как stride=1 (7 окон в 10 кадрах)")
}

func testMotionClipManifest() {
    section("Фаза 3: манифест клипов (JSON round-trip)")
    let refs = (0..<12).map { ExtractedFrameRef(source: "clip.mov", index: $0) }
    let clips = motionClips(from: refs, window: 5, stride: 2)
    do {
        let data = try motionClipManifestJSON(clips: clips)
        let restored = try motionClips(fromManifest: data)
        expectEqual(restored, clips, "JSON round-trip сохраняет клипы")
        let text = String(data: data, encoding: .utf8) ?? ""
        expect(text.contains("\"source\""), "манифест содержит поле source")
        expect(text.contains("\"startIndex\""), "манифест содержит поле startIndex")
    } catch {
        expect(false, "манифест: \(error)")
    }
    expect((try? motionClips(fromManifest: Data("не json".utf8))) == nil,
           "битый JSON → nil")
    expect((try? motionClips(fromManifest: Data("{\"clips\":[{\"source\":1}]}".utf8))) == nil,
           "неверный тип полей → nil")
}

func testHTTPParserFuzz() {
    section("Стресс: фаззинг парсера LocalHTTPRequest (не падает)")
    var rng = SplitMix64(seed: 2026)
    let validHead = "POST /api/train HTTP/1.1\r\nHost: 127.0.0.1:17851\r\nContent-Length: 2\r\n\r\n{}"
    for _ in 0..<300 {
        let mode = Int.random(in: 0..<3, using: &rng)
        let bytes: [UInt8]
        switch mode {
        case 0:
            // случайные байты
            bytes = (0..<Int.random(in: 0..<1024, using: &rng))
                .map { _ in UInt8.random(in: 0...255, using: &rng) }
        case 1:
            // легальный заголовок + случайный «мусор» в теле
            var b = Array(validHead.utf8) + Array(repeating: UInt8(0), count: Int.random(in: 0..<64, using: &rng))
            if Int.random(in: 0..<2, using: &rng) == 0 {
                b.append(contentsOf: [0x00, 0xFF, 0x0D, 0x0A, 0x80])
            }
            bytes = b
        default:
            // случайные куски UTF-8
            let alphabet = Array("GET /api/status HTTP/1.1\r\nHost: localhost:17851\r\n\r\n{}".utf8)
            bytes = (0..<Int.random(in: 0..<512, using: &rng))
                .map { _ in alphabet[Int.random(in: 0..<alphabet.count, using: &rng)] }
        }
        // Никаких крашей: либо nil (нужно ещё / битый), либо текст.
        let result = try? LocalHTTPRequest.parse(Data(bytes))
        expect(result == nil || result is String, "фаззинг: безопасный результат")
    }
    // Лимит размера остаётся жёстким.
    expect((try? LocalHTTPRequest.parse(Data(repeating: 65, count: 65537))) == nil,
           "фаззинг: >65536 байт отклоняется")
}

func testParamContract() {
    section("API: контракт приведения параметров (JSON → поле/переключатель)")
    expectEqual(paramStringValue(NSNumber(value: 5)), "5", "int → строка без точки")
    expectEqual(paramStringValue(NSNumber(value: 5.5)), "5.500000", "float → 6 знаков")
    expectEqual(paramStringValue(true), "1", "bool true → '1'")
    expectEqual(paramStringValue("abc"), "abc", "строка → как есть")
    expect(paramStringValue([1, 2]) == nil, "массив → nil")
    expect(paramStringValue(NSNull()) == nil, "NULL → nil")

    expectEqual(paramBoolValue(true), true, "bool true")
    expectEqual(paramBoolValue(1), true, "1 → true")
    expectEqual(paramBoolValue(0), false, "0 → false")
    expectEqual(paramBoolValue("on"), true, "'on' → true")
    expectEqual(paramBoolValue("OFF"), false, "'OFF' → false")
    expect(paramBoolValue(2) == nil, "2 → nil (только 0/1)")
    expect(paramBoolValue("maybe") == nil, "'maybe' → nil")
    expect(paramBoolValue(1.5) == nil, "1.5 → nil")
}
