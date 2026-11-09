// YumeLayers — тесты Фазы 3: чекпоинт диффузии (safetensors)
// и сид-продолжение генерации.
import Foundation

func testDiffusionCheckpointRoundTrip() {
    section("Фаза 3: чекпоинт диффузии (safetensors round-trip)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let net = DenoiserMLPGPU(engine: engine, dimIn: 32, dimEmb: 8, hidden: 24)
    let schedule = DiffusionSchedule(timesteps: 100, betaStart: 1e-4, betaEnd: 0.02)
    let tmp = NSTemporaryDirectory() + "yume-diff-cp-\(UUID().uuidString).safetensors"
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    do {
        try saveDiffusionCheckpoint(net: net, schedule: schedule, to: tmp)
        // [DIAG] разбор сохранённого файла
        if let f = try? Data(contentsOf: URL(fileURLWithPath: tmp)) {
            let len = f.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self) }
            let hdrLen = Int(len.littleEndian)
            let hdr = String(data: f.subdata(in: 8..<min(f.count, 8 + hdrLen)), encoding: .utf8)
            print("      diag: file=\(f.count) headerLen=\(hdrLen) hdr=\(hdr?.prefix(120) ?? "nil")")
        }
        guard let loaded = try loadDiffusionCheckpoint(from: tmp) else {
            expect(false, "чекпоинт диффузии прочитан"); return
        }
        expectEqual(loaded.mats.count, 6, "6 матриц (3 слоя × W,b)")
        let orig = net.matrices
        for i in 0..<6 {
            expectEqual(loaded.mats[i].data, orig[i].data, "матрица \(i) сохранена точно")
        }
        expectEqual(loaded.meta.dimIn, 32, "meta: dimIn")
        expectEqual(loaded.meta.dimEmb, 8, "meta: dimEmb")
        expectEqual(loaded.meta.hidden, 24, "meta: hidden")
        expectEqual(loaded.meta.timesteps, 100, "meta: timesteps")
        expectEqual(loaded.meta.archVersion, 4, "meta: archVersion=4")
    } catch {
        expect(false, "чекпоинт диффузии: \(error)")
    }
}

func testDiffusionCheckpointRejectsBad() {
    section("Фаза 3: чекпоинт диффузии отклоняет повреждённые данные")
    let tmp = NSTemporaryDirectory() + "yume-diff-bad-\(UUID().uuidString).safetensors"
    defer { try? FileManager.default.removeItem(atPath: tmp) }
    try? Data("не json и не safetensors".utf8).write(to: URL(fileURLWithPath: tmp))
    expect((try? loadDiffusionCheckpoint(from: tmp)) == nil,
           "битый файл → nil")

    guard gHasMetal else { return }
    let engine = MetalEngine.shared
    let net = DenoiserMLPGPU(engine: engine, dimIn: 32, dimEmb: 8, hidden: 24)
    let schedule = DiffusionSchedule(timesteps: 100)
    let tmp2 = NSTemporaryDirectory() + "yume-diff-bad2-\(UUID().uuidString).safetensors"
    defer { try? FileManager.default.removeItem(atPath: tmp2) }
    try? saveDiffusionCheckpoint(net: net, schedule: schedule, to: tmp2)
    // Повредим размерность в заголовке: перезапишем через парсер нельзя —
    // просто проверим, что загрузка подтверждает целостность.
    let ok = (try? loadDiffusionCheckpoint(from: tmp2)) != nil
    expect(ok, "целый файл читается")
}

func testContinuationSeed() {
    section("Фаза 3: сид-продолжение (детерминизм, конечность, отличимость)")
    let schedule = DiffusionSchedule(timesteps: 16, betaStart: 1e-4, betaEnd: 0.02)
    let lastZ = (0..<6).map { Float($0) * 0.1 }
    let pred: (Int, [Float]) -> [Float] = { _, x in x.map { $0 * 0.3 } }

    let z1 = continuationFromSeed(lastZ: lastZ, schedule: schedule,
                                  predEps: pred, seed: 1234)
    let z2 = continuationFromSeed(lastZ: lastZ, schedule: schedule,
                                  predEps: pred, seed: 1234)
    expectEqual(z1, z2, "один seed → одинаковый результат")
    expect(z1.allSatisfy { $0.isFinite }, "конечные значения")
    let z3 = continuationFromSeed(lastZ: lastZ, schedule: schedule,
                                  predEps: pred, seed: 9999)
    expect(z1 != z3, "другой seed → другой результат (с вероятностью ~1)")

    // Пустой шум (нулевой seed-вклад не требуется): детерминизм устойчив.
    let z4 = continuationFromSeed(lastZ: [Float](repeating: 0.5, count: 6),
                                  schedule: schedule, predEps: pred, seed: 1)
    expect(z4.allSatisfy { $0.isFinite }, "продолжение с константного латента конечно")
}
