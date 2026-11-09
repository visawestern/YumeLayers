// YumeLayers — тесты Фазы 3: GPU-генерация зашумлённых латентов
// (обучение денойзинга диффузии в латентном пространстве).
import Foundation

func testGPUForwardDiffuse() {
    section("Фаза 3: GPU x_t ≈ CPU forwardDiffuse")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let n = 4096
    let schedule = DiffusionSchedule(timesteps: 100, betaStart: 1e-4, betaEnd: 0.02)
    var rng = SplitMix64(seed: 555)
    let x0 = (0..<n).map { _ in Float.random(in: -2...2, using: &rng) }
    let noise = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }

    for t in [0, 17, 50, 99] {
        let x0T = GPUTensor(engine: engine, data: x0, rows: 1)
        let noiseT = GPUTensor(engine: engine, data: noise, rows: 1)
        session.begin()
        let xT = session.encodeDiffusedLatents(x0: x0T, t: t,
                                               schedule: schedule, noise: noiseT)
        _ = session.end()
        // Чтение ТОЛЬКО после end() — внутри сессии toArray гоняется с command buffer.
        let got = xT.xT.toArray()

        let expected = forwardDiffuse(x0: x0, t: t, schedule: schedule, noise: noise)
        let diff = maxAbsDiff(got, expected.xT)
        expect(diff < 1e-4,
               "t=\(t): GPU x_t ≈ CPU (max|Δ| = \(diff))")
        expectClose(xT.sqrtAlphaBar, expected.sqrtAlphaBar, 1e-9, "t=\(t): √ᾱ")
        expectClose(xT.sqrtOneMinus, expected.sqrtOneMinus, 1e-9, "t=\(t): √(1−ᾱ)")
    }
}

func testGPUForwardDiffuseProperties() {
    section("Фаза 3: свойства x_t на GPU (ограниченность, монотонность шума)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let n = 2048
    var rng = SplitMix64(seed: 42)
    let x0 = [Float](repeating: 1.0, count: n)
    let noise = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }
    let schedule = DiffusionSchedule(timesteps: 50, betaStart: 1e-4, betaEnd: 0.02)

    var meanAbsLow: Double = 0
    var meanAbsHigh: Double = 0
    for t in [0, 49] {
        let x0T = GPUTensor(engine: engine, data: x0, rows: 1)
        let noiseT = GPUTensor(engine: engine, data: noise, rows: 1)
        session.begin()
        let xT = session.encodeDiffusedLatents(x0: x0T, t: t,
                                               schedule: schedule, noise: noiseT)
        _ = session.end()
        let arr = xT.xT.toArray()
        let meanAbs = Double(arr.reduce(0) { $0 + abs($1) }) / Double(arr.count)
        if t == 0 { meanAbsLow = meanAbs } else { meanAbsHigh = meanAbs }
    }
    expect(meanAbsLow > 0.9 && meanAbsLow < 1.1,
           "t=0: x_t ≈ x0 (среднее |x| ≈ 1): \(meanAbsLow)")
    expect(meanAbsHigh < meanAbsLow * 0.95,
           "t=49: шум заметно подавляет сигнал (\(meanAbsHigh) < 0.95×\(meanAbsLow))")
    expect(meanAbsHigh > 0.4,
           "t=49: среднее |x| всё ещё ненулевое: \(meanAbsHigh)")
}

func testGPURandnStats() {
    section("Фаза 3: GPU randn — статистика N(0,1)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let n = 200_000
    session.begin()
    let t = session.randn(seed: 1234, rows: 1, cols: n)
    _ = session.end()
    let arr = t.toArray()

    let mean = Double(arr.reduce(0) { $0 + Double($1) }) / Double(arr.count)
    let variance = Double(arr.reduce(0) { $0 + Double($1) * Double($1) }) / Double(arr.count) - mean * mean
    expect(abs(mean) < 0.01, "randn: mean ≈ 0 (получено \(mean))")
    expect(abs(variance - 1) < 0.05, "randn: variance ≈ 1 (получено \(variance))")
    expect(arr.allSatisfy { $0.isFinite }, "randn: конечные значения")
}
