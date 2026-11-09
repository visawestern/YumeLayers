// YumeLayers — тесты Фазы 7: приближённый FID и статистика фич.
import Foundation

func testFeatureStats() {
    section("Фаза 7: статистика фич (mean, диагональная дисперсия)")
    let samples = [
        [Float](repeating: 1.0, count: 3),
        [Float](repeating: 1.0, count: 3),
        [Float](repeating: 1.0, count: 3),
    ]
    guard let s = featureStats(samples: samples) else {
        expect(false, "статистика посчитана"); return
    }
    expectEqual(s.mean.count, 3, "mean: 3 компоненты")
    expectClose(s.mean[0], 1.0, 1e-9, "mean[0] = 1")
    expectClose(s.diagVar[0], 0.0, 1e-9, "var[0] = 0 (константа)")

    let hetero = [
        [Float](repeating: 0.0, count: 2),
        [Float](repeating: 1.0, count: 2),
        [Float](repeating: 2.0, count: 2),
    ]
    guard let h = featureStats(samples: hetero) else {
        expect(false, "статистика hetero"); return
    }
    expectClose(h.mean[0], 1.0, 1e-9, "mean = 1")
    expectClose(h.diagVar[0], 2.0 / 3.0, 1e-9, "дисперсия = 2/3")

    expect(featureStats(samples: []) == nil, "пусто → nil")
    expect(featureStats(samples: [[1.0], [2.0, 3.0]]) == nil,
           "разные длины → nil")
}

func testFIDOnVAELatents() {
    section("Фаза 7: FID на латентах VAE (same=0, different>0)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let vae = ConvVAEGPU(engine: engine, seed: 0x1234_0000_0000_0001)
    func latentsOf(_ baseSeed: UInt64) -> [[Float]] {
        let frames = (0..<4).map { f -> [Float] in
            var rng = SplitMix64(seed: baseSeed + UInt64(f))
            return makeSyntheticScene(w: w, h: h, rng: &rng)
        }
        let tensors = frames.map { GPUTensor(engine: engine, data: hwcToChw($0), rows: 3) }
        return encodeWindowLatents(vae: vae, session: session,
                                   frames: tensors, w: w, h: h)
    }
    let a = latentsOf(8000)
    let b = latentsOf(9000)
    let statsA = featureStats(samples: a)!
    let statsB = featureStats(samples: b)!
    let fidAA = frechetDistanceDiagonal(mean1: statsA.mean, var1: statsA.diagVar,
                                        mean2: statsA.mean, var2: statsA.diagVar)
    expectClose(fidAA, 0.0, 1e-9, "FID(A,A)=0")
    let fidAB = frechetDistanceDiagonal(mean1: statsA.mean, var1: statsA.diagVar,
                                        mean2: statsB.mean, var2: statsB.diagVar)
    expect(fidAB > 0.05, "FID(A,B)>0.05 (разные сцены): \(fidAB)")
    expect(fidAB.isFinite, "FID конечен")
}

func testFrechetDistanceDiagonal() {
    section("Фаза 7: Fréchet-расстояние (диагональная ковариация)")
    // Идентичные распределения → 0.
    let mean = [0.0, 0.0]
    let var_ = [1.0, 1.0]
    expectClose(frechetDistanceDiagonal(mean1: mean, var1: var_,
                                       mean2: mean, var2: var_),
                0.0, 1e-9, "одинаковые распределения → 0")

    // Сдвиг среднего: d² = Σ (Δm²) + Σ (√v1 − √v2)².
    let mean2 = [3.0, -1.0]
    let expected = (3.0 * 3.0 + (-1.0) * (-1.0)) + 0.0
    expectClose(frechetDistanceDiagonal(mean1: mean, var1: var_,
                                       mean2: mean2, var2: var_),
                expected, 1e-9, "сдвиг среднего")

    // Разница дисперсий.
    let varSmall = [0.25, 0.25]
    let expectedV = (1.0 - 0.5) * (1.0 - 0.5) * 2.0
    expectClose(frechetDistanceDiagonal(mean1: mean, var1: var_,
                                       mean2: mean, var2: varSmall),
                expectedV, 1e-9, "разница дисперсий")

    // Симметричность.
    let d12 = frechetDistanceDiagonal(mean1: mean, var1: var_,
                                      mean2: mean2, var2: varSmall)
    let d21 = frechetDistanceDiagonal(mean1: mean2, var1: varSmall,
                                      mean2: mean, var2: var_)
    expectClose(d12, d21, 1e-9, "симметричность")
}
