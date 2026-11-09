// YumeLayers — тесты Фазы 3: диффузия в латентном пространстве (чистая математика).
import Foundation

func testDiffusionSchedule() {
    section("Фаза 3: линейное расписание DDPM (betas/alphas/alpha_bar)")
    let s = DiffusionSchedule(timesteps: 1000, betaStart: 1e-4, betaEnd: 0.02)
    expectEqual(s.betas.count, 1000, "1000 значений betas")
    expectEqual(s.alphas.count, 1000, "1000 значений alphas")
    expectEqual(s.alphaBars.count, 1000, "1000 значений alpha_bar")

    let b0 = s.betas[0]
    expect(b0 > 0 && b0 < 0.05, "beta_0 ∈ (0, 0.05): \(b0)")
    expect(s.betas.allSatisfy { $0 > 0 && $0 < 0.05 },
           "все betas ∈ (0, 0.05)")

    // alpha_bar[0] ≈ 1, монотонно убывает к ~0.
    expectClose(s.alphaBars[0], 1.0, 1e-3, "alpha_bar_0 ≈ 1")
    expect(s.alphaBars[999] < 0.05, "alpha_bar_999 < 0.05 (сильный шум)")
    var mono = true
    for i in 1..<1000 where s.alphaBars[i] > s.alphaBars[i - 1] { mono = false }
    expect(mono, "alpha_bar монотонно убывает")

    // Маленькое число шагов — для тестов и экспресс-обучения.
    let small = DiffusionSchedule(timesteps: 5, betaStart: 0.02, betaEnd: 0.2)
    expectEqual(small.alphaBars.count, 5, "5 шагов")
    expect(small.alphaBars[4] > 0.3 && small.alphaBars[4] < 0.8,
           "alpha_bar_4 в разумных пределах (0.3..0.8): \(small.alphaBars[4])")
    expect(small.alphaBars[0] > small.alphaBars[1] && small.alphaBars[1] > small.alphaBars[2],
           "по порядку убывания")
}

func testForwardDiffuse() {
    section("Фаза 3: прямое зашумление x0 → x_t")
    let schedule = DiffusionSchedule(timesteps: 100, betaStart: 1e-4, betaEnd: 0.02)
    let x0: [Float] = [0.5, -0.25, 0.0, 1.0, -1.0, 0.125]
    var rng = SplitMix64(seed: 7)
    let noise = (0..<x0.count).map { _ in Float.random(in: -1...1, using: &rng) }

    // t = 0: alpha_bar ≈ 1 → x_0 ≈ x0 (примесь шума ≤ √(1−0.9999)·|ε| ≈ 0.01).
    let t0 = forwardDiffuse(x0: x0, t: 0, schedule: schedule, noise: noise)
    for i in 0..<x0.count {
        expectClose(Double(t0.xT[i]), Double(x0[i]), 0.02, "t=0: x_t ≈ x0")
    }
    expectClose(t0.sqrtAlphaBar, 1.0, 1e-3, "t=0: sqrt(alpha_bar)=1")

    // t = 99: много шума, но значения конечны и ограничены.
    let t99 = forwardDiffuse(x0: x0, t: 99, schedule: schedule, noise: noise)
    expect(t99.xT.allSatisfy { $0.isFinite }, "t=99: конечные значения")
    expect(t99.xT.allSatisfy { abs($0) < 4 }, "t=99: значения ограничены")

    // Восстановление шума из x_t при известных x0/noise (формула DDPM).
    let t = 40
    let xT = forwardDiffuse(x0: x0, t: t, schedule: schedule, noise: noise)
    for i in 0..<x0.count {
        let expected = Float(xT.sqrtAlphaBar) * x0[i] + Float(xT.sqrtOneMinus) * noise[i]
        expectClose(Double(xT.xT[i]), Double(expected), 1e-4, "t=\(t): x_t = √ᾱ·x0 + √(1−ᾱ)·ε")
    }
    // eps вернулся как поданный шум.
    for i in 0..<noise.count {
        expectClose(Double(xT.eps[i]), Double(noise[i]), 1e-6, "eps = поданный шум")
    }

    // Детерминизм при одном и том же шуме.
    let again = forwardDiffuse(x0: x0, t: t, schedule: schedule, noise: noise)
    expectEqual(again.xT, xT.xT, "x_t детерминирован при фиксированном шуме")
}

func testLatentWindowDataset() {
    section("Фаза 3: датасет окон → батчи ссылок на кадры")
    // 3 клипа: a (8 кадров), b (6), c (4).
    let clips = [
        MotionClip(source: "a.mp4", startIndex: 0, frameCount: 8),
        MotionClip(source: "b.mp4", startIndex: 0, frameCount: 6),
        MotionClip(source: "c.mp4", startIndex: 0, frameCount: 4),
    ]

    // Батчи по 5 ссылок: окна разбиваются по порядку, хвост меньше.
    let batches = latentBatchRefs(windows: clips, framesPerBatch: 5)
    expectEqual(batches.count, 4, "8+6+4=18 кадров → 4 батча (5/5/5/3)")
    expectEqual(batches[0].count, 5, "батч 0: 5")
    expectEqual(batches[0][0], ExtractedFrameRef(source: "a.mp4", index: 0), "порядок сохранён: a0")
    expectEqual(batches[3].count, 3, "батч 3 (хвост): 3")
    expectEqual(batches[3][2], ExtractedFrameRef(source: "c.mp4", index: 3), "хвост: c3")

    // Пустой вход → пусто.
    expectEqual(latentBatchRefs(windows: [], framesPerBatch: 4).count, 0, "пусто → пусто")

    // framesPerBatch <= 0 → дефолт 1.
    let capped = latentBatchRefs(windows: clips, framesPerBatch: 0)
    expectEqual(capped.count, 18, "framesPerBatch<=0 → по 1 кадру")

    // Детерминированная выборка датасета.
    var rng = SplitMix64(seed: 99)
    let sampled = subsampleWindows(clips, maxWindows: 2, rng: &rng)
    expectEqual(sampled.count, 2, "выборка: ровно 2 окна")
    expect(sampled.allSatisfy { clips.contains($0) }, "выборка — подмножество исходных клипов")

    // maxWindows >= count → всё.
    var rng2 = SplitMix64(seed: 1)
    let all = subsampleWindows(clips, maxWindows: 10, rng: &rng2)
    expectEqual(all, clips, "maxWindows >= count → без изменений")
}
