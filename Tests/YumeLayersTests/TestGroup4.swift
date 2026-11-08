// YumeLayers — группа тестов (выделено из main.swift).
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import Metal
import AVFoundation
import UniformTypeIdentifiers
import Accelerate
import VideoToolbox
import IOKit

func testBufferLifecycle() {
    section("GPU: буферы и сессия")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // Пул не выдаёт один буфер дважды.
    do {
        let pool = BufferPool(engine)
        let initial = pool.get(16)
        pool.put(initial)
        let a = pool.get(16), b = pool.get(16)
        expect(a === initial, "пул возвращает освобождённый буфер")
        expect(!(a === b), "пул не выдаёт один буфер дважды")
    }

    // Повторные upload в одну сессию не затирают друг друга.
    do {
        let a = GPUTensor(engine: engine, rows: 1, cols: 4)
        let b = GPUTensor(engine: engine, rows: 1, cols: 4)
        for _ in 0..<3 {
            session.begin()
            session.upload(a, from: [1, 2, 3, 4])
            session.upload(b, from: [5, 6, 7, 8])
            _ = session.end()
            expectEqual(a.toArray(), [1, 2, 3, 4], "upload a не затёрт")
            expectEqual(b.toArray(), [5, 6, 7, 8], "upload b не затёрт")
        }
    }

    // copy внутри сессии: копирует данные без отдельного command buffer.
    do {
        let src = GPUTensor(engine: engine, data: [1, 2, 3, 4, 5, 6], rows: 2)
        let dst = GPUTensor(engine: engine, rows: 2, cols: 3)
        session.begin()
        session.copy(dst, from: src)
        _ = session.end()
        expectEqual(dst.toArray(), [1, 2, 3, 4, 5, 6], "copy в сессии копирует данные")
    }

    // zeroFill внутри сессии: гарантированные нули поверх мусора.
    do {
        let t = GPUTensor(engine: engine, data: [7, 7, 7, 7, 7, 7], rows: 2)
        session.begin()
        session.zeroFill(t)
        _ = session.end()
        expectEqual(t.toArray(), [0, 0, 0, 0, 0, 0], "zeroFill обнуляет тензор")
    }

    // addInPlace после zeroFill даёт ровно второе слагаемое
    // (регрессия на «сумма с неинициализированной памятью»).
    do {
        let acc = GPUTensor(engine: engine, data: [123, -456, 789, 0.5], rows: 1)
        let g = GPUTensor(engine: engine, data: [1, 2, 3, 4], rows: 1)
        session.begin()
        session.zeroFill(acc)
        session.addInPlace(acc, g)
        _ = session.end()
        expectEqual(acc.toArray(), [1, 2, 3, 4], "zeroFill + addInPlace = точная копия")
    }
}

// MARK: - 7. Свёрточный VAE: форма и детерминизм


func testConvVAEShape() {
    section("ConvVAE: форма и детерминизм")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24

    let vae = ConvVAEGPU(engine: engine)
    let img = makeSyntheticScene(w: w, h: h, rng: &systemRNG)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h, deterministic: true)
    _ = session.end()

    expectEqual(cache.z.rows, ConvVAEGPU.latentC, "латент: C каналов")
    expectEqual(cache.z.cols, (w / 8) * (h / 8), "латент: (w/8)×(h/8)")
    expectEqual(cache.recon.rows, 3, "реконструкция: 3 канала")
    expectEqual(cache.recon.cols, w * h, "реконструкция: w×h")

    let recon = cache.recon.toArray()
    expect(recon.allSatisfy { $0.isFinite }, "реконструкция без NaN")
    expect(recon.allSatisfy { $0 >= 0 && $0 <= 1 }, "реконструкция в [0,1] (после сигмоиды)")
}

// MARK: - 7b. Вариационный путь: примитивы

/// Максимальное |a-b| по двум массивам (для сверки тензоров).
func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return .infinity }
    var m = 0.0
    for i in 0..<a.count { m = max(m, abs(Double(a[i] - b[i]))) }
    return m
}


func testVAEPrimitives() {
    section("VAE: примитивы")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // exp
    do {
        let a = GPUTensor(engine: engine, data: [0, 1, -2], rows: 1)
        session.begin()
        let e = session.exp(a)
        _ = session.end()
        let v = e.toArray()
        expectClose(Double(v[0]), 1.0, 1e-6, "exp(0)=1")
        expectClose(Double(v[1]), 2.718281828, 1e-5, "exp(1)=e")
        expectClose(Double(v[2]), 0.135335283, 1e-6, "exp(-2)")
    }

    // expHalf: σ = exp(0.5·logvar), logvar зажат в [-10, 10]
    do {
        let a = GPUTensor(engine: engine, data: [0, 1.386294361, 40, -40], rows: 1)
        session.begin()
        let e = session.expHalf(a)
        _ = session.end()
        let v = e.toArray()
        expectClose(Double(v[0]), 1.0, 1e-6, "σ(logvar=0)=1")
        expectClose(Double(v[1]), 2.0, 1e-5, "σ(logvar=ln4)=2")
        expectClose(Double(v[2]), exp(5.0), 1e-3, "σ: logvar зажат сверху (10)")
        expectClose(Double(v[3]), exp(-5.0), 1e-6, "σ: logvar зажат снизу (-10)")
    }

    // addConst
    do {
        let a = GPUTensor(engine: engine, data: [1, 2, 3], rows: 1)
        session.begin()
        let c = session.addConst(a, -1.5)
        _ = session.end()
        expectEqual(c.toArray(), [-0.5, 0.5, 1.5], "addConst: A + s")
    }

    // fma: C = A + B*D  (z = mu + σ·ε)
    do {
        let a = GPUTensor(engine: engine, data: [1, 2, 3], rows: 1)
        let b = GPUTensor(engine: engine, data: [2, 2, 2], rows: 1)
        let d = GPUTensor(engine: engine, data: [0.5, 1, -1], rows: 1)
        session.begin()
        let c = session.fma(a, b, d)
        _ = session.end()
        expectEqual(c.toArray(), [2, 4, 1], "fma: A + B*D")
    }

    // split2 / merge2 — взаимно обратны
    do {
        let c = 3, plane = 5
        var raw = [Float]()
        for ch in 0..<(2 * c) { for i in 0..<plane { raw.append(Float(ch * 100 + i)) } }
        let x = GPUTensor(engine: engine, data: raw, rows: 2 * c)
        session.begin()
        let (mu, lv) = session.split2(x, c: c, plane: plane)
        let back = session.merge2(mu, lv, c: c, plane: plane)
        _ = session.end()
        expectEqual(maxAbsDiff(back.toArray(), raw), 0, "merge2∘split2 = тождество")
        expectEqual(mu.toArray().first!, 0, "split2: mu[0] = X[0]")
        expectEqual(lv.toArray().first!, 300, "split2: logvar[0] = X[C·plane]")
    }

    // randn: детерминированный, среднее≈0, σ≈1
    do {
        let n = 1 << 16
        session.begin()
        let g1 = session.randn(seed: 7, rows: 1, cols: n)
        let g2 = session.randn(seed: 7, rows: 1, cols: n)
        let g3 = session.randn(seed: 8, rows: 1, cols: n)
        _ = session.end()
        let v1 = g1.toArray(), v2 = g2.toArray(), v3 = g3.toArray()
        expectEqual(maxAbsDiff(v1, v2), 0, "randn: одинаковый seed → одинаковый шум")
        expect(maxAbsDiff(v1, v3) > 0.1, "randn: другой seed → другой шум")
        let mean = v1.reduce(0) { $0 + Double($1) } / Double(n)
        let varSum = v1.reduce(0) { $0 + pow(Double($1) - mean, 2) } / Double(n)
        expectClose(mean, 0.0, 0.02, "randn: среднее ≈ 0")
        expectClose(sqrt(varSum), 1.0, 0.02, "randn: σ ≈ 1")
        expect(v1.allSatisfy { $0.isFinite }, "randn: без NaN/Inf")
    }

    // KL: 0 при (mu=0, logvar=0); 0.5·n при (mu=1, logvar=0)
    do {
        let n = 10
        let mu0 = GPUTensor(engine: engine, data: [Float](repeating: 0, count: n), rows: 1)
        let lv0 = GPUTensor(engine: engine, data: [Float](repeating: 0, count: n), rows: 1)
        session.begin()
        let k0 = session.kl(mu: mu0, logvar: lv0)
        _ = session.end()
        expectClose(Double(k0.contents().assumingMemoryBound(to: Float.self)[0]), 0, 1e-5,
                    "KL(N(0,1)‖N(0,1)) = 0")

        let mu1 = GPUTensor(engine: engine, data: [Float](repeating: 1, count: n), rows: 1)
        session.begin()
        let k1 = session.kl(mu: mu1, logvar: lv0)
        _ = session.end()
        expectClose(Double(k1.contents().assumingMemoryBound(to: Float.self)[0]),
                    0.5 * Double(n), 1e-4, "KL: mu=1 → 0.5·n")
    }
}

// MARK: - 7c. Вариационный путь: прямой проход и сэмплирование


func testVAEVariational() {
    section("VAE: перепараметризация")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let plane = (w / 8) * (h / 8)

    let vae = ConvVAEGPU(engine: engine)
    let img = makeSyntheticScene(w: w, h: h, rng: &systemRNG)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    session.begin()
    let dc = vae.forward(session, image: x, w: w, h: h, deterministic: true)
    _ = session.end()

    expectEqual(dc.mu.rows, ConvVAEGPU.latentC, "mu: C каналов латента")
    expectEqual(dc.logvar.rows, ConvVAEGPU.latentC, "logvar: C каналов латента")
    expectEqual(dc.mu.cols, plane, "mu: (w/8)×(h/8)")
    expectEqual(dc.z.n, dc.mu.n, "z той же формы, что mu")
    expectEqual(dc.recon.rows, 3, "реконструкция: 3 канала")

    let mu = dc.mu.toArray(), lv = dc.logvar.toArray(), sig = dc.sigma.toArray()
    expect(mu.allSatisfy { $0.isFinite }, "mu без NaN")
    expect(lv.allSatisfy { $0.isFinite }, "logvar без NaN")
    // σ = exp(0.5·lv) — проверяем на всех элементах
    var sigErr = 0.0
    for i in 0..<lv.count {
        let expectS = exp(0.5 * Double(min(max(lv[i], -10), 10)))
        sigErr = max(sigErr, abs(Double(sig[i]) - expectS))
    }
    expect(sigErr < 1e-5, "σ = exp(0.5·logvar) — макс. ошибка \(sigErr)")

    // Детерминированный режим: z ровно mu (ε = 0).
    expectEqual(maxAbsDiff(dc.z.toArray(), mu), 0, "deterministic: z == mu")

    // Стохастический режим: z = mu + σ·ε, ε ≠ 0.
    session.begin()
    let sc = vae.forward(session, image: x, w: w, h: h, seed: 424242)
    _ = session.end()
    let zs = sc.z.toArray(), eps = sc.eps.toArray()
    expect(eps.contains { abs($0) > 1e-6 }, "стохастический режим: ε ≠ 0")
    var zErr = 0.0
    for i in 0..<zs.count {
        zErr = max(zErr, abs(Double(zs[i]) - (Double(sc.mu.toArray()[i]) + Double(sig[i]) * Double(eps[i]))))
    }
    expect(zErr < 1e-4, "z = mu + σ·ε — макс. ошибка \(zErr)")

    // Тот же seed → тот же z (воспроизводимость обучения).
    session.begin()
    let sc2 = vae.forward(session, image: x, w: w, h: h, seed: 424242)
    _ = session.end()
    expectEqual(maxAbsDiff(sc2.z.toArray(), zs), 0, "тот же seed → тот же z")

    // Другой seed → другой z.
    session.begin()
    let sc3 = vae.forward(session, image: x, w: w, h: h, seed: 424243)
    _ = session.end()
    expect(maxAbsDiff(sc3.z.toArray(), zs) > 1e-4, "другой seed → другой z")

    // Реконструкция в [0,1] (сигмоида) и без NaN в обоих режимах.
    let r = sc.recon.toArray()
    expect(r.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "реконструкция в [0,1] без NaN")

    // Латент должен быть УЗКИМ МЕСТОМ, а не копией кадра.
    // Сжатие = (3·hw) / (C·hw/64) = 192/C.
    let ratio = ConvVAEGPU.compressionRatio
    expect(ratio >= 8,
           String(format: "латент сжимает кадр в %.0f× (≥8× — иначе это не латент)", ratio))
    expectEqual(dc.mu.n * Int(ratio.rounded()), 3 * w * h,
                "сжатие согласовано с формой латента")
    // Проверяем на реальном кадре: 3·w·h значений RGB → mu.n значений латента.
    let actualRatio = Double(3 * w * h) / Double(dc.mu.n)
    expectClose(actualRatio, ratio, 0.001, "фактическое сжатие = 192/C")
}

// MARK: - 7d. Вариационный путь: градиенты против конечных разностей


func testVAEGradients() {
    section("VAE: градиенты vs конечные разности")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let beta: Float = 0.5

    let vae = ConvVAEGPU(engine: engine)
    let img = makeSyntheticScene(w: w, h: h, rng: &systemRNG)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    /// Полная потеря на CPU: MSE + β·KL (в точности та же формула, что в backward).
    ///
    /// `deterministic` здесь всегда false: шум ε задаётся только (seed, index),
    /// поэтому между вызовами он не меняется, а путь перепараметризации
    /// `dlv += dz·ε·0.5·σ` оказывается задействован — иначе он не проверяется.
    func loss() -> Double {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
        _ = session.end()
        let r = c.recon.toArray()
        let mu = c.mu.toArray(), lv = c.logvar.toArray()
        var mse = 0.0
        for i in 0..<r.count {
            let d = Double(r[i] - img[i])
            mse += d * d
        }
        mse /= Double(r.count)
        var kl = 0.0
        for i in 0..<mu.count {
            let l = Double(min(max(lv[i], -10), 10))
            kl += 0.5 * (Double(mu[i]) * Double(mu[i]) + exp(l) - 1 - l)
        }
        kl /= Double(mu.count)
        return mse + Double(beta) * kl
    }

    // Небольшая тренировка, чтобы веса ушли от инициализации и градиенты были значимы.
    for step in 1...20 {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
        let (_, g) = vae.backward(session, c, target: x, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: 0.002)
        _ = session.end()
    }

    // Аналитические градиенты (детерминированный режим — ε фиксирован).
    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
    let (_, grads) = vae.backward(session, cache, target: x, beta: beta)
    _ = session.end()

    // Градиенты должны быть НЕнулевыми, иначе сверка 0 == 0 проходит вхолостую.
    var maxGrad = 0.0
    var perLayer: [String] = []
    for (i, g) in grads.enumerated() {
        let a = g.toArray()
        let m = a.map { abs(Double($0)) }.max() ?? 0
        maxGrad = max(maxGrad, m)
        if i % 2 == 0 { perLayer.append(String(format: "L%d=%.2e", i / 2, m)) }
    }
    print("      |grad| по слоям: \(perLayer.joined(separator: " "))")
    expect(maxGrad > 1e-6,
           String(format: "градиенты невырождены (max |g| = %.2e)", maxGrad))

    // Активации энкодера не должны схлопнуться в ноль целиком.
    expect(cache.a1.toArray().contains { $0 > 1e-6 }, "a1 не вырождена")
    expect(cache.a2.toArray().contains { $0 > 1e-6 }, "a2 не вырождена")

    // Точки для сверки выбираем по наибольшему |градиенту|: у мёртвых ReLU-каналов
    // и аналитика, и конечная разность равны нулю, и такая проверка ничего не значит.
    let eps: Float = 1e-2
    var checked = 0
    for layerIdx in [2, 3] {   // e3 — перед латентом, d1 — сразу после
        let g = grads[layerIdx * 2].toArray()
        let top = (0..<g.count).sorted { abs(g[$0]) > abs(g[$1]) }.prefix(2)
        for weightIdx in top {
            let layer = vae.layers[layerIdx]
            let analytic = Double(g[weightIdx])
            expect(abs(analytic) > 1e-7,
                   String(format: "слой %d вес %d: градиент значим (%.2e)", layerIdx, weightIdx, analytic))

            let orig = layer.W.getValue(weightIdx)
            layer.W.setValue(weightIdx, orig + eps)
            let lp = loss()
            layer.W.setValue(weightIdx, orig - eps)
            let lm = loss()
            layer.W.setValue(weightIdx, orig)

            let numeric = (lp - lm) / (2 * Double(eps))
            let denom = max(1e-9, abs(analytic) + abs(numeric))
            let rel = abs(analytic - numeric) / denom
            expect(rel < 0.10,
                   String(format: "слой %d вес %d: аналит %.4e числ %.4e (отн %.1f%%)",
                          layerIdx, weightIdx, analytic, numeric, rel * 100))
            checked += 1
        }
    }
    expectEqual(checked, 4, "проверено 4 точки")
}

// MARK: - 7e. Обучение действительно снижает потерю


func testVAEConvergence() {
    section("VAE: обучение снижает потерю")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let beta: Float = 0.01

    let vae = ConvVAEGPU(engine: engine)
    let img = makeSyntheticScene(w: w, h: h, rng: &systemRNG)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    func loss() -> Double {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        _ = session.end()
        let r = c.recon.toArray()
        var s = 0.0
        for i in 0..<r.count {
            let d = Double(r[i] - img[i])
            s += d * d
        }
        return s / Double(r.count)
    }

    let before = loss()
    for step in 1...60 {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: UInt32(step))
        let (_, g) = vae.backward(session, c, target: x, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: 0.01)
        _ = session.end()
    }
    let after = loss()
    expect(after < before, String(format: "потеря упала: %.5f → %.5f", before, after))
    expect(after.isFinite, "потеря конечна после обучения")
}

var systemRNG = SystemRandomNumberGenerator()

// MARK: - 7f. Бюджет видеопамяти на пачку


func testVRAMBudget() {
    section("Бюджет VRAM: размер пачки")

    // Один кадр 854×480 требует ~547 МБ активаций (доминирует im2col слоя d3).
    let big = convVAEBytesPerSample(w: 856, h: 480)
    let bigMB = Double(big) / 1_048_576
    expect(bigMB > 450 && bigMB < 650,
           String(format: "854×480: %.0f МБ на кадр (ожидалось 450–650)", bigMB))

    // Для сравнения: сам кадр — всего 4.9 МБ. Старая формула видела только его.
    let frameMB = Double(856 * 480 * 3 * 4) / 1_048_576
    expect(frameMB < 6, String(format: "кадр целиком: %.1f МБ — в 100 раз меньше активаций", frameMB))

    // Квадратичный рост по числу пикселей.
    let half = convVAEBytesPerSample(w: 428, h: 240)
    let ratio = Double(big) / Double(max(half, 1))
    expect(ratio > 3.5 && ratio < 4.5,
           String(format: "память ∝ площади: 4× пикселей → %.2f× памяти", ratio))

    // 8 ГБ рабочего набора, половина под пачку → не 16 кадров, а единицы.
    let budget8GB = 8 * 1024 * 1024 * 1024 / 2
    let b = effectiveBatchSize(requested: 16, w: 856, h: 480, budgetBytes: budget8GB)
    expect(b >= 5 && b <= 9,
           "8 ГБ VRAM, 854×480: пачка \(b) (ожидалось 5–9, а не 16)")

    // Маленькое разрешение — пачка ограничена только запросом.
    let small = effectiveBatchSize(requested: 4, w: 64, h: 64,
                                   budgetBytes: 8 * 1024 * 1024 * 1024 / 2)
    expectEqual(small, 4, "64×64: пачка = запрошенная")

    // Нулевой бюджет → всё равно 1 (иначе обучение невозможно).
    expectEqual(effectiveBatchSize(requested: 8, w: 856, h: 480, budgetBytes: 0), 1,
                "нулевой бюджет → пачка 1")

    // Запрос меньше бюджета → запрос не увеличивается.
    expectEqual(effectiveBatchSize(requested: 1, w: 856, h: 480, budgetBytes: budget8GB), 1,
                "запрос 1 → пачка 1")

    // Пул кадров (60%) + активации (30%) не должны превышать рабочий набор.
    let ws: UInt64 = 8 * 1024 * 1024 * 1024
    let poolBytes = gpuFramePoolCapacity(framesTotal: 43187, frameBytes: 856 * 480 * 3 * 4,
                                         workingSet: ws) * 856 * 480 * 3 * 4
    let actBytes = activationBudgetBytes(workingSet: ws)
    expect(poolBytes + actBytes <= Int(ws),
           "пул + активации ≤ рабочий набор (\(poolBytes / 1_048_576) + \(actBytes / 1_048_576) МБ)")
    expectClose(activationBudgetFraction, 0.30, 1e-9, "доля активаций = 30%")
    // Пул ограничен ещё и сверху (preferred = 1024 кадра), поэтому доля ≤ 60%.
    let poolFraction = Double(poolBytes) / Double(ws)
    expect(poolFraction <= 0.60 + 1e-6,
           String(format: "доля пула кадров ≤ 60%% (%.1f%%)", poolFraction * 100))
    expect(poolFraction > 0.5,
           String(format: "пул загружает больше половины VRAM (%.1f%%)", poolFraction * 100))

    // Фаза 1: бюджеты под новые целевые разрешения (96/128/192px).
    for (w, h) in [(96, 56), (128, 72), (192, 112)] {
        let perSample = convVAEBytesPerSample(w: w, h: h)
        expect(perSample > 0, "\(w)×\(h): байт на сэмпл > 0")
        let batch = effectiveBatchSize(requested: 16, w: w, h: h,
                                       budgetBytes: Int(ws))
        expect(batch >= 1, "\(w)×\(h): пачка ≥ 1 (получено \(batch))")
        expect(perSample * batch <= Int(ws),
               "\(w)×\(h): активации пачки ≤ рабочий набор (\(perSample / 1_048_576) МБ×\(batch))")
    }

    // Сообщение при OOM должно называть объём и подсказывать, что делать.
    let msg = allocationFailureMessage(bytes: 512 * 1_048_576, availableBytes: 100 * 1_048_576,
                                       what: "тензор 16×410880")
    expect(msg.contains("512 МБ"), "OOM-сообщение: нужный объём — \(msg)")
    expect(msg.contains("100 МБ"), "OOM-сообщение: свободный объём")
    expect(msg.contains("тензор 16×410880"), "OOM-сообщение: что именно не влезло")
    expect(msg.contains("пачки"), "OOM-сообщение: подсказка про размер пачки")
}

// MARK: - 7g. Отметки «кадр обработан»


