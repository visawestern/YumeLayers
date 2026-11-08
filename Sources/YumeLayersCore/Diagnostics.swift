// YumeLayers — диагностические прогоны (выделено из ConvTrainer.swift).
//
// Отдельный модуль для соблюдения лимита 600 строк/файл.
import Foundation
import Metal

public func runGpuHeat(seconds: Double = 60, duty: Double = 0.85,
                       w: Int = 856, h: Int = 480) {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let vae = ConvVAEGPU(engine: engine)
    let sensor = GpuSensor()
    let img = makeNormalSynthetic(w: w, h: h)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    // duty <= 0 — автоматический режим: нагрузку задаёт контур по температуре,
    // ровно как в обучении. Это и есть проверка «работает на грани».
    // В ОБОИХ режимах потолок — физический предел 98%: диагностика не имеет
    // права быть опаснее обучения. Раньше здесь стояло 98% в авто и 100% в ручном.
    let auto = duty <= 0
    let requestedDuty = clampedGpuDuty(duty)
    var power = PowerController(ceiling: safeDutyCap,
                                startDuty: auto ? 0.5 : requestedDuty)
    var currentDuty = auto ? power.duty : requestedDuty

    yumeLog(String(format: "Прогрев: %@, %d×%d, %@, %.0f с",
                   engine.deviceName, w, h,
                   auto ? String(format: "авто (цель %.0f °C)", power.targetTempC)
                        : String(format: "доля работы %.0f%%", requestedDuty * 100),
                   seconds))
    yumeLog(String(format: "Потолок доли работы %.0f%% · запрос %.0f%%",
                   safeDutyCap * 100, duty * 100))
    let start = Date()
    var t = 0
    var lastLog = Date.distantPast
    var peakTemp = 0.0
    var stepMsAll: [Double] = []

    while Date().timeIntervalSince(start) < seconds {
        if TrainingControl.shared.isCancelled { break }
        session.begin()
        let cache = vae.forward(session, image: x, w: w, h: h)
        let (_, g) = vae.backward(session, cache, target: x)
        t += 1
        vae.applyAdam(session, grads: g, t: t, lr: 0.0001)
        let gpuMs = session.end()
        stepMsAll.append(gpuMs)

        // Датчик читаем один раз за шаг и В ОБОИХ режимах: аварийная
        // температура обязана останавливать и ручной прогрев тоже.
        let stats = sensor.read()
        if auto {
            currentDuty = clampedGpuDuty(power.update(stats: stats))
        }
        if let s = stats, s.temperatureC >= emergencyTempC {
            yumeLog(String(format: "🔥 %.0f °C — остывание 20 с", s.temperatureC))
            Thread.sleep(forTimeInterval: 20)
            currentDuty = clampedGpuDuty(max(power.duty * 0.5, 0.05))
        }

        // Финальный клэмп на месте применения: даже если контур или ручной
        // запрос когда-нибудь вернут больше, до GPU дойдёт не выше 98%.
        currentDuty = clampedGpuDuty(currentDuty)
        let sleepMs = dutySleepMs(gpuMs: gpuMs, duty: currentDuty, maxPauseMs: 5000)
        if sleepMs > 0 { Thread.sleep(forTimeInterval: sleepMs / 1000) }

        if Date().timeIntervalSince(lastLog) > 3 {
            lastLog = Date()
            guard let s = stats else { continue }
            peakTemp = max(peakTemp, s.temperatureC)
            yumeLog(String(format: "  %.0f °C · %.0f Вт · нагрузка %.0f%% · %.0f мс/шаг",
                           s.temperatureC, s.powerW, currentDuty * 100, gpuMs))
        }
    }

    let third = max(1, stepMsAll.count / 3)
    let head = stepMsAll.prefix(third).reduce(0, +) / Double(third)
    let tail = stepMsAll.suffix(third).reduce(0, +) / Double(third)
    yumeLog(String(format: "Итог: пик %.0f °C, нагрузка %.0f%%, шаг %.0f → %.0f мс, шагов %d",
                   peakTemp, currentDuty * 100, head, tail, t))
}

public func runGpuBench() {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let dev = engine.device
    yumeLog("⚠️ Диагностика — не запускайте при активном обучении (нагрузка на GPU).")
    yumeLog("Устройство: \(dev.name)")
    yumeLog("Унифицированная память: \(dev.hasUnifiedMemory)")
    yumeLog("Рекомендуемый рабочий набор: \(dev.recommendedMaxWorkingSetSize / 1024 / 1024) МБ")
    yumeLog("Макс. потоков в группе: \(dev.maxThreadsPerThreadgroup.width)")

    // Пропускная способность: relu над 8.4 млн float (32 МиБ буфера,
    // 64 МиБ трафика читаем+пишем), 3 повтора
    let n = 32 * 1024 * 1024 / 4
    let a = GPUTensor(engine: engine, rows: 1, cols: n)
    session.begin()
    _ = session.relu(a)
    session.end()
    var t0 = Date()
    for _ in 0..<3 {
        session.begin()
        _ = session.relu(a)
        session.end()
    }
    var dt = Date().timeIntervalSince(t0) / 3
    let gb = Double(n * 4 * 2) / 1e9
    yumeLog(String(format: "Память (relu, буфер 32 МиБ, трафик %.0f МиБ): %.2f ГБ/с, %.2f мс",
                   gb * 1024, gb / dt, dt * 1000))

    // Вычисления: тайловый matmul 512³
    let m = 512, k = 512, nn = 512
    let A = GPUTensor(engine: engine, rows: m, cols: k)
    let B = GPUTensor(engine: engine, rows: k, cols: nn)
    t0 = Date()
    session.begin()
    _ = session.matmul(A, B)
    session.end()
    dt = Date().timeIntervalSince(t0)
    let flops = 2.0 * Double(m) * Double(k) * Double(nn)
    yumeLog(String(format: "Matmul 512³: %.1f мс, %.1f ГФЛОП/с", dt * 1000, flops / dt / 1e9))
}

public func profileConv(w: Int, h: Int) {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let vae = ConvVAEGPU(engine: engine)
    let img = makeNormalSynthetic(w: w, h: h)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    func time(_ name: String, _ reps: Int, _ body: () -> Void) {
        body()  // прогрев
        let t0 = Date()
        for _ in 0..<reps { body() }
        let dt = Date().timeIntervalSince(t0) / Double(reps)
        yumeLog(String(format: "%@: %.1f мс", name, dt * 1000))
    }

    time("forward", 3) {
        session.begin()
        _ = vae.forward(session, image: x, w: w, h: h)
        let gpu = session.end()
        yumeLog("  (GPU: \(String(format: "%.1f", gpu)) мс)")
    }

    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h)
    session.end()

    time("backward (полный)", 3) {
        session.begin()
        _ = vae.backward(session, cache, target: x)
        session.end()
    }

    session.begin()
    let dP6 = session.sigmoidGrad(cache.p6, session.smul(2.0 / Float(3 * w * h), session.sub(cache.recon, x)))
    session.end()
    time("gW всех слоёв", 3) {
        session.begin()
        _ = vae.layers[0].backward(session, cache.c1, dOut: cache.p1)
        _ = vae.layers[1].backward(session, cache.c2, dOut: cache.p2)
        _ = vae.layers[2].backward(session, cache.c3, dOut: cache.p3)
        _ = vae.layers[3].backward(session, cache.c4, dOut: cache.p4)
        _ = vae.layers[4].backward(session, cache.c5, dOut: cache.p5)
        _ = vae.layers[5].backward(session, cache.c6, dOut: dP6)
        session.end()
    }

    time("активации backward (relu/tanh/сигмоид)", 3) {
        session.begin()
        _ = session.reluGrad(cache.a5, cache.p5)
        _ = session.reluGrad(cache.a4, cache.p4)
        _ = session.reluGrad(cache.a2, cache.p2)
        _ = session.reluGrad(cache.a1, cache.p1)
        _ = session.sigmoidGrad(cache.p6, cache.p6)
        _ = session.tanhGrad(cache.p3)
        session.end()
    }

    time("downsample2x ×3", 3) {
        session.begin()
        _ = session.downsample2x(cache.p6, h: h / 2, w: w / 2)
        _ = session.downsample2x(cache.p5, h: h / 4, w: w / 4)
        _ = session.downsample2x(cache.p4, h: h / 8, w: w / 8)
        session.end()
    }

    profileConvKernels(engine, session, vae, cache, w: w, h: h)
}

/// Изоляция тяжёлых kernels и служебных операций (хвост профилирования).
private func profileConvKernels(_ engine: MetalEngine, _ session: GpuSession,
                                _ vae: ConvVAEGPU, _ cache: VAECache,
                                w: Int, h: Int) {
    func time(_ name: String, _ reps: Int, _ body: () -> Void) {
        body()
        let t0 = Date()
        for _ in 0..<reps { body() }
        let dt = Date().timeIntervalSince(t0) / Double(reps)
        yumeLog(String(format: "%@: %.1f мс", name, dt * 1000))
    }

    let cols6 = cache.c6.cols
    let dOut6 = cache.p6
    time("gradW d3 (3×144, K=410880)", 3) {
        session.begin()
        _ = session.gradW(dOut6, cols6, oc: 3, kk: 144)
        session.end()
    }
    time("biasRowGrad d3", 3) {
        session.begin()
        _ = session.biasRowGrad(dOut6)
        session.end()
    }
    time("convBackInput d3", 3) {
        session.begin()
        _ = session.convBackInputS1(dOut6, vae.layers[5].W, ic: 16, oc: 3,
                                    ih: h, iw: w, k: 3, pad: 1)
        session.end()
    }
    let cols1 = cache.c1.cols
    let dOut1 = cache.p1
    time("gradW e1 (16×27, K=102720)", 3) {
        session.begin()
        _ = session.gradW(dOut1, cols1, oc: 16, kk: 27)
        session.end()
    }

    let recon = cache.recon
    time("readback 1 кадр", 5) {
        _ = recon.toArray()
    }

    time("аллокация \(16 * w * h * 4 / 1_048_576) МБ (16×\(w)×\(h))", 5) {
        _ = GPUTensor(engine: engine, rows: 16, cols: w * h)
    }
    time("заполнение \(16 * w * h * 4 / 1_048_576) МБ (relu)", 5) {
        let a = GPUTensor(engine: engine, rows: 16, cols: w * h)
        session.begin()
        _ = session.relu(a)
        session.end()
    }

    let bigIn = GPUTensor(engine: engine, rows: 16, cols: w * h)
    time("im2col d3 (16×480×856)", 3) {
        session.begin()
        _ = session.im2col(bigIn, ih: h, iw: w, ic: 16, kh: 3, kw: 3, stride: 1, pad: 1)
        session.end()
    }
    time("col2im d3 (16×480×856)", 3) {
        session.begin()
        let cols = session.im2col(bigIn, ih: h, iw: w, ic: 16, kh: 3, kw: 3, stride: 1, pad: 1)
        _ = session.col2im(cols, ic: 16, ih: h, iw: w, kh: 3, kw: 3, stride: 1, pad: 1,
                           oh: h, ow: w)
        session.end()
    }
}

public func checkConvGradients() {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let vae = ConvVAEGPU(engine: engine)
    let w = 32, h = 24   // 32/8=4, 24/8=3 — минимальный рабочий размер
    let beta: Float = 1.0
    let img = makeNormalSynthetic(w: w, h: h)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    func lossValue() -> Double {
        session.begin()
        // deterministic: ε = 0, чтобы шум не менялся между вызовами при
        // численном дифференцировании.
        let cache = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        session.end()
        // Та же потеря, что в backward: MSE + β·KL. Без KL-слагаемого
        // аналитические градиенты заведомо не сойдутся с конечными разностями.
        return vaeLoss(cache: cache, target: img, beta: beta)
    }

    yumeLog(String(format: "loss до тренировки: %.8f", lossValue()))
    // короткая тренировка, чтобы веса ушли от нулевого инициализации и градиенты были значимыми
    for step in 1...30 {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        let (_, g) = vae.backward(session, c, target: x, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: 0.001)
        session.end()
    }
    yumeLog(String(format: "loss после тренировки: %.8f", lossValue()))

    // аналитические градиенты
    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h, deterministic: true)
    let (_, grads) = vae.backward(session, cache, target: x, beta: beta)
    session.end()

    for (li, _) in vae.layers.enumerated() {
        let a = grads[li * 2].toArray()
        let mx = a.map { abs($0) }.max() ?? 0
        let nan = a.contains { $0.isNaN }
        yumeLog("слой \(li): max|grad| = \(mx)\(nan ? "  ЕСТЬ NaN" : "")")
    }

    // 1e-2, а не 1e-3: градиенты вариационного узла ~1e-5, при шаге 1e-3
    // изменение потери тонет в шуме float32.
    let eps: Float = 1e-2
    var worst: Double = 0
    var checks = 0

    for (li, layer) in vae.layers.enumerated() {
        let analytic = grads[li * 2].toArray()
        let n = layer.W.n
        // берём 3 крупнейших по модулю градиента — там сигнал заведомо выше шума
        let idxs = (0..<n).sorted { abs(analytic[$0]) > abs(analytic[$1]) }.prefix(3)

        for idx in idxs {
            let orig = layer.W.getValue(idx)
            layer.W.setValue(idx, orig + eps)
            let lp = lossValue()
            layer.W.setValue(idx, orig - eps)
            let lm = lossValue()
            layer.W.setValue(idx, orig)
            let numeric = (lp - lm) / Double(2 * eps)
            let a = Double(analytic[idx])
            let denom = max(abs(numeric), abs(a), 1e-6)
            let rel = abs(numeric - a) / denom
            worst = max(worst, rel)
            checks += 1
            yumeLog(String(format: "слой %d  idx %6d  numeric %+.8f  analytic %+.8f  rel %.3f",
                         li, idx, numeric, a, rel))
        }
    }

    yumeLog(String(format: "Проверок: %d, худшая относительная ошибка: %.4f", checks, worst))
    let score = worst < 0.05 ? 100.0 : (worst < 0.15 ? 85.0 : 50.0)
    yumeLog(String(format: "Оценка корректности backward: %.0f/100", score))
}

