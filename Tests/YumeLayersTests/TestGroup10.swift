// YumeLayers — тесты движения и «законченности» сцены (Фаза 3).
import Foundation

func testSceneMotion() {
    section("Движение по последовательным кадрам: фон/объекты/законченность")

    // 1) Статичная сцена: все кадры одинаковые → фон стабилен, движения нет.
    let w = 32, h = 32
    let px = w * h
    let staticFrame = [Float](repeating: 0.5, count: px * 3)
    let staticSeq = (0..<5).map { _ in staticFrame }
    let s = motionSummary(frames: staticSeq, w: w, h: h)
    expectClose(s.staticFraction, 1.0, 1e-6, "статик: фон стабилен (staticFraction≈1)")
    expect(s.peakMotion == 0, "статик: пиковое движение = 0")
    expect(s.traveledCoverage == 0, "статик: покрытие движения = 0")
    expect(!s.sceneComplete, "статик: нет движения → сцена не «закончена»")

    // 2) Движущийся объект: крупный квадрат идёт через весь кадр.
    var moving: [[Float]] = []
    let bw = w / 4          // ширина объекта
    let steps = 16
    for k in 0..<steps {
        var img = [Float](repeating: 0.2, count: px * 3)   // тёмный фон
        let x0 = Int(Double(k) * Double(w - bw) / Double(steps - 1))
        for y in 0..<h {
            for x in x0..<(x0 + bw) {
                let p = y * w + x
                img[p * 3] = 0.9; img[p * 3 + 1] = 0.9; img[p * 3 + 2] = 0.9
            }
        }
        moving.append(img)
    }
    let m = motionSummary(frames: moving, w: w, h: h)
    expect(m.peakMotion > 0.1, "объект: есть движение (peakMotion>0.1)")
    expect(m.traveledCoverage > 0.9, "объект: покрыл почти весь кадр (traveled>0.9)")
    expect(m.sceneComplete, "объект: фон стабилен и движение покрыло кадр → сцена закончена")

    // 3) Шумовая/нестабильная сцена: каждый кадр разный → фона нет.
    var noiseSeq: [[Float]] = []
    for k in 0..<12 {
        var img = [Float](repeating: 0, count: px * 3)
        for i in 0..<img.count {
            let v = (UInt64(i) &* 2654435761 &+ UInt64(k) &* 97) % 1000
            img[i] = Float(v) / 1000
        }
        noiseSeq.append(img)
    }
    let c = motionSummary(frames: noiseSeq, w: w, h: h)
    expect(c.staticFraction < 0.6, "шум: фон нестабилен (staticFraction<0.6)")
    expect(!c.sceneComplete, "шум: нет стабильного фона → сцена не закончена")
}

func testConvVAEWidthScales() {
    section("Архитектура VAE: рычаг ширины каналов (width) — Фаза 1–2")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let base = ConvVAEGPU(engine: engine)
    let wide = ConvVAEGPU(engine: engine, width: 2)
    expectEqual(base.e1.oc, 16, "width=1 (дефолт): e1.oc = 16")
    expectEqual(wide.e1.oc, 32, "width=2: e1.oc = 32 (16×2)")
    expectEqual(wide.e2.oc, 64, "width=2: e2.oc = 64 (32×2)")
    expectEqual(wide.d3.ic, 32, "width=2: d3.ic = f1 = 32")
    let w = 128, h = 72
    let img = makeNormalSynthetic(w: w, h: h)
    let x = GPUTensor(engine: engine, data: img, rows: 3)
    let s = GpuSession(engine)
    s.begin()
    let cache = wide.forward(s, image: x, w: w, h: h)
    _ = s.end()
    expectEqual(cache.recon.toArray().count, w * h * 3,
                "width=2: recon имеет размер w×h×3")
}

func testCheckpointModelWidth() {
    section("Чекпоинт хранит ширину модели (modelWidth) для реконструкций")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let tmp = NSTemporaryDirectory() + "yume-cp-width-\(UUID().uuidString).safetensors"
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    let wide = ConvVAEGPU(engine: engine, latentChannels: 32, width: 2)
    try? saveState(wide.matrices, to: tmp, epoch: 1, step: 5, totalSteps: 10,
                   optimizer: CheckpointOptimizerState(step: 5, moments: wide.optimizerMoments()),
                   modelWidth: 2, latentChannels: 32)
    guard let st = try? loadStateWithMeta(path: tmp) else {
        expect(false, "чекпоинт с шириной прочитан"); return
    }
    expectEqual(st.modelWidth, 2, "modelWidth=2 сохранён и прочитан")
    expectEqual(st.latentChannels, 32, "latentChannels=32 сохранён и прочитан")
    let stMats = st.mats
    expectEqual(stMats.count, wide.layers.count * 2, "число матриц соответствует ширине модели")

    // Старый формат (без метаданных modelWidth) → по умолчанию 1.
    let base = ConvVAEGPU(engine: engine)
    try? saveState(base.matrices, to: tmp, modelWidth: nil)
    guard let st2 = try? loadStateWithMeta(path: tmp) else {
        expect(false, "чекпоинт без ширины прочитан"); return
    }
    expectEqual(st2.modelWidth, 1, "без метаданных modelWidth = 1")
}

func testCheckpointArchVersion() {
    section("Чекпоинт хранит версию архитектуры (archVersion)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let tmp = NSTemporaryDirectory() + "yume-cp-av-\(UUID().uuidString).safetensors"
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    let vae = ConvVAEGPU(engine: engine)
    try? saveState(vae.matrices, to: tmp, epoch: 1, step: 1, totalSteps: 2,
                   optimizer: CheckpointOptimizerState(step: 1, moments: vae.optimizerMoments()),
                   modelWidth: 1, latentChannels: 16, archVersion: 2)
    guard let st = try? loadStateWithMeta(path: tmp) else {
        expect(false, "чекпоинт прочитан"); return
    }
    expectEqual(st.archVersion, 2, "archVersion=2 сохранён и прочитан")

    // Старый формат (без archVersion) → по умолчанию 1.
    let base = ConvVAEGPU(engine: engine)
    try? saveState(base.matrices, to: tmp, archVersion: nil)
    guard let st2 = try? loadStateWithMeta(path: tmp) else {
        expect(false, "чекпоинт без версии прочитан"); return
    }
    expectEqual(st2.archVersion, 1, "без метаданных archVersion = 1")
}

// MARK: - Этап 0. Диагностика, ворота G1

/// G1: overfit на 16 кадрах → цель PSNR ≥35 дБ.
///
/// Если сеть не может запомнить 16 кадров одной сцены, это баг в обучении/
/// лоссе, а не «не хватает ёмкости». Тест идёт по продуктовой трассе:
/// forward → vaeLoss/backward → applyAdam, детерминированный z.
func testOverfitGate_16Frames() {
    section("G1: overfit на 16 кадрах — цель PSNR ≥35 дБ (Этап 0)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 128, h = 72
    let nFrames = 16
    var frames = [[Float]]()
    for f in 0..<nFrames {
        var rng = SplitMix64(seed: UInt64(0xABCD_F000 + f))
        frames.append(makeSyntheticScene(w: w, h: h, rng: &rng))
    }

    // Финальная диагностика этапа 0: подтверждённый конфиг (β=0.01, lr=0.004,
    // до 9500 шагов; ранний выход при достижении ≥35 дБ).
    let beta: Float = 0.01
    let lr: Float = 0.004
    let totalSteps = 9500

    let vae = ConvVAEGPU(engine: engine)
    // Модель ожидает CHW (как в продуктовом gatherBatch: hwcToChw).
    func avgPSNR() -> Float {
        var sum: Float = 0
        for img in frames {
            let chw = hwcToChw(img)
            let x = GPUTensor(engine: engine, data: chw, rows: 3)
            session.begin()
            let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
            _ = session.end()
            let reconHWC = chwToHwc(c.recon.toArray())
            sum += psnr(reference: img, candidate: reconHWC)
        }
        return sum / Float(nFrames)
    }
    var reachedAt = -1
    for step in 1...totalSteps {
        let img = frames[(step - 1) % nFrames]
        let x = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        let (_, g) = vae.backward(session, c, target: x, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: lr)
        _ = session.end()
        if step % 500 == 0, avgPSNR() >= 35 {
            reachedAt = step
            break
        }
    }

    let p = avgPSNR()
    let atNote = reachedAt > 0 ? "35 дБ достигнут на шаге \(reachedAt)" : "не достигнут за \(totalSteps)"
    print("      G1: PSNR = " + String(format: "%.2f", p)
          + " дБ (β \(beta), lr \(lr); " + atNote + ")")
    expect(p >= 35, String(format: "G1: overfit 16 кадров ≥35 дБ (получено %.2f)", p))
}

// MARK: - Этап 0.4. Hold-out по сценам (запоминание vs генерализация)

/// Другая «сцена» (горизонтальные полосы, без круга/света) для hold-out.
func makeStripedScene(w: Int, h: Int) -> [Float] {
    var img = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            let i = (y * w + x) * 3
            if (y / 12) % 2 == 0 {
                img[i] = 0.2; img[i + 1] = 0.4; img[i + 2] = 0.7
            } else {
                img[i] = 0.8; img[i + 1] = 0.6; img[i + 2] = 0.2
            }
        }
    }
    return img
}

/// Обучаемся на сцене A (16 кадров), затем меряем PSNR на A (train) и на B
/// (hold-out). Если модель генерализует — разрыв мал; если запоминает —
/// train заметно выше hold-out. Диагностика этапа 0.4.
func testHoldoutByScene() {
    section("Этап 0.4: hold-out по сценам (запоминание vs генерализация)")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 128, h = 72
    let nFrames = 16

    var sceneA = [[Float]]()   // сцена A: градиент + круг + свет
    for f in 0..<nFrames {
        var rng = SplitMix64(seed: UInt64(0x1111_0000 + f))
        sceneA.append(makeSyntheticScene(w: w, h: h, rng: &rng))
    }
    let sceneB = makeStripedScene(w: w, h: h)   // сцена B: полосы (девіm)

    let beta: Float = 0.01
    let lr: Float = 0.004
    let totalSteps = 6000

    let vae = ConvVAEGPU(engine: engine)
    func psnrOn(_ frames: [[Float]]) -> Float {
        var sum: Float = 0
        for img in frames {
            let x = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
            session.begin()
            let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
            _ = session.end()
            sum += psnr(reference: img, candidate: chwToHwc(c.recon.toArray()))
        }
        return sum / Float(frames.count)
    }
    for step in 1...totalSteps {
        let img = sceneA[(step - 1) % nFrames]
        let x = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        let (_, g) = vae.backward(session, c, target: x, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: lr)
        _ = session.end()
    }
    let trainPSNR = psnrOn(sceneA)
    let holdoutPSNR = psnrOn([sceneB])
    print("      hold-out: train PSNR = " + String(format: "%.2f", trainPSNR)
          + " дБ, hold-out PSNR = " + String(format: "%.2f", holdoutPSNR) + " дБ")
    expect(trainPSNR - holdoutPSNR >= 3.0,
           String(format: "запоминание: train заметно выше hold-out (разрыв ≥3 дБ; получено %.1f)",
                  trainPSNR - holdoutPSNR))
}
