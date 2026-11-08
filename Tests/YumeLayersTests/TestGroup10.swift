// YumeLayers — тесты движения и «законченности» сцены (Фаза 3).

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
