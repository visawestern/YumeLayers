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

func testLayerRenderer() {
    section("Рендер слоёв: процедурные сцены с разметкой")
    testLayerRendererDeterminism()
    let scene = makeLayeredScene(w: 128, h: 96, seed: 2026)
    let n = scene.w * scene.h
    testLayerRanges(scene, n)
    testLayerComposition(scene, n)
    testLightGradient(scene)
    testLayerChwLayout(scene, n)
    testLayer480pSmoke()
    testLayerPreviewSave(scene)
}

func testLayerRendererDeterminism() {
    // Детерминизм: один seed → побитово одна сцена.
    do {
        let a = makeLayeredScene(w: 96, h: 64, seed: 42)
        let b = makeLayeredScene(w: 96, h: 64, seed: 42)
        expectEqual(a.background, b.background, "детерминизм: фон")
        expectEqual(a.objects, b.objects, "детерминизм: объекты")
        expectEqual(a.mask, b.mask, "детерминизм: маска")
        expectEqual(a.light, b.light, "детерминизм: свет")
        expectEqual(a.composite, b.composite, "детерминизм: сборка")
    }
    // Разные seed → разные сцены.
    do {
        let a = makeLayeredScene(w: 96, h: 64, seed: 1)
        let b = makeLayeredScene(w: 96, h: 64, seed: 2)
        expect(a.background != b.background, "разные seed → разные фоны")
        expect(a.mask != b.mask, "разные seed → разные маски")
    }
}

func testLayerRanges(_ scene: LayeredScene, _ n: Int) {
    // Диапазоны: всё конечно и в [0,1].
    expect(scene.background.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "фон: значения конечны и в [0,1]")
    expect(scene.objects.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "объекты: значения конечны и в [0,1]")
    expect(scene.light.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "свет: значения конечны и в [0,1]")
    expect(scene.composite.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "сборка: значения конечны и в [0,1]")

    // Маска бинарна и непуста с обеих сторон.
    expect(scene.mask.allSatisfy { $0 == 0 || $0 == 1 }, "маска бинарна {0,1}")
    expect(scene.mask.contains(where: { $0 == 1 }), "маска: объекты есть")
    expect(scene.mask.contains(where: { $0 == 0 }), "маска: фон есть")
    let objPixels = scene.mask.filter { $0 == 1 }.count
    expect(objPixels > 0 && objPixels < n, "маска покрывает часть, а не весь кадр")

}

func testLayerComposition(_ scene: LayeredScene, _ n: Int) {
    // Композиция восстанавливается из слоёв ТОЧНО (формула декодера).
    let rebuilt = composeLayers(w: scene.w, h: scene.h, background: scene.background,
                                objects: scene.objects, mask: scene.mask, light: scene.light)
    expectEqual(rebuilt, scene.composite, "composeLayers(слои) == кадр (побитово)")

    // Формула сборки вручную на подмножестве пикселей.
    do {
        var ok = true
        for i in stride(from: 0, to: n, by: 97) {
            let m = scene.mask[i], l = scene.light[i]
            for c in 0..<3 {
                let v = (scene.background[i * 3 + c] * (1 - m) + scene.objects[i * 3 + c] * m) * l
                let expectV = min(1, max(0, v))
                if abs(scene.composite[i * 3 + c] - expectV) > 1e-6 { ok = false }
            }
        }
        expect(ok, "rgb = (фон·(1−m)+obj·m)·свет — формула выполняется попиксельно")
    }
    // Вне маски объекты нулевые, кадр = фон·свет.
    do {
        var okBg = true, okZero = true
        for i in 0..<n where scene.mask[i] == 0 {
            if scene.objects[i * 3] != 0 || scene.objects[i * 3 + 1] != 0 || scene.objects[i * 3 + 2] != 0 {
                okZero = false
            }
            for c in 0..<3 {
                if abs(scene.composite[i * 3 + c] - min(1, scene.background[i * 3 + c] * scene.light[i])) > 1e-6 {
                    okBg = false
                }
            }
        }
        expect(okZero, "вне маски объекты = 0")
        expect(okBg, "вне маски кадр = фон·свет")
    }

}

func testLightGradient(_ scene: LayeredScene) {
    // Свет: направленный (у солнца ярче) + виньетка → есть перепад.
    do {
        var mn = Float.greatestFiniteMagnitude, mx = -Float.greatestFiniteMagnitude
        for v in scene.light { mn = min(mn, v); mx = max(mx, v) }
        expect(mx - mn > 0.1, "свет: перепад > 0.1 (есть направление)")
        expect(mn >= 0.05, "свет: нет чёрных дыр (min ≥ 0.05)")
    }

}

func testLayerChwLayout(_ scene: LayeredScene, _ n: Int) {
    // CHW-раскладка: 8 каналов, порядок bgR..bgB objR..objB mask light.
    do {
        let ch = scene.channelsCHW()
        expectEqual(ch.count, 8 * n, "CHW: размер 8·H·W")
        var ok = true
        for i in 0..<n {
            if ch[6 * n + i] != scene.mask[i] { ok = false }
            if abs(ch[7 * n + i] - scene.light[i]) > 0 { ok = false }
            if abs(ch[0 * n + i] - scene.background[i * 3]) > 0 { ok = false }
            if abs(ch[3 * n + i] - scene.objects[i * 3]) > 0 { ok = false }
        }
        expect(ok, "CHW: порядок каналов фон|объекты|маска|свет")
    }

}

func testLayer480pSmoke() {
    // 480p smoke: рендерится целиком, макет цел. Порог времени — для -O
    // (штатный scripts/test.sh). Под -Onone Swift-массивы без оптимизации
    // дают ~100× замедления, это не показатель архитектуры.
    do {
        let t0 = Date()
        let big = makeLayeredScene(w: 854, h: 480, seed: 7)
        let dt = Date().timeIntervalSince(t0)
        expectEqual(big.background.count, 854 * 480 * 3, "480p: размер фона")
        expectEqual(big.channelsCHW().count, 854 * 480 * 8, "480p: 8 каналов CHW")
        expect(dt < 5, "480p: рендер < 5 с (фактически \(String(format: "%.2f", dt)) с)")
    }

}

func testLayerPreviewSave(_ scene: LayeredScene) {
    // Предпросмотр: 4 PNG создаются и не пустые.
    do {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("yume_layers_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let prefix = dir.appendingPathComponent("scene").path
        try saveLayerPreview(scene: scene, prefix: prefix)
        for name in ["scene_composite.png", "scene_background.png",
                     "scene_objects.png", "scene_light.png"] {
            let url = dir.appendingPathComponent(name)
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
            expect(size > 100, "превью \(name) записано (\(size) байт)")
        }
    } catch {
        expect(false, "превью слоёв: ошибка \(error)")
    }
}


// MARK: - Универсальная слоистая архитектура


func testUniversalLayers() {
    section("Универсальные слои: любые комбинации экспертов")

    // Фон дополняется объектными экспертами — фон при этом НЕ меняется.
    do {
        let bg = makeComposedScene(w: 96, h: 64, recipe: .backdrop(), seed: 9)
        let ext = makeComposedScene(w: 96, h: 64,
            recipe: .backdrop().adding(LayerSpec(.figure, count: 2, role: .objects)), seed: 9)
        expectEqual(bg.flattened().background, ext.flattened().background,
                    "добавление объектных экспертов не меняет фон (побитово)")
        expect(bg.composite != ext.composite, "добавление объектов меняет кадр")
        expect(ext.flattened().mask.contains(where: { $0 == 1 }), "объекты появились в маске")
        expectEqual(ext.layers.count, bg.layers.count + 2, "каждая фигура = отдельный слой-эксперт")
    }

    // Роль — КТО произвёл слой, а не ЧТО в нём: фон-эксперт может рисовать фигуры.
    do {
        let a = makeComposedScene(w: 96, h: 64, recipe: LayerRecipe([
            LayerSpec(.sky, role: .background),
            LayerSpec(.figure, role: .objects)]), seed: 3)
        let b = makeComposedScene(w: 96, h: 64, recipe: LayerRecipe([
            LayerSpec(.sky, role: .background),
            LayerSpec(.figure, role: .background)]), seed: 3)   // та же семантика
        expectEqual(a.composite, b.composite, "роль эксперта не влияет на композицию")
        expectEqual(a.flattened().mask, b.flattened().mask, "фон-эксперт тоже делает объекты")
    }

    // Любая комбинация: композиция восстанавливается из 8 каналов ТОЧНО.
    let recipes: [(String, LayerRecipe)] = [
        ("пусто", LayerRecipe([])),
        ("только небо", LayerRecipe([LayerSpec(.sky, role: .background)])),
        ("только фигуры", LayerRecipe([LayerSpec(.figure, count: 2, role: .objects)])),
        ("небо+земля+фигуры", LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.ground, role: .background),
            LayerSpec(.figure, count: 3, role: .objects)])),
        ("перекрытие: земля поверх фигур", LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.figure, role: .objects),
            LayerSpec(.ground, role: .background)])),
        ("облако поверх фигур", LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.figure, role: .objects),
            LayerSpec(.clouds, count: 2, role: .background)])),
        ("фон-эксперт делает всё", LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.ground, role: .background),
            LayerSpec(.figure, count: 2, role: .background, group: .objects)])),
        ("только свет", LayerRecipe([LayerSpec(.lightGradient, role: .lighting)])),
        ("свет+фигуры+тени", LayerRecipe([
            LayerSpec(.lightGradient, role: .lighting), LayerSpec(.figure, count: 2, role: .objects),
            LayerSpec(.shadow, count: 2, role: .lighting)])),
        ("солнце+облака", LayerRecipe([
            LayerSpec(.sun, role: .background), LayerSpec(.clouds, count: 4, role: .background)])),
        ("классика", .classic(figureCount: 3)),
    ]
    for (name, recipe) in recipes {
        let scene = makeComposedScene(w: 96, h: 64, recipe: recipe, seed: 77).flattened()
        let rebuilt = composeLayers(w: scene.w, h: scene.h, background: scene.background,
                                    objects: scene.objects, mask: scene.mask, light: scene.light)
        expectEqual(rebuilt, scene.composite, "[\(name)] 8 каналов → кадр побитово")
        expect(scene.mask.allSatisfy { $0 == 0 || $0 == 1 }, "[\(name)] маска бинарна")
        expect(scene.composite.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
               "[\(name)] кадр в [0,1]")
        expect(scene.light.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
               "[\(name)] свет в [0,1]")
    }

    // Перекрытие: земля, отрисованная ПОСЛЕ фигуры, скрывает её — маска это видит.
    do {
        let alone = makeComposedScene(w: 96, h: 64, recipe: LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.figure, role: .objects)]), seed: 5)
            .flattened()
        let occluded = makeComposedScene(w: 96, h: 64, recipe: LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.figure, role: .objects),
            LayerSpec(.ground, role: .background)]), seed: 5).flattened()
        let a = alone.mask.filter { $0 == 1 }.count
        let b = occluded.mask.filter { $0 == 1 }.count
        expect(b < a, "перекрытие: скрытая часть фигур ушла из маски (\(a) → \(b))")
    }

    // Масштаб: 8 облаков + 8 фигур на 480p рендерится целиком (-O).
    do {
        let heavy = LayerRecipe([
            LayerSpec(.sky, role: .background),
            LayerSpec(.clouds, count: 8, role: .background),
            LayerSpec(.ground, role: .background),
            LayerSpec(.figure, count: 8, role: .objects),
            LayerSpec(.lightGradient, role: .lighting),
        ])
        let t0 = Date()
        let s = makeComposedScene(w: 854, h: 480, recipe: heavy, seed: 11).flattened()
        let dt = Date().timeIntervalSince(t0)
        expectEqual(s.channelsCHW().count, 854 * 480 * 8, "480p: 8 каналов при 16 слоях")
        expect(dt < 8, "480p: 16 слоёв < 8 с (\(String(format: "%.2f", dt)) с)")
    }
}

// MARK: - Качество и покадровый план


func testSceneQuality() {
    section("Качество не хуже входа · ускорение движения")

    // PSNR: идентично → ∞, известная ошибка → 20 дБ.
    let ref = [Float](repeating: 0.5, count: 300)
    expect(psnr(reference: ref, candidate: ref) == .infinity, "PSNR идентичных кадров = ∞")
    let shifted = [Float](repeating: 0.6, count: 300)
    expectClose(Double(psnr(reference: ref, candidate: shifted)), 20, 0.01,
                "PSNR при MSE 0.01 = 20 дБ")
    expect(luminanceSSIM(reference: ref, candidate: ref) > 0.999, "SSIM идентичных = 1")
    expect(luminanceSSIM(reference: ref, candidate: shifted) < 1, "SSIM differing < 1")

    // Ворот качества: не хуже входа → принят, регрессия → отклонён.
    do {
        let floor = QualityFloor(minPSNR: 30, minSSIM: 0.90)
        expect(floor.accepts(reference: ref, reconstruction: ref), "пол: идеальная реконструкция принята")
        let bad = [Float](repeating: 0.2, count: 300)
        expect(!floor.accepts(reference: ref, reconstruction: bad), "пол: регрессия отклонена")
        expect(floor.accepts(current: 33, previous: 32), "пол: улучшение принято")
        expect(!floor.accepts(current: 30.1, previous: 32), "пол: просадка > 0.5 дБ отклонена")
        expect(floor.report(reference: ref, reconstruction: ref).contains("принят"),
               "отчёт качества: принят")
    }

    // Покадровый план: статика на ключевых кадрах, объекты — каждый кадр.
    do {
        let plan = FramePlan(keyframeInterval: 8)
        expect(plan.needsRecompute(frame: 0, kind: .sky), "ключевой кадр: небо пересчитано")
        expect(!plan.needsRecompute(frame: 3, kind: .sky), "промежуточный кадр: небо из кэша")
        expect(plan.needsRecompute(frame: 3, kind: .figure), "объекты пересчитываются покадрово")
        expect(plan.needsRecompute(frame: 3, kind: .sky, motionScore: 0.9),
               "смена плана: кэш сброшен")
        let layers: [LayerSpec] = [
            LayerSpec(.sky, role: .background), LayerSpec(.ground, role: .background),
            LayerSpec(.lightGradient, role: .lighting), LayerSpec(.figure, count: 2, role: .objects),
        ]
        expect(plan.estimatedSpeedup(layers: layers) > 1.5,
               "кэш статичных экспертов даёт ускорение > 1.5×")
    }
}

// MARK: - 8-канальный layered VAE: математика потерь (CPU, без Metal)


func testLayeredLossMath() {
    section("Layered loss: композиция + градиенты (CPU)")

    // 1. Точная сборка в CHW-раскладке обязана совпадать с HWC-композитором
    //    рендерера. Это тот самый инвариант composeLayers(flattened) == composite.
    let scene = makeLayeredScene(w: 32, h: 32, seed: 2026)
    let t = LayeredTarget(scene: scene)
    expectEqual(t.channelsCHW.count, 8 * 32 * 32, "таргет: 8·w·h значений")
    expectEqual(t.compositeCHW.count, 3 * 32 * 32, "композит: 3·w·h значений")
    let exactCHW = composeCHW(w: 32, h: 32, recon: t.channelsCHW)
    expect(maxAbsDiff(exactCHW, t.compositeCHW) < 1e-6,
           "composeCHW(таргет) == composite поканально")
    let exactHWC = composeHWC(w: 32, h: 32, recon: t.channelsCHW)
    expect(maxAbsDiff(exactHWC, scene.composite) < 1e-6,
           "composeHWC(таргет) == composite кадра")

    // 2. Идеальная реконструкция → нулевая потеря, возмущённая → > 0.
    expectClose(layeredLoss(recon: t.channelsCHW, target: t), 0, 1e-9,
                "идеальная реконструкция: loss = 0")
    var bad = t.channelsCHW
    for i in 0..<bad.count { bad[i] = min(1, max(0, bad[i] + 0.05)) }
    expect(layeredLoss(recon: bad, target: t) > 0, "испорченная реконструкция: loss > 0")

    // 3. Аналитический градиент == конечная разность по ВСЕМ 8 каналам,
    //    включая маску и свет: именно они несут композиционный член.
    let n = 4 * 4
    let smallChannels: [Float] = (0..<(8 * n)).map {
        0.25 + 0.03 * Float($0 / n) + 0.01 * Float($0 % n)
    }
    let smallComposite: [Float] = [Float](repeating: 0.42, count: 3 * n)
    let small = LayeredTarget(w: 4, h: 4,
                              channelsCHW: smallChannels,
                              compositeCHW: smallComposite)
    var recon = [Float](repeating: 0, count: 8 * n)
    for i in 0..<recon.count { recon[i] = 0.30 + 0.02 * Float(i % 7) }
    let g = layeredLossGradient(recon: recon, target: small,
                                layerWeight: 1, composeWeight: 1)
    expectEqual(g.count, 8 * n, "градиент: 8·w·h значений")
    let eps: Float = 1e-3
    var worst = 0.0
    for ch in 0..<8 {
        for i in [0, 5, 11] {
            let idx = ch * n + i
            var rp = recon, rm = recon
            rp[idx] += eps
            rm[idx] -= eps
            let lp = layeredLoss(recon: rp, target: small,
                                 layerWeight: 1, composeWeight: 1)
            let lm = layeredLoss(recon: rm, target: small,
                                 layerWeight: 1, composeWeight: 1)
            let numeric: Double = (lp - lm) / (2 * Double(eps))
            let analytic = Double(g[idx])
            let denom = max(1e-6, abs(analytic) + abs(numeric))
            let rel = abs(analytic - numeric) / denom
            worst = max(worst, rel)
            expect(rel < 0.05,
                   String(format: "канал %d пикс %d: аналит %.3e / числ %.3e (%.2f%%)",
                          ch, i, analytic, numeric, rel * 100))
        }
    }
    print(String(format: "      макс. относительная ошибка градиента: %.2f%%", worst * 100))

    // 4. Композиционный член реально влияет на градиенты маски, света и фона.
    let gNoComp = layeredLossGradient(recon: recon, target: small,
                                      layerWeight: 1, composeWeight: 0)
    expect(maxAbsDiff(Array(g[6 * n..<8 * n]), Array(gNoComp[6 * n..<8 * n])) > 1e-6,
           "без composeWeight градиенты маски/света меняются")
    expect(maxAbsDiff(Array(g[0..<3 * n]), Array(gNoComp[0..<3 * n])) > 1e-6,
           "composeWeight влияет и на каналы фона")

    // 5. Пол качества считается по СОБРАННОМУ кадру, а не по слоям по отдельности.
    let q = composedQuality(recon: t.channelsCHW, target: t)
    expect(q.psnr == .infinity && q.ssim > 0.999, "идеальная сборка: PSNR ∞ / SSIM 1")
    let qBad = composedQuality(recon: bad, target: t)
    expect(qBad.psnr.isFinite && qBad.psnr < 40, "испорченная сборка: PSNR конечен")

    // 6. Бюджет памяти layered-пути учитывает дополнительные буферы.
    expect(layeredActivationBytes(w: 32, h: 32) > 0, "layered-буферы учтены в бюджете")
    expectEqual(layeredActivationBytes(w: 64, h: 64),
                4 * layeredActivationBytes(w: 32, h: 32),
                "память layered-буферов ∝ площади")
}

// MARK: - 8-канальный layered VAE: Metal-ядро градиента


func testLayeredLossGPU() {
    section("Layered loss: Metal-ядро градиента")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24, n = w * h
    let t = LayeredTarget(scene: makeLayeredScene(w: w, h: h, seed: 77))
    let tg = LayeredTargetGPU(engine: engine, target: t)

    // Возмущённая реконструкция строго внутри (0,1): кламп не зануляет градиент.
    var recon = [Float](repeating: 0, count: 8 * n)
    for i in 0..<recon.count { recon[i] = 0.30 + 0.02 * Float(i % 7) }
    let r = GPUTensor(engine: engine, data: recon, rows: 8)

    session.begin()
    let gpuGrad = session.layeredGrad(recon: r, target: tg,
                                      layerWeight: 1, composeWeight: 1)
    _ = session.end()

    let got = gpuGrad.toArray()
    let cpu = layeredLossGradient(recon: recon, target: t,
                                  layerWeight: 1, composeWeight: 1)
    expectEqual(got.count, 8 * n, "Metal: форма градиента 8·w·h")
    expect(got.allSatisfy { $0.isFinite }, "Metal: градиент без NaN")
    expect(maxAbsDiff(got, cpu) < 1e-5,
           String(format: "Metal-градиент == CPU-аналитике (max Δ %.2e)", maxAbsDiff(got, cpu)))
    expect(got.contains { abs($0) > 1e-7 }, "Metal: градиент невырожден")
}

// MARK: - 8-канальный layered VAE: форма, память, чекпоинт


func testLayeredVAEShape() {
    section("Layered VAE: 8 каналов вместо 3")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let t = LayeredTarget(scene: makeLayeredScene(w: w, h: h, seed: 11))
    let x = GPUTensor(engine: engine, data: t.channelsCHW, rows: 8)

    let vae3 = ConvVAEGPU(engine: engine)
    let vae8 = ConvVAEGPU(engine: engine, channels: 8)
    expectEqual(vae3.channels, 3, "старый VAE остался 3-канальным")
    expectEqual(vae8.channels, 8, "новый VAE: 8 каналов")
    expectEqual(vae8.e1.ic, 8, "e1 принимает 8 каналов")
    expectEqual(vae8.d3.oc, 8, "d3 выдаёт 8 каналов")
    expectClose(ConvVAEGPU.layeredCompressionRatio, 32, 1e-9,
                "сжатие 8-канального входа = 32×")
    expectClose(ConvVAEGPU.compressionRatio, 12, 1e-9,
                "сжатие 3-канального входа = 12×")

    session.begin()
    let cache = vae8.forward(session, image: x, w: w, h: h, deterministic: true)
    _ = session.end()
    expectEqual(cache.recon.rows, 8, "реконструкция: 8 каналов")
    expectEqual(cache.recon.cols, w * h, "реконструкция: w×h")
    let r = cache.recon.toArray()
    expect(r.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "8 каналов в [0,1] (сигмоида)")
    expectEqual(cache.z.rows, ConvVAEGPU.latentC, "латент не изменился")

    // Бюджет VRAM: 8 каналов + layered-буферы дороже, но кадр всё ещё влезает.
    let b3 = convVAEBytesPerSample(w: 856, h: 480)
    let b8 = convVAEBytesPerSample(w: 856, h: 480, channels: 8, layered: true)
    expect(b8 > b3, "8-канальный путь требует больше активаций")
    expect(Double(b8) / 1_048_576 < 650, "480p layered-кадр укладывается в бюджет")
    expect(effectiveBatchSize(requested: 16, w: 856, h: 480,
                              budgetBytes: 8 * 1024 * 1024 * 1024 / 2,
                              channels: 8, layered: true) >= 1,
           "layered-пачка ≥ 1")

    // Старый 3-канальный чекпоинт несовместим с 8-канальной моделью:
    // это не битый файл, а другая архитектура (код 11).
    do {
        try vae8.load(from: vae3.matrices)
        expect(false, "3-канальный чекпоинт отклонён 8-канальной моделью")
    } catch {
        expect(isIncompatibleCheckpoint(error), "несовместимость распознана (код 11)")
        expectEqual((error as NSError).code, 11, "код ошибки = 11")
    }
}

// MARK: - 8-канальный layered VAE: градиенты сети vs конечные разности


func testLayeredVAEGradients() {
    section("Layered VAE: градиенты vs конечные разности")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let beta: Float = 0.05
    let t = LayeredTarget(scene: makeLayeredScene(w: w, h: h, seed: 33))
    let tg = LayeredTargetGPU(engine: engine, target: t)
    let x = GPUTensor(engine: engine, data: t.channelsCHW, rows: 8)
    let vae = ConvVAEGPU(engine: engine, channels: 8)

    /// Полная потеря: MSE по 8 слоям + MSE собранного кадра + β·KL.
    /// Именно её должен дифференцировать `layered_grad_kernel`.
    func loss() -> Double {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
        _ = session.end()
        return layeredVAELoss(cache: c, target: t,
                              layerWeight: 1, composeWeight: 1, beta: beta)
    }

    // Небольшая тренировка, чтобы веса ушли от инициализации.
    for step in 1...12 {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
        let (_, g) = vae.backwardLayered(session, c, target: tg, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: 0.002)
        _ = session.end()
    }

    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
    let (_, grads) = vae.backwardLayered(session, cache, target: tg, beta: beta)
    _ = session.end()

    var maxGrad = 0.0
    for g in grads {
        maxGrad = max(maxGrad, g.toArray().map { abs(Double($0)) }.max() ?? 0)
    }
    expect(maxGrad > 1e-6,
           String(format: "layered-градиенты невырождены (max |g| = %.2e)", maxGrad))

    let eps: Float = 1e-2
    var checked = 0
    for layerIdx in [2, 3] {   // e3 — перед латентом, d1 — сразу после
        let g = grads[layerIdx * 2].toArray()
        let top = (0..<g.count).sorted { abs(g[$0]) > abs(g[$1]) }.prefix(2)
        for weightIdx in top {
            let layer = vae.layers[layerIdx]
            let analytic = Double(g[weightIdx])
            let orig = layer.W.getValue(weightIdx)
            layer.W.setValue(weightIdx, orig + eps)
            let lp = loss()
            layer.W.setValue(weightIdx, orig - eps)
            let lm = loss()
            layer.W.setValue(weightIdx, orig)

            let numeric = (lp - lm) / (2 * Double(eps))
            let rel = abs(analytic - numeric) / max(1e-9, abs(analytic) + abs(numeric))
            expect(rel < 0.12,
                   String(format: "слой %d вес %d: аналит %.4e числ %.4e (отн %.1f%%)",
                          layerIdx, weightIdx, analytic, numeric, rel * 100))
            checked += 1
        }
    }
    expectEqual(checked, 4, "проверено 4 точки")
}

// MARK: - 8-канальный layered VAE: тренер и пол качества


