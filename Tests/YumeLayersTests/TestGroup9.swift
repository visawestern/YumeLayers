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

func testChunkPlanAndMemoryBudgets() {
    section("Планировщик чанков ffmpeg и бюджеты памяти")

    // План чанков: покрывает [0, target) без дыр, растёт 60→120→240→1920.
    do {
        let plan = ParallelExtractor.chunkPlan(target: 500)
        expectEqual(plan.map { $0.frameCount }, [60, 120, 240, 80],
                    "чанки: рост 60→120→240, хвост неполный")
        expectEqual(plan.map { $0.startFrame }, [0, 60, 180, 420],
                    "чанки: стыки без дыр и перекрытий")
        expectEqual(plan.reduce(0) { $0 + $1.frameCount }, 500, "чанки: сумма = target")

        let one = ParallelExtractor.chunkPlan(target: 50)
        expectEqual(one.count, 1, "чанки: маленький ролик — один чанк")
        expect(ParallelExtractor.chunkPlan(target: 0).isEmpty, "чанки: target=0 → пусто")

        let big = ParallelExtractor.chunkPlan(target: 100_000)
        expect(big.allSatisfy { $0.frameCount <= 1920 }, "чанки: потолок размера 1920")
        expectEqual(big.reduce(0) { $0 + $1.frameCount }, 100_000,
                    "чанки: 100k покрывается целиком")
    }

    // Порция дозагрузки кадров в VRAM: [64, 1024], от бюджетов RAM/GPU.
    do {
        let fb = 854 * 480 * 3 * 4
        expectEqual(frameLoadChunkSize(frameBytes: 0, availableRAM: 32 << 30,
                                       maxWorkingSet: 8 << 30),
                    64, "порция: frameBytes=0 → консервативные 64")
        expectEqual(frameLoadChunkSize(frameBytes: fb, availableRAM: 64 << 30,
                                       maxWorkingSet: 64 << 30),
                    1024, "порция: огромные бюджеты → потолок 1024")
        // Нижняя граница — когда бюджета не хватает даже на пару кадров:
        // 50% от 64 МБ = 32 МБ; 32 МБ / (2 × 4.9 МБ) ≈ 3 → зажим до 64.
        expectEqual(frameLoadChunkSize(frameBytes: fb, availableRAM: 64 << 30,
                                       maxWorkingSet: 64 << 20),
                    64, "порция: малый GPU-бюджет → нижняя граница 64")
        // 25% RAM против 50% рабочего набора: берётся минимум.
        // 32 ГБ RAM → 8 ГиБ; 8 ГБ GPU → 4 ГиБ; 4 ГиБ / (2 × 4.9 МБ) = 436.
        expectEqual(frameLoadChunkSize(frameBytes: fb, availableRAM: 32 << 30,
                                       maxWorkingSet: 8 << 30),
                    436, "порция: 32 ГБ RAM / 8 ГБ GPU → 436 кадров")
    }
}

// MARK: - Рендер: солнце (регрессия инвертированного smoothstep)


func testSunDiscRegression() {
    section("Рендер: солнце — компактный диск, а не всё небо")

    let sc = makeLayeredScene(w: 160, h: 120, seed: 2026)
    let n = sc.w * sc.h
    var warm = 0
    // Солнце — слой группы .background: в sc.objects его нет (там только
    // фигуры), поэтому ищем тёплый диск именно в фоне. Раньше проверка
    // смотрела в objects и не могла увидеть регрессию вообще.
    for i in 0..<n {
        let r = sc.background[i * 3], g = sc.background[i * 3 + 1], b = sc.background[i * 3 + 2]
        // Солнце: (1.0, 0.97, 0.88). Облака (0.98, 0.99, 1.0) отсекаются
        // синим каналом, небо — тоже (синий канал высокий).
        if r > 0.95 && g > 0.92 && b < 0.95 { warm += 1 }
    }
    let frac = Double(warm) / Double(n)
    expect(warm >= 5, "солнце: тёплые пиксели в фоне есть (\(warm))")
    expect(frac < 0.15,
           "солнце: компактный диск — \(String(format: "%.3f", frac)) кадра (< 0.15); раньше инвертированный smoothstep заливал солнечным цветом всё небо")
}

// MARK: - Стабилизация: крошечные кадры (регрессия выхода за границы)


func testStabilizeTinyFrames() {
    section("Стабилизация: крошечные кадры не ломают границы выборки")

    // 12×12: раньше availW=6, cw=max(8, 0)=8 > availW → offX<0 → sx выходил
    // за [0, w) и чтение шло за границей буфера (крэш). Теперь сдвиг зажат
    // запасом на кратность 8.
    do {
        var rng = SplitMix64(seed: 3)
        let frames = (0..<5).map { _ in makeSyntheticScene(w: 12, h: 12, rng: &rng) }
        let (aligned, cw, ch, report) = stabilizeFrames(frames, w: 12, h: 12, maxShift: 3)
        expect(cw % 8 == 0 && ch % 8 == 0, "12×12: выход кратен 8 (\(cw)×\(ch))")
        expect(cw <= 12 && ch <= 12, "12×12: выход не больше исходного кадра")
        expect(aligned.allSatisfy { $0.count == cw * ch * 3 },
               "12×12: размер выровненных кадров сходится")
        expect(!aligned.isEmpty, "12×12: хотя бы один кадр остался")
        expectEqual(report.sourceWidth, 12, "12×12: отчёт ссылается на источник")
    }

    // 16×16: поведение прежнее (умещается и сдвиг, и кратность 8).
    do {
        var rng = SplitMix64(seed: 7)
        let frames = (0..<4).map { _ in makeSyntheticScene(w: 16, h: 16, rng: &rng) }
        let (aligned, cw, ch, _) = stabilizeFrames(frames, w: 16, h: 16, maxShift: 8)
        expectEqual(cw, 8, "16×16: выход 8×8 (сдвиг ±4, область 8)")
        expectEqual(ch, 8, "16×16: выход 8×8")
        expect(!aligned.isEmpty, "16×16: кадры остались")
    }

    // 8×8: на сдвиг места нет вовсе — честный ранний выход без изменений.
    do {
        var rng = SplitMix64(seed: 11)
        let frames = (0..<3).map { _ in makeSyntheticScene(w: 8, h: 8, rng: &rng) }
        let (aligned, cw, ch, _) = stabilizeFrames(frames, w: 8, h: 8, maxShift: 8)
        expectEqual(aligned.count, 3, "8×8: кадры без изменений")
        expectEqual(cw, 8, "8×8: размер без изменений")
        expectEqual(ch, 8, "8×8: размер без изменений")
    }
}

// MARK: - GPUFramePool: битый кадр (регрессия «фантома»)


func testChannelLayout() {
    section("Канальная раскладка: HWC ↔ CHW")

    // 1. hwcToChw ставит каналы в планарный (CHW) порядок.
    do {
        // Кадр 2×1: пиксель 0 = (R0,G0,B0) = (1,2,3), пиксель 1 = (R1,G1,B1).
        let hwc: [Float] = [1, 2, 3, 4, 5, 6]
        let chw = hwcToChw(hwc)
        // CHW: R-плоскость [1,4], G [2,5], B [3,6].
        expectEqual(chw, [1, 4, 2, 5, 3, 6], "hwcToChw: планарная раскладка")
    }

    // 2. Круговая поездка — тождество.
    do {
        var rng = SplitMix64(seed: 99)
        let img = makeSyntheticScene(w: 16, h: 12, rng: &rng)
        expectEqual(maxAbsDiff(chwToHwc(hwcToChw(img)), img), 0,
                    "chwToHwc∘hwcToChw = тождество")
        expectEqual(maxAbsDiff(hwcToChw(chwToHwc(img)), img), 0,
                    "hwcToChw∘chwToHwc = тождество")
    }

    // 3. Поведенческий: с правильной CHW-раскладкой и малым β модель учится
    //    сохранять яркость кадра. В HWC-кормушке (каналы перепутаны) этого
    //    не происходило — реконструкция схлопывалась в тёмный шум.
    guard gHasMetal else { skip("Metal недоступен"); return }
    do {
        let engine = MetalEngine.shared
        let session = GpuSession(engine)
        let w = 32, h = 24
        var rng = SplitMix64(seed: 7)
        let hwc = makeSyntheticScene(w: w, h: h, rng: &rng)
        let chw = hwcToChw(hwc)
        let x = GPUTensor(engine: engine, data: chw, rows: 3)
        let vae = ConvVAEGPU(engine: engine)
        let beta: Float = 0.01
        let steps = 200
        for step in 1...steps {
            session.begin()
            let c = vae.forward(session, image: x, w: w, h: h,
                                deterministic: false, seed: UInt32(step))
            let (_, g) = vae.backward(session, c, target: x, beta: beta)
            vae.applyAdam(session, grads: g, t: step, lr: 0.01)
            _ = session.end()
        }
        func lumaMean(_ hwc: [Float]) -> Double {
            let n = hwc.count / 3
            var s = 0.0
            for i in 0..<n {
                s += 0.299 * Double(hwc[i * 3])
                    + 0.587 * Double(hwc[i * 3 + 1])
                    + 0.114 * Double(hwc[i * 3 + 2])
            }
            return s / Double(n)
        }
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        _ = session.end()
        let reconHWC = chwToHwc(c.recon.toArray())
        let targetLuma = lumaMean(hwc)
        let reconLuma = lumaMean(reconHWC)
        expect(abs(reconLuma - targetLuma) < 0.30,
               String(format: "CHW-кормушка: яркость реконструкции близка к исходной (%.2f vs %.2f)",
                      reconLuma, targetLuma))
    }
}

// MARK: - GPUFramePool: битый кадр (регрессия «фантома»)


func testGPUFramePoolBrokenFrame() {
    section("GPUFramePool: битый кадр → чёрная заглушка, не мусор")

    guard MTLCreateSystemDefaultDevice() != nil else {
        skip("Metal недоступен"); return
    }
    let engine = MetalEngine.shared

    func solidFrame(_ v: Float) -> MemoryFrame {
        let img = [Float](repeating: v, count: 8 * 8 * 3)
        let png = encodePNG(width: 8, height: 8, rgb: img)!
        return MemoryFrame(name: "f\(v)", jpeg: png)
    }
    // Кадр 1 — мусор вместо JPEG: раньше слот пула оставался с байтами
    // предыдущего окна («фантом»), а непроинициализированный приватный
    // буфер вообще содержал случайные значения.
    let frames = [solidFrame(0.1),
                  MemoryFrame(name: "bad", jpeg: Data([0x12, 0x34, 0x56])),
                  solidFrame(0.9)]
    guard let pool = GPUFramePool(engine: engine, frames: frames, w: 8, h: 8, size: 2) else {
        expect(false, "пул не создался"); return
    }

    let a0 = pool.next().toArray()
    expect(a0.allSatisfy { abs($0 - 0.1) < 0.02 },
           "пул: валидный кадр декодирован (значение ~0.1)")

    let a1 = pool.next().toArray()
    expect(a1.allSatisfy { $0 == 0 },
           "пул: битый кадр — чёрная заглушка (нули), не мусор и не чужой кадр")
}

// MARK: - main


