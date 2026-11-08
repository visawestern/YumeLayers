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

func testCheckpointQualityDiag() {
    section("Диагностика: качество обученного чекпоинта")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let env = ProcessInfo.processInfo.environment
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    let ckpt = env["YUME_DIAG_CKPT"] ?? (home + "/YumeLayers/checkpoints/conv.safetensors")
    guard FileManager.default.fileExists(atPath: ckpt) else {
        skip("нет чекпоинта \(ckpt)"); return
    }
    let ref = env["YUME_DIAG_IMAGE"] ?? "/tmp/ref_a.png"
    guard FileManager.default.fileExists(atPath: ref) else {
        skip("нет опорной картинки \(ref)"); return
    }
    let w = Int(env["YUME_DIAG_W"] ?? "") ?? 80
    let h = Int(env["YUME_DIAG_H"] ?? "") ?? 56
    guard let original = loadImageRGB(path: ref, w: w, h: h) else {
        expect(false, "картинка прочитана"); return
    }
    let vae = ConvVAEGPU(engine: MetalEngine.shared)
    do {
        let state = try loadStateWithMeta(path: ckpt)
        print(String(format: "диагностика: чекпоинт эпоха %d, шаг %d, отпечаток %@",
                     state.epoch, state.step, state.fingerprint ?? "нет"))
        try vae.load(from: state.mats)
    } catch {
        expect(false, "веса загружены: \(error)"); return
    }
    // Путь 1: полный forward (как при обучении и реконструкциях).
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let input = GPUTensor(engine: engine, rows: 3, cols: w * h)
    session.begin()
    session.upload(input, from: hwcToChw(original))
    let cache = vae.forward(session, image: input, w: w, h: h, deterministic: true)
    session.end()
    let forwardRecon = chwToHwc(cache.recon.toArray())
    // Путь 2: раздельно encode + decode (как в интерполяции).
    let mu = (try? encodeLatent(vae: vae, image: original, w: w, h: h)) ?? []
    let recon = (try? decodeLatent(vae: vae, latent: mu, w: w, h: h)) ?? []
    expectEqual(mu.count, ConvVAEGPU.latentC * (h / 8) * (w / 8), "латент: C×(h/8)×(w/8)")
    expectEqual(recon.count, original.count, "реконструкция: тот же размер")

    func psnr(_ a: [Float], _ b: [Float]) -> Double {
        var sum = 0.0
        for i in 0..<min(a.count, b.count) {
            let d = Double(a[i] - b[i])
            sum += d * d
        }
        let mse = sum / Double(max(min(a.count, b.count), 1))
        return mse > 0 ? 10 * log10(1.0 / mse) : 99
    }
    let forwardPSNR = psnr(forwardRecon, original)
    let splitPSNR = psnr(recon, original)
    var same = forwardRecon.count == recon.count
    if same {
        for i in 0..<forwardRecon.count where abs(forwardRecon[i] - recon[i]) > 1e-5 {
            same = false; break
        }
    }
    print(String(format: "диагностика: PSNR forward = %.2f дБ, encode+decode = %.2f дБ, совпадают = %@",
                 forwardPSNR, splitPSNR, same ? "да" : "нет"))
    expect(forwardPSNR > 20, String(format: "PSNR forward = %.2f дБ (ожидается > 20)", forwardPSNR))
    expect(same, "forward и encode+decode дают одинаковый кадр")

    // Путь 3: forward СО ШУМОМ (как в трейнинге: eps = randn(seed), не 0).
    // Сравниваем разные seed: если качество зависит от seed, значит модель
    // обучалась с «недетерминированным» z = mu + sigma*eps, и оценка через
    // eps = 0 (детерминированный путь) нерепрезентативна.
    var sampledPSNRs: [String] = []
    for seed in [UInt32(0), 1, 2] {
        let s2 = GpuSession(engine)
        let in2 = GPUTensor(engine: engine, rows: 3, cols: w * h)
        s2.begin()
        s2.upload(in2, from: hwcToChw(original))
        let c2 = vae.forward(s2, image: in2, w: w, h: h, deterministic: false, seed: seed)
        s2.end()
        let r2 = chwToHwc(c2.recon.toArray())
        let p = psnr(r2, original)
        sampledPSNRs.append(String(format: "seed %u: %.2f дБ", seed, p))
    }
    print("диагностика: PSNR со шумом — " + sampledPSNRs.joined(separator: ", "))
}

// MARK: - GPU-MSE: совпадает ли число в логе с реальной ошибкой

/// Сумма квадратов разностей, посчитанная на CPU по тем же двум массивам.
private func cpuSumSquares(_ a: [Float], _ b: [Float]) -> Double {
    var sum = 0.0
    for i in 0..<min(a.count, b.count) {
        let d = Double(a[i] - b[i])
        sum += d * d
    }
    return sum
}

/// GPU-редукция MSE обязана совпадать с CPU-суммой.
///
/// Именно это число (`-10·log10(mse)`) печатается в логе как PSNR. Если
/// редукция врёт, весь лог обучения недостоверен, а качество на диске
/// выглядит « inexplicably » хуже.

func testGPUMseMatchesCPU() {
    section("GPU-MSE совпадает с CPU-суммой квадратов")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let w = 80, h = 56
    let n = 3 * w * h
    var a = [Float](repeating: 0, count: n)
    var b = [Float](repeating: 0, count: n)
    for i in 0..<n {
        a[i] = Float((i * 37) % 251) / 251
        b[i] = Float((i * 91) % 241) / 241
    }
    let ta = GPUTensor(engine: engine, data: a, rows: 3)
    let tb = GPUTensor(engine: engine, data: b, rows: 3)
    let session = GpuSession(engine)
    session.begin()
    let buf = session.encodeMSE(recon: ta, target: tb)
    _ = session.end()
    let gpuSum = Double(buf.contents().assumingMemoryBound(to: Float.self)[0])
    let cpuSum = cpuSumSquares(a, b)
    let rel = abs(gpuSum - cpuSum) / max(cpuSum, 1e-12)
    print(String(format: "MSE-сумма: GPU = %.8f, CPU = %.8f, расхождение = %.3e",
                 gpuSum, cpuSum, rel))
    expect(rel < 1e-5, String(format: "GPU-MSE = CPU-MSE (расхождение %.3e)", rel))
}

/// Round-trip весов: обучили → сохранили → загрузили в новую модель.
///
/// Если веса теряются или искажаются при сериализации, лог обучения
/// показывает одно качество, а на диске лежит совсем другое.

func testCheckpointRoundTrip() {    section("Чекпоинт: round-trip весов без потерь")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let w = 80, h = 56
    let vae = ConvVAEGPU(engine: engine)

    // Простой синтетический кадр (градиент цвета + пятно).
    var img = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            let i = (y * w + x) * 3
            img[i] = Float(x) / Float(w)
            img[i + 1] = Float(y) / Float(h)
            img[i + 2] = (x + y) % 17 == 0 ? 1.0 : 0.25
        }
    }
    let targetCHW = hwcToChw(img)
    let target = GPUTensor(engine: engine, data: targetCHW, rows: 3)
    let session = GpuSession(engine)

    // 24 шага обучения — веса должны уйти от инициализации.
    for step in 1...24 {
        session.begin()
        let cache = vae.forward(session, image: target, w: w, h: h,
                                deterministic: false, seed: UInt32(step))
        let (_, grads) = vae.backward(session, cache, target: target, beta: rgbKLBeta)
        vae.applyAdam(session, grads: grads, t: step, lr: 0.002)
        _ = session.end()
    }

    // Кадр из живой модели.
    // ВАЖНО: `toArray()` читает отдельным command buffer — его нельзя
    // вызывать, пока сессия не завершена (`end()`), иначе читаются данные
    // ДО выполнения forward.
    session.begin()
    let cacheMem = vae.forward(session, image: target, w: w, h: h,
                                deterministic: false, seed: 777)
    _ = session.end()
    let reconMem = cacheMem.recon.toArray()

    // Тот же кадр, но посчитанный GPU-редкукцией (как в логе).
    session.begin()
    let mseBuf = session.encodeMSE(recon: cacheMem.recon, target: target)
    _ = session.end()
    let gpuSum = Double(mseBuf.contents().assumingMemoryBound(to: Float.self)[0])
    let cpuSum = cpuSumSquares(reconMem, targetCHW)
    let rel = abs(gpuSum - cpuSum) / max(cpuSum, 1e-12)
    print(String(format: "после обучения: MSE-сумма GPU = %.8f, CPU = %.8f, расхождение = %.3e",
                 gpuSum, cpuSum, rel))
    expect(rel < 1e-5, String(format: "лог-PSNR соответствует реальной ошибке (%.3e)", rel))

    // Сохранение → загрузка в новую модель → тот же кадр.
    let path = NSTemporaryDirectory() + "yume_roundtrip_\(UUID().uuidString).safetensors"
    defer { try? FileManager.default.removeItem(atPath: path) }
    let matsMem = vae.matrices
    do {
        try saveState(matsMem, to: path, epoch: 0, step: 24)
    } catch {
        expect(false, "чекпоинт сохранён: \(error)")
        return
    }
    let vae2 = ConvVAEGPU(engine: engine)
    var matsFile: [Matrix] = []
    do {
        let state = try loadStateWithMeta(path: path)
        matsFile = state.mats
        try vae2.load(from: state.mats)
    } catch {
        expect(false, "чекпоинт загружен: \(error)")
        return
    }
    let matsLoaded = vae2.matrices

    func maxMatrixDiff(_ a: [Matrix], _ b: [Matrix]) -> Double {
        var worst = 0.0
        for i in 0..<min(a.count, b.count) {
            guard a[i].data.count == b[i].data.count else { return .infinity }
            for k in 0..<a[i].data.count {
                worst = max(worst, Swift.abs(Double(a[i].data[k] - b[i].data[k])))
            }
        }
        return worst
    }
    let fileDiff = maxMatrixDiff(matsMem, matsFile)
    let gpuDiff = maxMatrixDiff(matsFile, matsLoaded)
    print(String(format: "веса: память↔файл max diff = %.3e, файл↔GPU max diff = %.3e",
                 fileDiff, gpuDiff))
    expect(fileDiff == 0, String(format: "файл весов совпадает с памятью (%.3e)", fileDiff))
    expect(gpuDiff < 1e-6, String(format: "загруженные веса совпадают с файлом (%.3e)", gpuDiff))

    session.begin()
    let cacheLoaded = vae2.forward(session, image: target, w: w, h: h,
                                    deterministic: false, seed: 777)
    _ = session.end()
    let reconLoaded = cacheLoaded.recon.toArray()

    expectEqual(reconLoaded.count, reconMem.count, "размер реконструкции совпадает")
    var maxDiff = 0.0
    for i in 0..<min(reconMem.count, reconLoaded.count) {
        maxDiff = max(maxDiff, Swift.abs(Double(reconMem[i] - reconLoaded[i])))
    }
    print(String(format: "round-trip: максимальное расхождение кадра = %.3e", maxDiff))
    expect(maxDiff < 1e-4, String(format: "round-trip без потерь (%.3e)", maxDiff))
}

/// Шум VAE обязан МЕНЯТЬСЯ от шага к шагу и от кадра пачки.
///
/// С фиксированным seed модель выучивает один вектор шума и не работает с
/// остальными; оценка через ε = 0 (интерполяция) тогда сильно расходится с
/// тем, что показывает обучение. Это и есть корень «26 дБ в логе vs 8 дБ
/// на диске».

func testTrainingNoiseSeeds() {
    section("Шум обучения: уникальный seed на шаг и кадр")

    // Разные шаги — разные seed.
    expect(trainingNoiseSeed(step: 1, batchIndex: 0) != trainingNoiseSeed(step: 2, batchIndex: 0),
           "разные шаги → разные seed")

    // Разные кадры пачки — разные seed.
    expect(trainingNoiseSeed(step: 1, batchIndex: 0) != trainingNoiseSeed(step: 1, batchIndex: 1),
           "разные кадры пачки → разные seed")

    // Воспроизводимость: один и тот же вызов даёт один seed.
    expectEqual(trainingNoiseSeed(step: 42, batchIndex: 3),
                trainingNoiseSeed(step: 42, batchIndex: 3),
                "детерминированность")

    // Покрытие: на диапазоне шагов не должно быть одного повторяющегося seed.
    var seen = Set<UInt32>()
    for s in 0..<500 {
        seen.insert(trainingNoiseSeed(step: s, batchIndex: 0))
    }
    expect(seen.count >= 490, String(format: "разнообразие seed (найдено %d из 500)", seen.count))

    // Разгон скорости: с шага 1 нарастает к 1, затем держится на 1.
    expectEqual(resumeWarmupFactor(step: 1, warmupSteps: 1000), 0.001, "warmup шаг 1 = 0.001")
    expectEqual(resumeWarmupFactor(step: 500, warmupSteps: 1000), 0.5, "warmup шаг 500 = 0.5")
    expectEqual(resumeWarmupFactor(step: 1000, warmupSteps: 1000), 1.0, "warmup шаг 1000 = 1.0")
    expectEqual(resumeWarmupFactor(step: 2000, warmupSteps: 1000), 1.0, "после warmup = 1.0")
    expectEqual(resumeWarmupFactor(step: 1, warmupSteps: 0), 1.0, "warmup выключен")
}

/// Защищённый путь финальной модели и выбор лучшей.

func testFinalCheckpointPath() {
    section("Защищённый финал: путь и выбор лучшей модели")

    let fm = FileManager.default
    let tmp = NSTemporaryDirectory() + "yume_final_check_\(UUID().uuidString)"
    defer { try? fm.removeItem(atPath: tmp) }
    try? fm.createDirectory(atPath: tmp, withIntermediateDirectories: true)
    let conv = tmp + "/conv.safetensors"

    expectEqual(finalCheckpointPath(for: conv),
                tmp + "/conv-final.safetensors",
                "финал рядом с рабочим чекпоинтом")
    expectEqual(finalCheckpointPath(for: "/tmp/model.ckpt"),
                "/tmp/model-final.ckpt",
                "произвольное расширение")

    // Если финала нет — используем рабочий файл.
    try? Data([1]).write(to: URL(fileURLWithPath: conv))
    expectEqual(bestCheckpointPath(for: conv), conv, "нет финала → рабочий файл")

    // Если финал есть — он приоритетнее (он всегда новее рабочего).
    let fin = finalCheckpointPath(for: conv)
    try? Data([2]).write(to: URL(fileURLWithPath: fin))
    expectEqual(bestCheckpointPath(for: conv), fin, "есть финал → выбираем его")
}

/// Round-trip состояния оптимизатора: моменты Adam переживают save/load.

func testOptimizerStateRoundTrip() {
    section("Состояние оптимизатора: round-trip")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine)
    let w = 80, h = 56
    var img = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            let i = (y * w + x) * 3
            img[i] = Float(x) / Float(w)
            img[i + 1] = Float(y) / Float(h)
            img[i + 2] = 0.4
        }
    }
    let target = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
    let session = GpuSession(engine)
    for step in 1...8 {
        session.begin()
        let cache = vae.forward(session, image: target, w: w, h: h,
                                deterministic: false, seed: UInt32(step))
        let (_, grads) = vae.backward(session, cache, target: target, beta: rgbKLBeta)
        vae.applyAdam(session, grads: grads, t: step, lr: 0.002)
        _ = session.end()
    }

    let momentsMem = vae.optimizerMoments()
    let path = NSTemporaryDirectory() + "yume_opt_\(UUID().uuidString).safetensors"
    defer { try? FileManager.default.removeItem(atPath: path) }
    do {
        let opt = CheckpointOptimizerState(step: 8, moments: momentsMem)
        try saveState(vae.matrices, to: path,
                      epoch: 0, step: 8, optimizer: opt)
    } catch {
        expect(false, "чекпоинт с оптимизатором сохранён: \(error)")
        return
    }
    var state: CheckpointState?
    do {
        state = try loadStateWithMeta(path: path)
    } catch {
        expect(false, "чекпоинт с оптимизатором загружен: \(error)")
        return
    }
    guard let st = state, let opt = st.optimizer else {
        expect(false, "состояние оптимизатора восстановлено")
        return
    }
    expectEqual(opt.step, 8, "счётчик шагов Adam восстановлен")
    expectEqual(opt.moments.count, momentsMem.count, "число моментов совпадает")
    var worst = 0.0
    for i in 0..<min(opt.moments.count, momentsMem.count) {
        for k in 0..<min(opt.moments[i].data.count, momentsMem[i].data.count) {
            worst = max(worst, Swift.abs(Double(opt.moments[i].data[k] - momentsMem[i].data[k])))
        }
    }
    expect(worst == 0, String(format: "моменты сохранены без потерь (%.3e)", worst))

    // Загрузка моментов в новую модель.
    let vae2 = ConvVAEGPU(engine: engine)
    do {
        try vae2.load(from: st.mats)
        try vae2.loadOptimizerMoments(opt.moments)
    } catch {
        expect(false, "моменты загружены в модель: \(error)")
        return
    }
    let momentsLoaded = vae2.optimizerMoments()
    var worst2 = 0.0
    for i in 0..<min(momentsLoaded.count, momentsMem.count) {
        for k in 0..<min(momentsLoaded[i].data.count, momentsMem[i].data.count) {
            worst2 = max(worst2, Swift.abs(Double(momentsLoaded[i].data[k] - momentsMem[i].data[k])))
        }
    }
    expect(worst2 == 0, String(format: "моменты загружены в модель (%.3e)", worst2))

    // Слишком мало моментов — ошибка.
    let vae3 = ConvVAEGPU(engine: engine)
    do {
        try vae3.loadOptimizerMoments(Array(opt.moments.prefix(4)))
        expect(false, "неполное состояние отклонено")
    } catch {
        expect(true, "неполное состояние отклонено")
    }
}


func testCodeGuideLimits() {
    section("Архитектурные стандарты: ≤600 строк/файл и атомарные функции")

    let maxFileLines = 600
    let maxFuncBody = 120
    // Долг — легаси, который план рефакторинга разбивает (см. ROADMAP.md).
    let knownOversized: Set<String> = [
        "Sources/YumeLayersCore/MetalKernelSources.swift",
        "Sources/YumeLayersUI/main.swift",
    ]
    let knownLargeFuncs: Set<String> = [
        "Sources/YumeLayersUI/ApiServer.swift:68",
        "Sources/YumeLayersCore/ParallelExtractor.swift:194",
        "Sources/YumeLayersCore/SoftwareFrameDecoder.swift:26",
        "Sources/YumeLayersCore/ConvGPU.swift:308",
        "Sources/YumeLayersCore/MetalEngine.swift:26",
        "Sources/YumeLayersUI/ApiServer.swift:194",
        "Sources/YumeLayersCore/LayeredTrainer.swift:133",
        "Sources/YumeLayersCore/ParallelExtractor.swift:268",
        "Sources/YumeLayersCore/SelfTest.swift:9",
        "Sources/YumeLayersCore/ConvTrainer.swift:131",
        "Sources/YumeLayersUI/main.swift:242",
        "Sources/YumeLayersUI/DelegatePart1.swift:170",
        "Sources/YumeLayersUI/DelegatePart4.swift:201",
        "Tests/YumeLayersTests/TestGroup6.swift:13",
        "Tests/YumeLayersTests/TestGroup5.swift:184",
        "Tests/YumeLayersTests/TestGroup5.swift:334",
    ]

    // Корень репозитория.
    var root = FileManager.default.currentDirectoryPath
    if !FileManager.default.fileExists(atPath: root + "/Sources") {
        let thisFile = #filePath
        var p = (thisFile as NSString).deletingLastPathComponent
        p = (p as NSString).deletingLastPathComponent
        p = (p as NSString).deletingLastPathComponent
        root = p
    }
    guard FileManager.default.fileExists(atPath: root + "/Sources") else {
        expect(false, "не найден корень репозитория (root=\(root))")
        return
    }

    func relPath(_ full: String) -> String {
        (full as NSString).substring(from: root.count + 1)
    }
    func occurrences(_ s: String, _ ch: Character) -> Int {
        s.reduce(0) { $0 + ($1 == ch ? 1 : 0) }
    }

    var fileViolations: [String] = []
    var funcViolations: [String] = []

    for dir in [root + "/Sources", root + "/Tests"] {
        guard let en = FileManager.default.enumerator(atPath: dir) else { continue }
        while let f = en.nextObject() as? String {
            guard f.hasSuffix(".swift") else { continue }
            let full = (dir as NSString).appendingPathComponent(f)
            let rel = relPath(full)
            guard let content = try? String(contentsOfFile: full, encoding: .utf8) else { continue }
            let trimmed = content.hasSuffix("\n") ? String(content.dropLast()) : content
            let lines = trimmed.components(separatedBy: "\n")
            let n = lines.count
            let oversizeKnown = knownOversized.contains(rel)
            if n > maxFileLines && !oversizeKnown {
                fileViolations.append("\(rel)=\(n)")
            }
            if oversizeKnown { continue }

            var depth = 0
            var inFunc = false
            var pending = -1
            var fstart = 0
            var base = 0
            for (i, raw) in lines.enumerated() {
                let noComment = raw.split(separator: "//").first.map(String.init) ?? raw
                let d = occurrences(noComment, "{") - occurrences(noComment, "}")
                if pending > 0 {
                    if noComment.contains("{") {
                        base = depth
                        fstart = pending
                        depth += d
                        pending = -1
                        inFunc = true
                    } else {
                        depth += d
                    }
                    continue
                }
                if inFunc {
                    depth += d
                    if depth <= base {
                        let body = i + 1 - fstart
                        if body > maxFuncBody && !knownLargeFuncs.contains("\(rel):\(fstart)") {
                            funcViolations.append("\(rel):\(fstart)=\(body)")
                        }
                        inFunc = false
                    }
                    continue
                }
                if raw.contains("func ") {
                    if noComment.contains("{") {
                        base = depth
                        fstart = i + 1
                        depth += d
                        inFunc = true
                    } else {
                        pending = i + 1
                        depth += d
                    }
                } else {
                    depth += d
                }
            }
        }
    }

    if fileViolations.isEmpty {
        expect(true, "нет файлов сверх \(maxFileLines) строк")
    } else {
        expect(false, "файлы сверх \(maxFileLines) строк: " + fileViolations.joined(separator: "; "))
    }
    if funcViolations.isEmpty {
        expect(true, "нет функций длиннее \(maxFuncBody) строк")
    } else {
        expect(false, "крупные функции > \(maxFuncBody): " + funcViolations.joined(separator: "; "))
    }
}


