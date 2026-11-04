// YumeLayers — CLI: тонкий диспатчер поверх Core (параметры задаются в GUI, не флагами).
import Foundation
import YumeLayersCore

setbuf(stdout, nil)   // немедленный вывод логов

let args = CommandLine.arguments
var p = Params()

let mode = args.count > 1 ? args[1] : "help"
switch mode {
case "vae": try trainAutoencoderGPU(p: p)
case "conv": try trainConvVAE(p: p)
case "diffusion": try trainDiffusionGPU(p: p)
case "generate": try generateGPU(p: p)
case "check": runCheck()
case "convcheck": checkConvGradients()
case "selftest": runSelfTest()
case "convprofile": profileConv(w: ((p.width + 7) / 8) * 8, h: ((p.height + 7) / 8) * 8)
case "gpubench": runGpuBench()
case "gpuheat":
    // Прогрев настоящим циклом обучения: gpuheat [секунды] [скважность %]
    let secs = args.count > 2 ? (Double(args[2]) ?? 60) : 60
    let duty = args.count > 3 ? (Double(args[3]) ?? 85) / 100 : 0.85
    runGpuHeat(seconds: secs, duty: duty)
case "layers":
    // Рендер слоёв с ground-truth разметкой: layers [ш] [в] [seed]
    let lw = args.count > 2 ? (Int(args[2]) ?? p.width) : p.width
    let lh = args.count > 3 ? (Int(args[3]) ?? p.height) : p.height
    let seed = args.count > 4 ? (UInt64(args[4]) ?? 2026) : 2026
    let scene = makeLayeredScene(w: lw, h: lh, seed: seed)
    try saveLayerPreview(scene: scene, prefix: p.outPrefix + "_layers")
    let objPixels = scene.mask.filter { $0 == 1 }.count
    // Раньше здесь был `specifier:` внутри интерполяции — это API SwiftUI,
    // а не String. Выражение не типизировалось, и CLI не собирался вовсе.
    let objPercent = Double(100.0 * Float(objPixels) / Float(lw * lh))
    print(String(format: "слои: %d×%d · объекты %d пикс. (%.1f%%) → %@_layers_*.png",
                 lw, lh, objPixels, objPercent, p.outPrefix))
case "layered":
    // Обучение 8-канальной факторизации: layered [шаги] [мощность %]
    // Потолок GPU жёстко ограничен 90%, чтобы не повторить перегрев.
    let steps = args.count > 2 ? (Int(args[2]) ?? 60) : 60
    let power = args.count > 3 ? (Int(args[3]) ?? defaultGpuPowerPercent)
                               : defaultGpuPowerPercent
    print("layered VAE: 8 каналов (фон | объекты | маска | свет) · \(steps) шагов")
    let report = trainLayeredVAE(steps: steps, powerPercent: power)
    print(String(format: "loss %.5f → %.5f · PSNR %.2f дБ · SSIM %.4f · пол качества: %@",
                 report.firstLoss, report.lastLoss, report.psnr, report.ssim,
                 report.acceptedByFloor ? "принят" : "не пройден"))
    print(String(format: "скважность GPU %.0f%% · сборка: %d канала (RGB)",
                 report.duty * 100, report.compositeChannels))
case "gpusensors":
    // Живая телеметрия дискретной GPU: температура, частота, загрузка, мощность.
    // Проверка, что IORegistry отдаёт данные и что контур по температуре
    // вообще есть на чём строить.
    let sensor = GpuSensor()
    let samples = args.count > 2 ? (Int(args[2]) ?? 10) : 10
    print("GPU: \(MetalEngine.shared.deviceName)")
    for _ in 0..<samples {
        if let s = sensor.read() {
            print(String(format: "  %.0f °C · %.0f МГц · загрузка %.0f%% · %.0f Вт",
                         s.temperatureC, s.coreClockMHz, s.activityPercent, s.powerW))
        } else {
            print("  нет телеметрии (IOAccelerator не отдаёт PerformanceStatistics)")
        }
        Thread.sleep(forTimeInterval: 1.0)
    }
default:
    print("""
    YumeLayers (чистый Swift, без PyTorch)
    GUI:  swift run -c release YumeLayersUI   (параметры задаются в окне)
    CLI:  swift run -c release YumeLayers conv   (480p, щадящий режим)
          swift run -c release YumeLayers layered [шаги] [мощность %]  (8 каналов, ≤90% GPU)
          swift run -c release YumeLayers layers [ш] [в] [seed]        (превью слоёв)
          swift run -c release YumeLayers check / convcheck / gpubench
    """)
}
