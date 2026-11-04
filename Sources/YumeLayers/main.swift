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
    print("слои: \(lw)×\(lh) · объекты \(objPixels) пикс. (\(100.0 * Float(objPixels) / Float(lw * lh), specifier: "%.1f")%) → \(p.outPrefix)_layers_*.png")
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
          swift run -c release YumeLayers check / convcheck / gpubench
    """)
}
