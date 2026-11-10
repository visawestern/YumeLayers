// YumeLayers — ad-hoc оценка НАТРЕНИРОВАННОГО чекпоинта на реальном hold-out.
//
// Это НЕ юнит-тест в сюите (зависит от летучих артефактов тренировки), а сквозной
// инструмент: загружает conv.safetensors, восстанавливает архитектуру из МЕТАДАННЫХ
// чекпоинта (как исправленный interpolateVideo), реконструирует кадры из raw-набора
// hold-out и выносит вердикт `g2Verdict` против published SD-VAE (23.4 дБ / 0.69).
//
// Сборка (из корня репо):
//   xcrun swiftc -O -sdk "$(xcrun --show-sdk-path)" \
//     Sources/YumeLayersCore/*.swift tools/eval_trained.swift \
//     -framework Metal -framework Accelerate -framework AVFoundation \
//     -framework CoreGraphics -framework ImageIO -framework VideoToolbox \
//     -framework UniformTypeIdentifiers -framework IOKit -o /tmp/yume_eval
// Запуск: /tmp/yume_eval [path/to/conv.safetensors] [holdout.raw]
import Foundation

@main
enum EvalTool {
    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())
        let ckptPath = args.first ?? (FileManager.default.homeDirectoryForCurrentUser.path
            + "/YumeLayers/checkpoints/conv.safetensors")
        let holdoutPath = args.count > 1 ? args[1] : "/tmp/yume_holdout/holdout.raw"
        let w = 128, h = 72
        let frameBytes = w * h * 3
        let fm = FileManager.default

        guard fm.fileExists(atPath: ckptPath) else { print("EVAL FAIL: нет чекпоинта \(ckptPath)"); exit(2) }
        guard let raw = fm.contents(atPath: holdoutPath), raw.count >= frameBytes else {
            print("EVAL FAIL: нет/пуст hold-out \(holdoutPath)"); exit(2)
        }
        let frameCount = min(raw.count / frameBytes, 64)

        // Архитектура — из метаданных чекпоинта (loadStateWithMeta отдаёт modelWidth/latent/archVersion).
        let st: CheckpointState
        do { st = try loadStateWithMeta(path: ckptPath) } catch { print("EVAL FAIL: чекпоинт не читается: \(error)"); exit(2) }

        let engine = MetalEngine.shared
        let session = GpuSession(engine)
        let vae = ConvVAEGPU(engine: engine,
                             latentChannels: st.latentChannels,
                             width: st.modelWidth,
                             residual: st.archVersion >= 2,
                             groupNorm: st.archVersion >= 3)
        try! vae.load(from: st.mats)
        print(String(format: "      чекпоинт: modelWidth=%d latent=%d archVersion=%d · hold-out=%d кадров 128×72",
                     st.modelWidth, st.latentChannels, st.archVersion, frameCount))

        var sumP: Float = 0, sumS: Float = 0
        raw.withUnsafeBytes { p in
            let b = p.bindMemory(to: UInt8.self)
            for i in 0..<frameCount {
                var img = [Float](repeating: 0, count: frameBytes)
                let base = i * frameBytes
                for j in 0..<frameBytes { img[j] = Float(b[base + j]) / 255.0 }
                let x = GPUTensor(engine: engine, data: hwcToChw(img), rows: 3)
                session.begin()
                let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
                _ = session.end()
                let recon = chwToHwc(c.recon.toArray())
                sumP += psnr(reference: img, candidate: recon)
                sumS += luminanceSSIM(reference: img, candidate: recon)
            }
        }
        let n = Float(frameCount)
        let psnrV = sumP / n, ssimV = sumS / n
        let verdict = g2Verdict(ourPSNR: psnrV, ourSSIM: ssimV)
        print(String(format: "      trained-model hold-out: PSNR %.2f дБ · SSIM %.3f", psnrV, ssimV))
        print("      g2Verdict vs published SD-VAE (23.4/0.69): \(verdict)")
        if verdict == .pass { print("EVAL: G2 PASS") } else { print("EVAL: G2 \(verdict)") }
    }
}
