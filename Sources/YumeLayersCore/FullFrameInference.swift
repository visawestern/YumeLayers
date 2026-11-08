// YumeLayers — инференс полного кадра (Этап 2: 480p+).
//
// Сеть полностью свёрточная, но требует размеров, кратных 8. Мода
// 854×480 (или любой другой) не кратна 8, поэтому кадр паддится до кратного,
// прогоняется через VAE и обрезается обратно до оригинальной области.
import Foundation

/// Реконструкция полного кадра с паддингом до кратности 8.
///
/// Вход/выход — RGB HWC 0…1. `nil`, если вход некорректен.
public func reconstructPadded(_ engine: MetalEngine, _ vae: ConvVAEGPU,
                              _ session: GpuSession,
                              rgb: [Float], w: Int, h: Int,
                              deterministic: Bool = true) -> [Float]? {
    guard w > 0, h > 0, rgb.count >= w * h * 3 else { return nil }
    let pw = ((w + 7) / 8) * 8
    let ph = ((h + 7) / 8) * 8

    // Паддинг: повторение крайнего пикселя (без введения новых значений).
    var padded = [Float](repeating: 0, count: pw * ph * 3)
    for y in 0..<ph {
        let sy = min(y, h - 1)
        for x in 0..<pw {
            let sx = min(x, w - 1)
            let si = (sy * w + sx) * 3
            let di = (y * pw + x) * 3
            padded[di] = rgb[si]; padded[di + 1] = rgb[si + 1]; padded[di + 2] = rgb[si + 2]
        }
    }

    let chw = hwcToChw(padded)
    let x = GPUTensor(engine: engine, data: chw, rows: 3)
    session.begin()
    let c = vae.forward(session, image: x, w: pw, h: ph, deterministic: deterministic)
    _ = session.end()
    let reconHWC = chwToHwc(c.recon.toArray())

    // Обрезка до оригинальной области.
    var out = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            let si = (y * pw + x) * 3
            let di = (y * w + x) * 3
            out[di] = reconHWC[si]; out[di + 1] = reconHWC[si + 1]; out[di + 2] = reconHWC[si + 2]
        }
    }
    return out
}
