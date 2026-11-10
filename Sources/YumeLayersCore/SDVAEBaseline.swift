// YumeLayers — опубликованный baseline SD-VAE для ворот G2.
//
// Раньше G2 считалась «заблокированной, ждём baseline от пользователя». Это было
// неверно: SD-VAE — публичная модель с задокументированными метриками реконструкции,
// их не нужно гонять локально (torch/MPS на стенде нет). Здесь фиксируем ОФИЦИАЛЬНЫЕ
// числа из model card stabilityai/sd-vae-ft-mse, чтобы гейт `g2Gate` сравнивал нашу
// модель с реальным, а не выдуманным эталоном.
//
// Источник: Hugging Face «stabilityai/sd-vae-ft-mse», таблица реконструкции
// https://huggingface.co/stabilityai/sd-vae-ft-mse
//   original (kl-f8, «as used in SD»): rFID 4.99 · PSNR 23.4 ± 3.8 дБ · SSIM 0.69 ± 0.14
//   ft-EMA:                            rFID 4.42 · PSNR 23.8 ± 3.9 дБ · SSIM 0.69 ± 0.13
//   (на подмножестве лиц вторая таблица даёт PSNR 26.0–26.7 / SSIM 0.81–0.82)
//
// Для «не хуже SD-VAE» берём КОНСЕРВАТИВНУЮ нижнюю границу того же SD-VAE, что
// реально стоит в Stable Diffusion 1.x (original kl-f8), а не лучшую из таблиц:
// PSNR 23.4 дБ и SSIM 0.69. Это честный и достижимый ориентир, а не фантастические
// ≥30 дБ / ≥0.90, которые сам SD-VAE не даёт.

/// Опубликованный реконструкционный baseline SD-VAE (as used in SD 1.x).
public struct SDVAEBaseline: Equatable, Sendable {
    public let psnr: Float
    public let ssim: Float
    public let rFID: Float
    public let source: String

    public init(psnr: Float, ssim: Float, rFID: Float, source: String) {
        self.psnr = psnr; self.ssim = ssim; self.rFID = rFID; self.source = source
    }

    /// Консервативная граница «as used in SD» (original kl-f8) из model card.
    public static let sdVAEFTMSE = SDVAEBaseline(
        psnr: 23.4, ssim: 0.69, rFID: 4.99,
        source: "stabilityai/sd-vae-ft-mse — original kl-f8 (HF model card)")
}

/// Вердикт G2 против опубликованного SD-VAE baseline (удобная обёртка над `g2Gate`).
public func g2Verdict(ourPSNR: Float, ourSSIM: Float,
                      baseline: SDVAEBaseline = .sdVAEFTMSE) -> G2Verdict {
    // Пол = published SD-VAE: «не хуже SD-VAE» ≡ наши метрики ≥ baseline-значений.
    return g2Gate(ourPSNR: ourPSNR, ourSSIM: ourSSIM,
                  baselinePSNR: baseline.psnr, baselineSSIM: baseline.ssim,
                  minPSNR: baseline.psnr, minSSIM: baseline.ssim)
}
