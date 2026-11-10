// YumeLayers — градиент пиксельной потери к латенту декодера.
// Вынесено из ConvGPU.swift для соблюдения лимита 600 строк на файл.
import Foundation

extension ConvVAEGPU {
    /// Градиент пиксельной потери к ВХОДНОМУ латенту декодера.
    ///
    /// Для сети движения: предсказанный латент декодируется, MSE с целевым
    /// кадром считается в пикселях, а этот метод возвращает dL/dz (без
    /// энкодера и без обновления весов VAE).
    func gradToLatent(_ s: GpuSession, _ cache: DecoderOutput,
                      target: GPUTensor, w: Int, h: Int,
                      dReconOverride: GPUTensor? = nil) -> GPUTensor {
        let n = Float(channels * w * h)
        let dRecon = dReconOverride ?? s.smul(2.0 / n, s.sub(cache.recon, target))
        let dP6 = s.sigmoidGrad(cache.p6, dRecon)
        let (dx6, _, _) = d3.backward(s, cache.c6, dOut: dP6)
        var dA5 = s.downsample2x(dx6, h: h / 2, w: w / 2)
        var dP5 = s.reluGrad(cache.a5, dA5)
        if groupNorm { dP5 = s.encodeGroupNormGrad(cache.p5, dP5, groups: 2) }
        let (dx5, _, _) = d2.backward(s, cache.c5, dOut: dP5)
        var dA4 = s.downsample2x(dx5, h: h / 4, w: w / 4)
        if let rb = resD, let rc = cache.resD {
            let (dPrev, _) = rb.backward(s, rc, dOut: dA4)
            dA4 = dPrev
        }
        var dP4 = s.reluGrad(cache.a4Raw, dA4)
        if groupNorm { dP4 = s.encodeGroupNormGrad(cache.p4, dP4, groups: 2) }
        let (dx4, _, _) = d1.backward(s, cache.c4, dOut: dP4)
        return s.downsample2x(dx4, h: h / 8, w: w / 8)
    }
}
