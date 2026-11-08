// YumeLayers — остаточный блок (ResidualBlockGPU), Этап 1.
//
// Модуль f(x) = x + conv2(ReLU(conv1(x))) на плоскости (CHW: rows=каналы,
// cols=плоскость). Реализует собственный forward/backward на существующих
// GPU-операциях (ConvLayerGPU, relu, add). Численно проверяется тестом
// (конечные разности). Готов к интеграции в `ConvVAEGPU` (архитектура v2).
import Foundation

final class ResidualBlockGPU {
    let c1: ConvLayerGPU
    let c2: ConvLayerGPU

    init(engine: MetalEngine, channels: Int, scale: Float? = nil,
         rng: inout SplitMix64) {
        c1 = ConvLayerGPU(engine: engine, ic: channels, oc: channels,
                          k: 3, stride: 1, pad: 1, scale: scale, rng: &rng)
        c2 = ConvLayerGPU(engine: engine, ic: channels, oc: channels,
                          k: 3, stride: 1, pad: 1, scale: scale, rng: &rng)
    }

    var matrices: [Matrix] { c1.matrices + c2.matrices }

    struct Cache {
        let x: GPUTensor
        let a1: GPUTensor
        let c1: ConvCache
        let c2: ConvCache
    }

    func forward(_ s: GpuSession, _ x: GPUTensor,
                 ih: Int, iw: Int) -> (GPUTensor, Cache) {
        let (p1, cc1) = c1.forward(s, x, ih: ih, iw: iw)
        let a1 = s.relu(p1)
        let (p2, cc2) = c2.forward(s, a1, ih: ih, iw: iw)
        let out = s.add(x, p2)
        return (out, Cache(x: x, a1: a1, c1: cc1, c2: cc2))
    }

    func backward(_ s: GpuSession, _ cache: Cache, dOut: GPUTensor)
        -> (dx: GPUTensor, grads: [GPUTensor]) {
        // out = x + p2 → dL/dp2 = dOut, dL/dx = dOut + dL/d(conv1-path).
        let (dA1, gW2, gb2) = c2.backward(s, cache.c2, dOut: dOut)
        let dP1 = s.reluGrad(cache.a1, dA1)
        let (dxC1, gW1, gb1) = c1.backward(s, cache.c1, dOut: dP1)
        let dx = s.add(dOut, dxC1)
        return (dx, [gW1, gb1, gW2, gb2])
    }

    func applyAdam(_ s: GpuSession, _ grads: [GPUTensor], t: Int, lr: Float) {
        c1.applyAdam(s, grads[0], grads[1], t: t, lr: lr)
        c2.applyAdam(s, grads[2], grads[3], t: t, lr: lr)
    }
}
