// YumeLayers — Фаза 4: сеть движения между опорными кадрами.
//
// Предсказывает промежуточный латент по двум опорным: f(zA, zB, t).
// Выход сети — ПОПРАВКА к линейному смешиванию lerp(zA, zB, t):
//   z(t) = lerp(zA, zB, t) + net(zA, zB, t)
// Сеть стартует «с нуля» (все матрицы малы, скрытые слои дают ~0 на
// старте только после обучения; на первых шагах выход близок к случайному,
// поэтому в начале обучения доминирует lerp, а движение «доучивается»).
// Это даёт устойчивую отправную точку и гарантирует, что при t=0 и t=1
// итоговый кадр остаётся рядом с опорами (см. пару тестов MotionMLP).
import Foundation

/// MLP движения: вход xT = [zA | zB] (2·latentDim) + эмбеддинг времени.
public final class MotionMLPGPU {
    /// Размерность одного латента (каналы × высота × ширина латента).
    public let latentDim: Int
    /// Размерность эмбеддинга времени.
    public let dimEmb: Int
    /// Число нейронов скрытых слоёв.
    public let hidden: Int

    let l1: InputLinearGPU
    let l2: LinearGPU
    let l3: LinearGPU

    /// Нормализация латентов: z' = (z − mean) / scale. Задаётся при обучении
    /// и сохраняется в чекпоинт; инференс использует те же значения.
    public var latentMean: Float = 0
    public var latentScale: Float = 1

    public init(engine: MetalEngine, latentDim: Int, dimEmb: Int = 8,
                hidden: Int = 96,
                seed: UInt64 = 0x900D_0000_0000_0001) {
        precondition(latentDim > 0 && dimEmb > 0 && hidden > 0)
        self.latentDim = latentDim
        self.dimEmb = dimEmb
        self.hidden = hidden
        var rng = SplitMix64(seed: seed)
        l1 = InputLinearGPU(engine: engine, dimX: 2 * latentDim, dimE: dimEmb,
                            oc: hidden, rng: &rng)
        l2 = LinearGPU(engine: engine, ic: hidden, oc: hidden, rng: &rng)
        // Выходной слой — нулевая инициализация: на старте сеть даёт ~0,
        // z(t) = lerp(zA,zB,t) + 0 = lerp. Сеть «доучивает» только то, что
        // реально улучшает предсказание, не ломая рабочий lerp.
        l3 = LinearGPU(engine: engine, ic: hidden, oc: latentDim, rng: &rng,
                       zeroInit: true)
    }

    struct Cache {
        let xT: GPUTensor
        let z1: GPUTensor, a1: GPUTensor
        let z2: GPUTensor, a2: GPUTensor
        let z3: GPUTensor
        let pred: GPUTensor
    }

    /// Forward с кэшем для backward. Вход: xT (1×2·latentDim), emb (dimEmb),
    /// base (1×latentDim) — lerp-опора, к которой добавляется поправка сети.
    func forwardCached(_ s: GpuSession, xT: GPUTensor, emb: [Float],
                       base: GPUTensor) -> (GPUTensor, Cache) {
        precondition(emb.count == dimEmb, "размер эмбеддинга")
        let embRow = GPUTensor(engine: s.engine, data: emb, rows: 1)
        let z1 = l1.forward(s, xT, embRow)
        let a1 = s.tanh(z1)
        let z2 = l2.forward(s, a1)
        let a2 = s.tanh(z2)
        let z3 = l3.forward(s, a2)
        let pred = s.add(base, z3)
        let cache = Cache(xT: xT, z1: z1, a1: a1,
                          z2: z2, a2: a2, z3: z3, pred: pred)
        return (pred, cache)
    }

    /// Convenience: только pred.
    func forward(_ s: GpuSession, xT: GPUTensor, emb: [Float],
                 base: GPUTensor) -> GPUTensor {
        forwardCached(s, xT: xT, emb: emb, base: base).0
    }

    /// Backward по MSE(pred, target). Возвращает [gW1, gb1, gW2, gb2, gW3, gb3].
    /// Так как pred = base + net(x), градиент к выходу сети равен dL/dpred.
    func backward(_ s: GpuSession, _ cache: Cache, target: GPUTensor) -> [GPUTensor] {
        let n = Float(latentDim)
        let dZ3 = s.smul(2.0 / n, s.sub(cache.pred, target))
        let (dA2, gW3, gb3) = l3.backward(s, cache.a2, dZ3)
        let dZ2 = s.mul(s.tanhGrad(cache.z2), dA2)
        let (dA1, gW2, gb2) = l2.backward(s, cache.a1, dZ2)
        let dZ1 = s.mul(s.tanhGrad(cache.z1), dA1)
        let (gW1, gb1) = l1.backwardX(s, cache.xT, dZ1)
        return [gW1, gb1, gW2, gb2, gW3, gb3]
    }

    /// Backward по ВНЕШНЕМУ градиенту выходного латента (например, от
    /// пиксельной потери через декодер VAE). Возвращает 6 градиентов весов.
    func backwardFromGrad(_ s: GpuSession, _ cache: Cache,
                          dOut: GPUTensor) -> [GPUTensor] {
        let (dA2, gW3, gb3) = l3.backward(s, cache.a2, dOut)
        let dZ2 = s.mul(s.tanhGrad(cache.z2), dA2)
        let (dA1, gW2, gb2) = l2.backward(s, cache.a1, dZ2)
        let dZ1 = s.mul(s.tanhGrad(cache.z1), dA1)
        let (gW1, gb1) = l1.backwardX(s, cache.xT, dZ1)
        return [gW1, gb1, gW2, gb2, gW3, gb3]
    }

    func applyAdam(_ s: GpuSession, _ grads: [GPUTensor], t: Int, lr: Float) {
        precondition(grads.count >= 6, "нужно 6 градиентов")
        l1.applyAdam(s, grads[0], grads[1], t: t, lr: lr)
        l2.applyAdam(s, grads[2], grads[3], t: t, lr: lr)
        l3.applyAdam(s, grads[4], grads[5], t: t, lr: lr)
    }

    /// Матрицы: [Wx, b1, W2, b2, W3, b3].
    var matrices: [Matrix] {
        [l1.Wx.toMatrix(), l1.b.toMatrix(),
         l2.W.toMatrix(), l2.b.toMatrix(),
         l3.W.toMatrix(), l3.b.toMatrix()]
    }

    /// Загрузка матриц (того же формата, что `matrices`).
    func load(from mats: [Matrix]) throws {
        guard mats.count == 6 else { throw NSError(domain: "MotionMLP", code: 1) }
        func into(_ t: GPUTensor, _ m: Matrix) {
            t.copyFrom(GPUTensor(engine: t.engine, data: m.data, rows: m.rows))
        }
        into(l1.Wx, mats[0]); into(l1.b, mats[1])
        into(l2.W, mats[2]); into(l2.b, mats[3])
        into(l3.W, mats[4]); into(l3.b, mats[5])
    }
}
