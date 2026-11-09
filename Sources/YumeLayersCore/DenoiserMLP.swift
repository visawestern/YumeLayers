// YumeLayers — Фаза 3: денойзер-сеть (MLP) для диффузии в латентном
// пространстве. Полностью на существующих GPU-операциях GpuSession.
import Foundation

/// Синусоидальное временное эмбеддинг (классическое из Transformer):
/// emb[2i] = sin(t / 10000^(2i/d)), emb[2i+1] = cos(t / 10000^(2i/d)).
public func timeEmbedding(t: Int, dim: Int) -> [Float] {
    precondition(dim > 0 && dim % 2 == 0, "dim чётный и > 0")
    var out = [Float](repeating: 0, count: dim)
    let ft = Double(t)
    for i in stride(from: 0, to: dim, by: 2) {
        let w = pow(10000.0, Double(i) / Double(dim))
        out[i] = Float(sin(ft / w))
        out[i + 1] = Float(cos(ft / w))
    }
    return out
}

/// Полносвязный слой поверх GpuSession: Y = X·Wᵀ + b.
final class LinearGPU {
    let ic: Int
    let oc: Int
    let W: GPUTensor
    let b: GPUTensor
    let mW: GPUTensor
    let vW: GPUTensor
    let mb: GPUTensor
    let vb: GPUTensor

    init(engine: MetalEngine, ic: Int, oc: Int, rng: inout SplitMix64) {
        self.ic = ic
        self.oc = oc
        let scale = 1.0 / sqrt(Float(ic))
        var wData = [Float](repeating: 0, count: ic * oc)
        for i in 0..<wData.count {
            wData[i] = Float.random(in: -scale...scale, using: &rng)
        }
        W = GPUTensor(engine: engine, data: wData, rows: oc)
        let zeroB = [Float](repeating: 0, count: oc)
        b = GPUTensor(engine: engine, data: zeroB, rows: oc)
        let zeroW = [Float](repeating: 0, count: ic * oc)
        mW = GPUTensor(engine: engine, data: zeroW, rows: oc)
        vW = GPUTensor(engine: engine, data: zeroW, rows: oc)
        mb = GPUTensor(engine: engine, data: zeroB, rows: oc)
        vb = GPUTensor(engine: engine, data: zeroB, rows: oc)
    }

    /// Y = X·Wᵀ + b.
    func forward(_ s: GpuSession, _ X: GPUTensor) -> GPUTensor {
        s.biasRow(s.matmul(X, W, transB: true), b)
    }

    /// dY (n×oc) → (dX, gW, gb).
    func backward(_ s: GpuSession, _ X: GPUTensor, _ dY: GPUTensor)
        -> (GPUTensor, GPUTensor, GPUTensor) {
        let dX = s.matmul(dY, W)
        // gW (oc×ic) = dYᵀ·X: dY (1×oc)ᵀ → oc×1; X 1×ic → oc×ic.
        let gW = s.matmul(dY, X, transA: true)
        let gb = s.sumCols(dY)
        return (dX, gW, gb)
    }

    func applyAdam(_ s: GpuSession, _ gW: GPUTensor, _ gb: GPUTensor,
                   t: Int, lr: Float) {
        s.adamUpdate(W, gW, mW, vW, t: t, lr: lr)
        s.adamUpdate(b, gb, mb, vb, t: t, lr: lr)
    }
}

/// Входной слой денойзера: z = xT·Wxᵀ + emb·Weᵀ + b.
/// Временное эмбеддинг не обучается (фиксированное, как в DDPM).
final class InputLinearGPU {
    let dimX: Int
    let dimE: Int
    let oc: Int
    let Wx: GPUTensor
    let We: GPUTensor
    let b: GPUTensor
    let mWx: GPUTensor
    let vWx: GPUTensor
    let mb: GPUTensor
    let vb: GPUTensor

    init(engine: MetalEngine, dimX: Int, dimE: Int, oc: Int,
         rng: inout SplitMix64) {
        self.dimX = dimX
        self.dimE = dimE
        self.oc = oc
        let scale = 1.0 / sqrt(Float(dimX + dimE))
        func randMatrix(_ rows: Int, _ cols: Int) -> [Float] {
            (0..<(rows * cols)).map { _ in Float.random(in: -scale...scale, using: &rng) }
        }
        let wx = randMatrix(oc, dimX)
        let we = randMatrix(oc, dimE)
        Wx = GPUTensor(engine: engine, data: wx, rows: oc)
        We = GPUTensor(engine: engine, data: we, rows: oc)
        let zeroB = [Float](repeating: 0, count: oc)
        b = GPUTensor(engine: engine, data: zeroB, rows: oc)
        mWx = GPUTensor(engine: engine, data: [Float](repeating: 0, count: oc * dimX), rows: oc)
        vWx = GPUTensor(engine: engine, data: [Float](repeating: 0, count: oc * dimX), rows: oc)
        mb = GPUTensor(engine: engine, data: zeroB, rows: oc)
        vb = GPUTensor(engine: engine, data: zeroB, rows: oc)
    }

    /// z = xT·Wxᵀ + emb·Weᵀ + b (xT: 1×dimX, embRow: 1×dimE).
    func forward(_ s: GpuSession, _ xT: GPUTensor, _ embRow: GPUTensor) -> GPUTensor {
        let z = s.add(s.matmul(xT, Wx, transB: true),
                      s.matmul(embRow, We, transB: true))
        return s.biasRow(z, b)
    }

    /// gWx (oc×dimX) = dZᵀ·xT; gb = Σ_строк dZ.
    func backwardX(_ s: GpuSession, _ xT: GPUTensor, _ dZ: GPUTensor)
        -> (GPUTensor, GPUTensor) {
        let gWx = s.matmul(dZ, xT, transA: true)
        let gb = s.sumCols(dZ)
        return (gWx, gb)
    }

    func applyAdam(_ s: GpuSession, _ gWx: GPUTensor, _ gb: GPUTensor,
                   t: Int, lr: Float) {
        s.adamUpdate(Wx, gWx, mWx, vWx, t: t, lr: lr)
        s.adamUpdate(b, gb, mb, vb, t: t, lr: lr)
    }
}

/// MLP-денойзер: f(x_t, t) → предсказание шума ε.
/// Слои: входной (dimX+dimE) → hidden → hidden → dimX; tanh между ними.
final class DenoiserMLPGPU {
    let dimIn: Int
    let dimEmb: Int
    let hidden: Int
    let l1: InputLinearGPU
    let l2: LinearGPU
    let l3: LinearGPU

    init(engine: MetalEngine, dimIn: Int, dimEmb: Int = 8, hidden: Int = 64,
         seed: UInt64 = 0xDEAD_BEEF_0000_0001) {
        self.dimIn = dimIn
        self.dimEmb = dimEmb
        self.hidden = hidden
        var rng = SplitMix64(seed: seed)
        l1 = InputLinearGPU(engine: engine, dimX: dimIn, dimE: dimEmb,
                            oc: hidden, rng: &rng)
        l2 = LinearGPU(engine: engine, ic: hidden, oc: hidden, rng: &rng)
        l3 = LinearGPU(engine: engine, ic: hidden, oc: dimIn, rng: &rng)
    }

    struct Cache {
        let xT: GPUTensor
        let z1: GPUTensor, a1: GPUTensor
        let z2: GPUTensor, a2: GPUTensor
        let z3: GPUTensor
        let pred: GPUTensor
    }

    /// Forward с кэшем для backward. Вход: xT (1×dimIn), emb (dimEmb).
    /// Никаких чтений GPU→CPU внутри сессии.
    func forwardCached(_ s: GpuSession, xT: GPUTensor, emb: [Float])
        -> (GPUTensor, Cache) {
        precondition(emb.count == dimEmb, "размер эмбеддинга")
        let embRow = GPUTensor(engine: s.engine, data: emb, rows: 1)
        let z1 = l1.forward(s, xT, embRow)
        let a1 = s.tanh(z1)
        let z2 = l2.forward(s, a1)
        let a2 = s.tanh(z2)
        let z3 = l3.forward(s, a2)
        let cache = Cache(xT: xT, z1: z1, a1: a1,
                          z2: z2, a2: a2, z3: z3, pred: z3)
        return (z3, cache)
    }

    /// Convenience: только pred.
    func forward(_ s: GpuSession, xT: GPUTensor, emb: [Float]) -> GPUTensor {
        forwardCached(s, xT: xT, emb: emb).0
    }

    /// Backward по MSE(pred, eps). Возвращает [gW1, gb1, gW2, gb2, gW3, gb3].
    func backward(_ s: GpuSession, _ cache: Cache, target: GPUTensor) -> [GPUTensor] {
        let n = Float(dimIn)
        let dZ3 = s.smul(2.0 / n, s.sub(cache.pred, target))
        let (dA2, gW3, gb3) = l3.backward(s, cache.a2, dZ3)
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

    /// Матрицы: [Wx, b1, W2, b2, W3, b3] (для чекпоинтов/численных проверок).
    var matrices: [Matrix] {
        [l1.Wx.toMatrix(), l1.b.toMatrix(),
         l2.W.toMatrix(), l2.b.toMatrix(),
         l3.W.toMatrix(), l3.b.toMatrix()]
    }

    /// Загрузка матриц (того же формата, что `matrices`).
    func load(from mats: [Matrix]) throws {
        guard mats.count == 6 else { throw NSError(domain: "Denoiser", code: 1) }
        func into(_ t: GPUTensor, _ m: Matrix) {
            t.copyFrom(GPUTensor(engine: t.engine, data: m.data, rows: m.rows))
        }
        into(l1.Wx, mats[0]); into(l1.b, mats[1])
        into(l2.W, mats[2]); into(l2.b, mats[3])
        into(l3.W, mats[4]); into(l3.b, mats[5])
    }
}
