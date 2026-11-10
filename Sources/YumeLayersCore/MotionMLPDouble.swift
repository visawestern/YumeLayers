// YumeLayers — сеть движения в двойной точности (CPU, Float64).
//
// Зачем: на полных латентах 854×480 (205k+ измерений) Float32-GPU-версия
// переполняется (NaN во всех градиентах). Эта CPU-версия считает в Double,
// что численно стабильно при любых размерах входа; VAE encode/decode при
// этом остаётся на GPU (там Float32 — отраслевой стандарт).
//
// Архитектура повторяет GPU-версию: z(t) = lerp(zA,zB,t) + net([zA|zB], emb(t)),
// выход сети инициализируется нулями (старт = lerp).
import Foundation

public final class MotionMLPDouble {
    /// Размерность вектора входа/выхода сети (1 вектор латента).
    public let latentDim: Int
    /// Размерность эмбеддинга времени.
    public let dimEmb: Int
    /// Число нейронов скрытых слоёв.
    public let hidden: Int

    /// Нормализация латентов (mean/scale), как в GPU-версии.
    public var latentMean: Double = 0
    public var latentScale: Double = 1

    // ---- Веса (строки × столбцы) ----
    private var W1: [Double]   // hidden × (2*latentDim + dimEmb)
    private var b1: [Double]   // hidden
    private var W2: [Double]   // hidden × hidden
    private var b2: [Double]   // hidden
    private var W3: [Double]   // latentDim × hidden
    private var b3: [Double]   // latentDim

    // ---- Adam-состояния ----
    private var mW1: [Double], vW1: [Double]
    private var mb1: [Double], vb1: [Double]
    private var mW2: [Double], vW2: [Double]
    private var mb2: [Double], vb2: [Double]
    private var mW3: [Double], vW3: [Double]
    private var mb3: [Double], vb3: [Double]

    public init(latentDim: Int, dimEmb: Int = 8, hidden: Int = 128,
                seed: UInt64 = 0x900D_0000_0000_0002) {
        precondition(latentDim > 0 && dimEmb > 0 && hidden > 0)
        self.latentDim = latentDim
        self.dimEmb = dimEmb
        self.hidden = hidden
        let dimIn = 2 * latentDim + dimEmb
        var rng = SplitMix64(seed: seed)
        func rand(_ scale: Double) -> Double {
            Double.random(in: -scale...scale, using: &rng)
        }
        let s1 = 1.0 / sqrt(Double(dimIn))
        W1 = (0..<(hidden * dimIn)).map { _ in rand(s1) }
        b1 = [Double](repeating: 0, count: hidden)
        let s2 = 1.0 / sqrt(Double(hidden))
        W2 = (0..<(hidden * hidden)).map { _ in rand(s2) }
        b2 = [Double](repeating: 0, count: hidden)
        // Выходной слой — нули (residual-старт: pred = lerp).
        W3 = [Double](repeating: 0, count: latentDim * hidden)
        b3 = [Double](repeating: 0, count: latentDim)
        func z(_ n: Int) -> [Double] { [Double](repeating: 0, count: n) }
        mW1 = z(W1.count); vW1 = z(W1.count)
        mb1 = z(hidden); vb1 = z(hidden)
        mW2 = z(W2.count); vW2 = z(W2.count)
        mb2 = z(hidden); vb2 = z(hidden)
        mW3 = z(W3.count); vW3 = z(W3.count)
        mb3 = z(latentDim); vb3 = z(latentDim)
    }

    // MARK: - Forward

    public struct Cache {
        public let x: [Double]
        public let z1: [Double], a1: [Double]
        public let z2: [Double], a2: [Double]
        public let z3: [Double]
        public let pred: [Double]
    }

    /// forward: pred = base + net([zA|zB], emb).
    /// xT — вектор длины 2*latentDim, emb — длины dimEmb, base — latentDim.
    public func forward(xT: [Double], emb: [Double], base: [Double]) -> (pred: [Double], cache: Cache) {
        precondition(xT.count == 2 * latentDim && emb.count == dimEmb && base.count == latentDim)
        let dimIn = 2 * latentDim + dimEmb
        var x = [Double](repeating: 0, count: dimIn)
        for i in 0..<xT.count { x[i] = xT[i] }
        for i in 0..<emb.count { x[xT.count + i] = emb[i] }
        // z1 = W1·x + b1
        var z1 = [Double](repeating: 0, count: hidden)
        for i in 0..<hidden {
            var s = b1[i]
            let row = i * dimIn
            for j in 0..<dimIn { s += W1[row + j] * x[j] }
            z1[i] = s
        }
        var a1 = [Double](repeating: 0, count: hidden)
        for i in 0..<hidden { a1[i] = tanh(z1[i]) }
        // z2 = W2·a1 + b2
        var z2 = [Double](repeating: 0, count: hidden)
        for i in 0..<hidden {
            var s = b2[i]
            let row = i * hidden
            for j in 0..<hidden { s += W2[row + j] * a1[j] }
            z2[i] = s
        }
        var a2 = [Double](repeating: 0, count: hidden)
        for i in 0..<hidden { a2[i] = tanh(z2[i]) }
        // z3 = W3·a2 + b3
        var z3 = [Double](repeating: 0, count: latentDim)
        for k in 0..<latentDim {
            var s = b3[k]
            let row = k * hidden
            for j in 0..<hidden { s += W3[row + j] * a2[j] }
            z3[k] = s
        }
        var pred = [Double](repeating: 0, count: latentDim)
        for k in 0..<latentDim { pred[k] = base[k] + z3[k] }
        return (pred, Cache(x: x, z1: z1, a1: a1, z2: z2, a2: a2,
                            z3: z3, pred: pred))
    }

    // MARK: - Backward

    public struct Grads {
        public var gW1: [Double], gb1: [Double]
        public var gW2: [Double], gb2: [Double]
        public var gW3: [Double], gb3: [Double]
        public var count: Int { gW1.count + gb1.count + gW2.count + gb2.count
                + gW3.count + gb3.count }
    }

    /// Обратный проход: dOut = ∂L/∂pred (длины latentDim).
    public func backward(_ cache: Cache, dOut: [Double]) -> Grads {
        let dimIn = 2 * latentDim + dimEmb
        var gW3 = [Double](repeating: 0, count: latentDim * hidden)
        var gb3 = [Double](repeating: 0, count: latentDim)
        var da2 = [Double](repeating: 0, count: hidden)
        for k in 0..<latentDim {
            let d = dOut[k]
            gb3[k] = d
            let row = k * hidden
            for j in 0..<hidden {
                gW3[row + j] = d * cache.a2[j]
                da2[j] += d * W3[row + j]
            }
        }
        var gW2 = [Double](repeating: 0, count: hidden * hidden)
        var gb2 = [Double](repeating: 0, count: hidden)
        var da1 = [Double](repeating: 0, count: hidden)
        for i in 0..<hidden {
            let dz2 = da2[i] * (1 - cache.a2[i] * cache.a2[i])
            gb2[i] = dz2
            let row = i * hidden
            for j in 0..<hidden {
                gW2[row + j] = dz2 * cache.a1[j]
                da1[j] += dz2 * W2[row + j]
            }
        }
        var gW1 = [Double](repeating: 0, count: hidden * dimIn)
        var gb1 = [Double](repeating: 0, count: hidden)
        for j in 0..<hidden {
            let dz1 = da1[j] * (1 - cache.a1[j] * cache.a1[j])
            gb1[j] = dz1
            let row = j * dimIn
            for l in 0..<dimIn { gW1[row + l] = dz1 * cache.x[l] }
        }
        return Grads(gW1: gW1, gb1: gb1, gW2: gW2, gb2: gb2,
                     gW3: gW3, gb3: gb3)
    }

    // MARK: - Adam

    public func applyAdam(_ g: Grads, t: Int, lr: Double) {
        let beta1 = 0.9, beta2 = 0.999, eps = 1e-8
        let b1c = 1 - pow(beta1, Double(t))
        let b2c = 1 - pow(beta2, Double(t))
        func step(_ p: inout [Double], _ m: inout [Double], _ v: inout [Double],
                  _ grad: [Double]) {
            for i in 0..<p.count {
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]
                let mHat = m[i] / b1c
                let vHat = v[i] / b2c
                p[i] -= lr * mHat / (sqrt(vHat) + eps)
            }
        }
        step(&W1, &mW1, &vW1, g.gW1)
        step(&b1, &mb1, &vb1, g.gb1)
        step(&W2, &mW2, &vW2, g.gW2)
        step(&b2, &mb2, &vb2, g.gb2)
        step(&W3, &mW3, &vW3, g.gW3)
        step(&b3, &mb3, &vb3, g.gb3)
    }

    // MARK: - Матрицы для чекпоинта (Float32-контейнер)

    /// Порядок: [W1, b1, W2, b2, W3, b3].
    public var matrices: [Matrix] {
        [Matrix(W1.map { Float($0) }, rows: hidden),
         Matrix(b1.map { Float($0) }, rows: hidden),
         Matrix(W2.map { Float($0) }, rows: hidden),
         Matrix(b2.map { Float($0) }, rows: hidden),
         Matrix(W3.map { Float($0) }, rows: latentDim),
         Matrix(b3.map { Float($0) }, rows: latentDim)]
    }

    public func load(from mats: [Matrix]) throws {
        guard mats.count == 6 else { throw NSError(domain: "MotionMLPDouble", code: 1) }
        func intoDouble(_ m: Matrix) -> [Double] { m.data.map { Double($0) } }
        W1 = intoDouble(mats[0]); b1 = intoDouble(mats[1])
        W2 = intoDouble(mats[2]); b2 = intoDouble(mats[3])
        W3 = intoDouble(mats[4]); b3 = intoDouble(mats[5])
    }
}
