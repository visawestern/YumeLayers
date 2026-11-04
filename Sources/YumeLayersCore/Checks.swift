// YumeLayers Core — проверки корректности GPU-операций против CPU-эталона.
// (Легаси CPU-обучение удалено; здесь только проверки.)
import Foundation
import Darwin

func transpose(_ m: Matrix) -> Matrix {
    var t = Matrix(rows: m.cols, cols: m.rows)
    for r in 0..<m.rows { for c in 0..<m.cols { t[c, r] = m[r, c] } }
    return t
}

public func runCheck() {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    var maxDelta: Float = 0

    func checkMatmul(transA: Bool, transB: Bool) {
        let M = 7, K = 5, N = 9
        let A = Matrix.random(rows: transA ? K : M, cols: transA ? M : K)
        let B = Matrix.random(rows: transB ? N : K, cols: transB ? K : N)
        session.begin()
        let Cgpu = session.matmul(GPUTensor(engine: engine, data: A.data, rows: A.rows),
                                   GPUTensor(engine: engine, data: B.data, rows: B.rows),
                                   transA: transA, transB: transB,
                                   M: M, N: N, K: K)
        session.end()
        let cgpu = Cgpu.toArray()
        // CPU-эталон
        let A2 = transA ? transpose(A) : A
        let B2 = transB ? transpose(B) : B
        let Ccpu = matmulCPU(A2, B2)
        for i in 0..<Ccpu.data.count {
            maxDelta = max(maxDelta, abs(Ccpu.data[i] - cgpu[i]))
        }
    }

    func checkElementwise() {
        let A = Matrix.random(rows: 4, cols: 6)
        let B = Matrix.random(rows: 4, cols: 6)
        session.begin()
        let sAdd = session.add(GPUTensor(engine: engine, data: A.data, rows: 4),
                               GPUTensor(engine: engine, data: B.data, rows: 4))
        let sMul = session.mul(GPUTensor(engine: engine, data: A.data, rows: 4),
                               GPUTensor(engine: engine, data: B.data, rows: 4))
        let sTanh = session.tanh(GPUTensor(engine: engine, data: A.data, rows: 4))
        let sSum = session.sumCols(GPUTensor(engine: engine, data: A.data, rows: 4))
        session.end()
        let gAdd = sAdd.toArray(), gMul = sMul.toArray(), gTanh = sTanh.toArray(), gSum = sSum.toArray()
        for i in 0..<A.data.count {
            maxDelta = max(maxDelta, abs((A.data[i] + B.data[i]) - gAdd[i]))
            maxDelta = max(maxDelta, abs((A.data[i] * B.data[i]) - gMul[i]))
            maxDelta = max(maxDelta, abs(tanhf(A.data[i]) - gTanh[i]))
        }
        for j in 0..<6 {
            var sum: Float = 0
            for r in 0..<4 { sum += A[r, j] }
            maxDelta = max(maxDelta, abs(sum - gSum[j]))
        }
    }

    func checkAdam() {
        let param = Matrix.random(rows: 1, cols: 8, scale: 1)
        let grad = Matrix.random(rows: 1, cols: 8, scale: 1)
        let pGpu = GPUTensor(engine: engine, data: param.data, rows: 1)
        let gGpu = GPUTensor(engine: engine, data: grad.data, rows: 1)
        let mGpu = GPUTensor(engine: engine, rows: 1, cols: 8)
        let vGpu = GPUTensor(engine: engine, rows: 1, cols: 8)
        session.begin()
        session.adamUpdate(pGpu, gGpu, mGpu, vGpu, t: 1, lr: 0.001)
        session.end()
        let pNew = pGpu.toArray()
        // эталон
        let b1: Float = 0.9, b2: Float = 0.999, eps: Float = 1e-8
        for i in 0..<8 {
            let m = b1 * 0 + (1 - b1) * grad.data[i]
            let v = b2 * 0 + (1 - b2) * grad.data[i] * grad.data[i]
            let mHat = m / (1 - b1)
            let vHat = v / (1 - b2)
            let expected = param.data[i] - 0.001 * mHat / (sqrt(vHat) + eps)
            maxDelta = max(maxDelta, abs(expected - pNew[i]))
        }
    }

    checkMatmul(transA: false, transB: false)
    checkMatmul(transA: true, transB: false)
    checkMatmul(transA: false, transB: true)
    checkElementwise()
    checkAdam()

    print("max delta vs CPU: \(maxDelta)")
    let score = maxDelta < 1e-4 ? 100.0 : (maxDelta < 1e-3 ? 90.0 : 60.0)
    print("Оценка раунда 1: \(score)/100")
    if maxDelta < 1e-4 {
        print("✅ Все операции корректны (< 1e-4)")
    } else {
        print("⚠️ Есть расхождения — нужна оптимизация")
    }
}

func matmulCPU(_ a: Matrix, _ b: Matrix) -> Matrix {
    precondition(a.cols == b.rows)
    var out = Matrix(rows: a.rows, cols: b.cols)
    for r in 0..<a.rows {
        for c in 0..<b.cols {
            var s: Float = 0
            for k in 0..<a.cols {
                s += a[r, k] * b[k, c]
            }
            out[r, c] = s
        }
    }
    return out
}
