// YumeLayers — обучение автоэнкодера и диффузии полностью на GPU (Metal).
import Foundation

public extension GPUTensor {
    func toMatrix() -> Matrix {
        return Matrix(toArray(), rows: rows)
    }
}

public func randArr(_ n: Int, scale: Float = 0.08) -> [Float] {
    var rng = SystemRandomNumberGenerator()
    return (0..<n).map { _ in Float.random(in: -scale...scale, using: &rng) }
}

/// Обучение автоэнкодера (VAE-прототип) на GPU.
/// Один command buffer на шаг; батч-векторизация; Adam на GPU.
public func trainAutoencoderGPU(p: Params) throws {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let W = max(p.width, 8), H = max(p.height, 8)
    let I = W * H * 3, L = p.latentDim
    let B = p.batch
    let lr: Float = 0.0008

    guard W <= 128 && H <= 128 else {
        print("MLP-прототип предназначен только для проверки операций (кадры ≤ 128 px).")
        print("Для 480p используйте свёрточный VAE: YumeLayers conv --data <папка>")
        return
    }
    // Один источник геометрии: диффузия и генерация кодируют латент при
    // p.imageSize — рассинхрон с p.width/height давал молча неверный matmul.
    precondition(p.imageSize == W && p.imageSize == H,
                 "trainAutoencoderGPU: imageSize (\(p.imageSize)) должен совпадать с кадром \(W)×\(H)")

    let W1 = GPUTensor(engine: engine, data: randArr(I * H, scale: 0.03), rows: I)
    let b1 = GPUTensor(engine: engine, data: randArr(H, scale: 0.03), rows: 1)
    let W2 = GPUTensor(engine: engine, data: randArr(H * L, scale: 0.03), rows: H)
    let b2 = GPUTensor(engine: engine, data: randArr(L, scale: 0.03), rows: 1)
    let D1 = GPUTensor(engine: engine, data: randArr(L * H, scale: 0.03), rows: L)
    let c1 = GPUTensor(engine: engine, data: randArr(H, scale: 0.03), rows: 1)
    let D2 = GPUTensor(engine: engine, data: randArr(H * I, scale: 0.03), rows: H)
    let c2 = GPUTensor(engine: engine, data: randArr(I, scale: 0.03), rows: 1)
    let params = [W1, b1, W2, b2, D1, c1, D2, c2]
    var ms: [GPUTensor] = []
    var vs: [GPUTensor] = []
    for prm in params {
        ms.append(GPUTensor(engine: engine, rows: prm.rows, cols: prm.cols))
        vs.append(GPUTensor(engine: engine, rows: prm.rows, cols: prm.cols))
    }
    // Моменты Adam — аккумуляторы: приватный буфер Metal не гарантированно
    // нулевой, а adam_kernel читает m/v до записи. Обнуляем явно.
    session.begin()
    for i in 0..<params.count {
        session.zeroFill(ms[i])
        session.zeroFill(vs[i])
    }
    session.end()

    var t = 0
    for ep in 0..<p.epochs {
        var lossSum: Float = 0
        for _ in 0..<20 {
            var Xdata = [Float](repeating: 0, count: B * I)
            for bb in 0..<B {
                let img = makeNormalSynthetic(w: W, h: H)
                for i in 0..<I { Xdata[bb * I + i] = img[i] }
            }
            let X = GPUTensor(engine: engine, data: Xdata, rows: B)

            session.begin()
            // forward
            let H1 = session.addBias(session.matmul(X, W1), b1)
            let A1 = session.tanh(H1)
            let H2 = session.addBias(session.matmul(A1, W2), b2)
            let A2 = session.tanh(H2)
            let Fp = session.addBias(session.matmul(A2, D1), c1)
            let F = session.tanh(Fp)
            let RECON = session.addBias(session.matmul(F, D2), c2)
            let err = session.smul(2.0 / Float(I), session.sub(RECON, X))

            // backward
            let gD2 = session.matmul(F, err, transA: true)
            let gc2 = session.sumCols(err)
            let G_F = session.mul(session.matmul(err, D2, transB: true), session.tanhGrad(Fp))
            let gD1 = session.matmul(A2, G_F, transA: true)
            let gc1 = session.sumCols(G_F)
            let G_Z = session.matmul(G_F, D1, transB: true)
            let gW2 = session.matmul(A1, G_Z, transA: true)
            let gb2 = session.sumCols(G_Z)
            let GH1 = session.mul(session.matmul(G_Z, W2, transB: true), session.tanhGrad(H1))
            let gW1 = session.matmul(X, GH1, transA: true)
            let gb1 = session.sumCols(GH1)

            t += 1
            for i in 0..<8 {
                session.adamUpdate(params[i], [gW1, gb1, gW2, gb2, gD1, gc1, gD2, gc2][i],
                                  ms[i], vs[i], t: t, lr: lr)
            }
            session.end()

            let reconArr = RECON.toArray()
            var mse: Float = 0
            for i in 0..<Xdata.count {
                let d = reconArr[i] - Xdata[i]
                mse += d * d
            }
            lossSum += mse / Float(Xdata.count)
        }
        print("epoch \(ep): loss \(lossSum / 20)")
    }

    try saveState(params.map { $0.toMatrix() }, to: p.vaePath)
    print("VAE (GPU) сохранён в \(p.vaePath)")
}

/// Латентные векторы из синтетических сцен через VAE-энкодер (GPU).
public func encodeLatentsGPU(session: GpuSession, engine: MetalEngine, p: Params,
                      W1: GPUTensor, b1: GPUTensor, W2: GPUTensor, b2: GPUTensor,
                      count: Int) -> [GPUTensor] {
    let S = p.imageSize, I = S * S * 3, B = p.batch
    var latents: [GPUTensor] = []
    for _ in 0..<count {
        var Xdata = [Float](repeating: 0, count: B * I)
        for bb in 0..<B {
            let img = makeNormalSynthetic(size: S)
            for i in 0..<I { Xdata[bb * I + i] = img[i] }
        }
        let X = GPUTensor(engine: engine, data: Xdata, rows: B)
        session.begin()
        let H1 = session.addBias(session.matmul(X, W1), b1)
        let A1 = session.tanh(H1)
        let H2 = session.addBias(session.matmul(A1, W2), b2)
        // Латент = то, что ест декодер при обучении AE: tanh(H2). Раньше
        // возвращался H2 до tanh — диффузия училась на одном распределении,
        // а декодер при генерации получал другое (потерянный tanh).
        let A2 = session.tanh(H2)
        session.end()
        latents.append(A2)
    }
    return latents
}

/// Обучение диффузии (DDPM) на GPU.
public func trainDiffusionGPU(p: Params) throws {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let L = p.latentDim
    let B = p.batch
    let lr: Float = 0.001
    let stepsCount = p.noiseSteps
    guard stepsCount > 0 else {
        print("noiseSteps должен быть > 0 (получено \(stepsCount))")
        return
    }

    // VAE-веса для энкодинга
    let ae = try loadState(path: p.vaePath)
    let W1 = GPUTensor(engine: engine, data: ae[0].data, rows: ae[0].rows)
    let b1 = GPUTensor(engine: engine, data: ae[1].data, rows: ae[1].rows)
    let W2 = GPUTensor(engine: engine, data: ae[2].data, rows: ae[2].rows)
    let b2 = GPUTensor(engine: engine, data: ae[3].data, rows: ae[3].rows)

    // Модель диффузии
    let DW1 = GPUTensor(engine: engine, data: randArr((L + 1) * 64, scale: 0.06), rows: L + 1)
    let Db1 = GPUTensor(engine: engine, data: randArr(64, scale: 0.06), rows: 1)
    let DW2 = GPUTensor(engine: engine, data: randArr(64 * 64, scale: 0.06), rows: 64)
    let Db2 = GPUTensor(engine: engine, data: randArr(64, scale: 0.06), rows: 1)
    let DW3 = GPUTensor(engine: engine, data: randArr(64 * L, scale: 0.06), rows: 64)
    let Db3 = GPUTensor(engine: engine, data: randArr(L, scale: 0.06), rows: 1)
    let params = [DW1, Db1, DW2, Db2, DW3, Db3]
    var ms: [GPUTensor] = []
    var vs: [GPUTensor] = []
    for prm in params {
        ms.append(GPUTensor(engine: engine, rows: prm.rows, cols: prm.cols))
        vs.append(GPUTensor(engine: engine, rows: prm.rows, cols: prm.cols))
    }
    // Моменты Adam — аккумуляторы: обнуляем явно (см. trainAutoencoderGPU).
    session.begin()
    for i in 0..<params.count {
        session.zeroFill(ms[i])
        session.zeroFill(vs[i])
    }
    session.end()

    var t = 0
    for ep in 0..<p.epochs {
        var lossSum: Float = 0
        var n = 0
        for _ in 0..<30 {
            let latents = encodeLatentsGPU(session: session, engine: engine, p: p,
                                           W1: W1, b1: b1, W2: W2, b2: b2, count: 1)
            for Z in latents {
                let tt = Int.random(in: 0..<stepsCount)
                let alpha = 1.0 - Float(tt) / Float(stepsCount)
                var noiseData = [Float](repeating: 0, count: B * L)
                for i in 0..<noiseData.count { noiseData[i] = Float.random(in: -1...1) }
                let noise = GPUTensor(engine: engine, data: noiseData, rows: B)
                // сессия A: zt = qSample(Z, noise)
                session.begin()
                let zt = session.qsample(Z, noise, alpha: alpha)
                session.end()
                let ztArr = zt.toArray()
                // собрать input = [zt | tNorm] на CPU
                let tNorm = Float(tt) / Float(stepsCount)
                var inputData = [Float](repeating: 0, count: B * (L + 1))
                for bb in 0..<B {
                    for j in 0..<L { inputData[bb * (L + 1) + j] = ztArr[bb * L + j] }
                    inputData[bb * (L + 1) + L] = tNorm
                }
                let input = GPUTensor(engine: engine, data: inputData, rows: B)
                // сессия B: forward + backward + adam
                session.begin()
                let H1 = session.addBias(session.matmul(input, DW1), Db1)
                let A1 = session.tanh(H1)
                let H2 = session.addBias(session.matmul(A1, DW2), Db2)
                let A2 = session.tanh(H2)
                let pred = session.addBias(session.matmul(A2, DW3), Db3)
                let err = session.sub(pred, noise)
                let gDW3 = session.matmul(A2, err, transA: true)
                let gDb3 = session.sumCols(err)
                let GA2 = session.matmul(err, DW3, transB: true)
                let GH2 = session.mul(GA2, session.tanhGrad(H2))
                let gDW2 = session.matmul(A1, GH2, transA: true)
                let gDb2 = session.sumCols(GH2)
                let GA1 = session.matmul(GH2, DW2, transB: true)
                let GH1 = session.mul(GA1, session.tanhGrad(H1))
                let gDW1 = session.matmul(input, GH1, transA: true)
                let gDb1 = session.sumCols(GH1)
                t += 1
                let grads = [gDW1, gDb1, gDW2, gDb2, gDW3, gDb3]
                for i in 0..<6 {
                    session.adamUpdate(params[i], grads[i], ms[i], vs[i], t: t, lr: lr)
                }
                session.end()
                // потеря (после выполнения)
                let errArr = err.toArray()
                var mse: Float = 0
                for v in errArr { mse += v * v }
                lossSum += mse / Float(errArr.count); n += 1
            }
        }
        print("diffusion epoch \(ep): loss \(lossSum / Float(max(n, 1)))")
    }
    try saveState(params.map { $0.toMatrix() }, to: p.diffPath)
    print("Диффузия (GPU) сохранена в \(p.diffPath)")
}

/// Генерация на GPU: шум → денойзинг → декодинг VAE → PNG.
public func generateGPU(p: Params) throws {
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let S = p.imageSize, L = p.latentDim
    let stepsCount = p.noiseSteps

    let ae = try loadState(path: p.vaePath)
    let diff = try loadState(path: p.diffPath)

    let D1 = GPUTensor(engine: engine, data: ae[4].data, rows: ae[4].rows)
    let c1 = GPUTensor(engine: engine, data: ae[5].data, rows: ae[5].rows)
    let D2 = GPUTensor(engine: engine, data: ae[6].data, rows: ae[6].rows)
    let c2 = GPUTensor(engine: engine, data: ae[7].data, rows: ae[7].rows)

    let DW1 = GPUTensor(engine: engine, data: diff[0].data, rows: diff[0].rows)
    let Db1 = GPUTensor(engine: engine, data: diff[1].data, rows: 1)
    let DW2 = GPUTensor(engine: engine, data: diff[2].data, rows: diff[2].rows)
    let Db2 = GPUTensor(engine: engine, data: diff[3].data, rows: 1)
    let DW3 = GPUTensor(engine: engine, data: diff[4].data, rows: diff[4].rows)
    let Db3 = GPUTensor(engine: engine, data: diff[5].data, rows: 1)

    var zData = [Float](repeating: 0, count: L)
    for i in 0..<L { zData[i] = Float.random(in: -1...1) }
    var z = GPUTensor(engine: engine, data: zData, rows: 1)

    for step in stride(from: stepsCount - 1, through: 0, by: -1) {
        let alpha = Float(step + 1) / Float(stepsCount)
        let tNorm = Float(step) / Float(stepsCount)
        var inputData = zData
        inputData.append(tNorm)
        let input = GPUTensor(engine: engine, data: inputData, rows: 1)
        session.begin()
        let H1 = session.addBias(session.matmul(input, DW1), Db1)
        let A1 = session.tanh(H1)
        let H2 = session.addBias(session.matmul(A1, DW2), Db2)
        let A2 = session.tanh(H2)
        let pred = session.addBias(session.matmul(A2, DW3), Db3)
        let zNew = session.denoise(z, pred, alpha: alpha)
        session.end()
        let zNewArr = zNew.toArray()
        if step > 0 {
            for i in 0..<L { zData[i] = zNewArr[i] + Float.random(in: -0.1...0.1) }
        } else {
            zData = zNewArr
        }
        z = GPUTensor(engine: engine, data: zData, rows: 1)
    }

    session.begin()
    let F = session.tanh(session.addBias(session.matmul(z, D1), c1))
    let recon = session.addBias(session.matmul(F, D2), c2)
    session.end()
    let rgb = recon.toArray()
    let outPath = "\(p.outPrefix)_0.png"
    try savePNG(path: outPath, width: S, height: S, rgb: rgb)
    print("Генерация (GPU): \(outPath)")
}
