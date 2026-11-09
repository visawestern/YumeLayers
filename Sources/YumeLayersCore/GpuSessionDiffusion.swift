// YumeLayers — GPU-операции Фазы 3: зашумление латентов для диффузии.
import Foundation
import Metal

extension GpuSession {
    /// Результат GPU-зашумления: xT + коэффициенты расписания.
    struct DiffusedLatentGPU {
        let xT: GPUTensor
        let sqrtAlphaBar: Double
        let sqrtOneMinus: Double
    }

    /// x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε на GPU. Формула совпадает с
    /// `forwardDiffuse` (CPU-эталон), но в одном command buffer с другими op.
    func encodeDiffusedLatents(x0: GPUTensor, t: Int,
                               schedule: DiffusionSchedule,
                               noise: GPUTensor) -> DiffusedLatentGPU {
        precondition(x0.n == noise.n, "encodeDiffusedLatents: размеры совпадают")
        precondition(t >= 0 && t < schedule.timesteps, "t вне [0, timesteps)")
        let alphaBar = schedule.alphaBars[t]
        let sqrtAB = sqrt(max(alphaBar, 0))
        let sqrtOne = sqrt(max(1 - alphaBar, 0))
        let xT = GPUTensor(engine: engine, rows: x0.rows, cols: x0.cols)
        enc.setComputePipelineState(engine.diffusePS)
        setBuffers([x0.buf, noise.buf, xT.buf])
        var sa = Float(sqrtAB)
        var so = Float(sqrtOne)
        enc.setBytes(&sa, length: 4, index: 3)
        enc.setBytes(&so, length: 4, index: 4)
        dispatch(256, 1, x0.n, 1)
        return DiffusedLatentGPU(xT: xT, sqrtAlphaBar: sqrtAB, sqrtOneMinus: sqrtOne)
    }
}
