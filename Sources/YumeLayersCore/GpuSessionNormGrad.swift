// YumeLayers — GPU-операция градиента групповой нормализации.
// Выделено в отдельный файл ради лимита 600 строк/файл.
import Foundation
import Metal

extension GpuSession {
    /// Градиент групповой нормализации (CHW) для backward VAE v3:
    /// dx = (1/σ)·(dy − mean(dy) − ((x−μ)/σ²)·mean(dy·(x−μ))) по группам.
    /// Входы/выход — тензоры rows=каналы, cols=плоскость.
    func encodeGroupNormGrad(_ A: GPUTensor, _ dOut: GPUTensor,
                             groups: Int) -> GPUTensor {
        precondition(A.rows == dOut.rows && A.cols == dOut.cols,
                     "encodeGroupNormGrad: размеры совпадают")
        precondition(A.rows % groups == 0 && groups >= 1,
                     "encodeGroupNormGrad: rows % groups == 0")
        let dX = GPUTensor(engine: engine, rows: A.rows, cols: A.cols)
        enc.setComputePipelineState(engine.groupNormGradPS)
        setBuffers([A.buf, dOut.buf, dX.buf])
        setUInt(UInt32(A.rows), index: 3)
        setUInt(UInt32(A.cols), index: 4)
        setUInt(UInt32(groups), index: 5)
        dispatch(1, 1, groups, 1)
        return dX
    }
}
