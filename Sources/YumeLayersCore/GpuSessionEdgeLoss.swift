// YumeLayers — GPU-операции потерь Этапа 1 (Sobel edge-loss).
// Выделено в отдельный файл ради лимита 600 строк/файл.
import Foundation
import Metal

extension GpuSession {
    /// Градиент Sobel edge-loss по реконструкции (CHW), для dReconOverride:
    /// L₁ = (1/N)·Σ_q (|∇R|_q − |∇T|_q)². Два ядра: карты краёв и
    /// распределение градиента по выходным пикселям (3×3 окно Собеля).
    func encodeSobelEdgeGrad(recon: GPUTensor, target: GPUTensor,
                             w: Int, h: Int) -> GPUTensor {
        precondition(recon.n == target.n && recon.rows >= 3,
                     "encodeSobelEdgeGrad: входы CHW 3×w·h")
        let n = recon.cols
        precondition(n == w * h, "encodeSobelEdgeGrad: cols ≠ w·h")

        let eR = GPUTensor(engine: engine, rows: 1, cols: n)
        let eT = GPUTensor(engine: engine, rows: 1, cols: n)
        let gxR = GPUTensor(engine: engine, rows: 1, cols: n)
        let gyR = GPUTensor(engine: engine, rows: 1, cols: n)
        let invMag = GPUTensor(engine: engine, rows: 1, cols: n)

        enc.setComputePipelineState(engine.sobelEdgeMapPS)
        setBuffers([recon.buf, target.buf, eR.buf, eT.buf,
                    gxR.buf, gyR.buf, invMag.buf])
        var iw = Int32(w), ih = Int32(h)
        enc.setBytes(&iw, length: 4, index: 7)
        enc.setBytes(&ih, length: 4, index: 8)
        dispatch(256, 1, n, 1)

        let out = GPUTensor(engine: engine, rows: 3, cols: n)
        enc.setComputePipelineState(engine.sobelEdgeGradPS)
        setBuffers([eR.buf, eT.buf, gxR.buf, gyR.buf, invMag.buf, out.buf])
        enc.setBytes(&iw, length: 4, index: 6)
        enc.setBytes(&ih, length: 4, index: 7)
        var scale = Float(2.0 / Double(n))
        enc.setBytes(&scale, length: 4, index: 8)
        dispatch(256, 1, n, 1)
        return out
    }
}
