// YumeLayers — минимальные матричные операции на Accelerate/vDSP.
import Foundation
import Accelerate

public struct Matrix {
    var data: [Float]
    let rows: Int
    let cols: Int
    init(rows: Int, cols: Int, fill: Float = 0) {
        self.rows = rows; self.cols = cols
        self.data = [Float](repeating: fill, count: rows * cols)
    }
    init(_ data: [Float], rows: Int) {
        self.rows = rows
        self.cols = data.count / rows
        self.data = data
    }
    subscript(r: Int, c: Int) -> Float {
        get { data[r * cols + c] }
        set { data[r * cols + c] = newValue }
    }
    func copy() -> Matrix { Matrix(data, rows: rows) }
    static func zerosLike(_ m: Matrix) -> Matrix { Matrix(rows: m.rows, cols: m.cols) }
    static func random(rows: Int, cols: Int, scale: Float = 0.05) -> Matrix {
        var m = Matrix(rows: rows, cols: cols)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<m.data.count {
            m.data[i] = Float.random(in: -scale...scale, using: &rng)
        }
        return m
    }
}

public func matmul(_ a: Matrix, _ b: Matrix) -> Matrix {
    precondition(a.cols == b.rows)
    let m = a.rows, n = b.cols, k = a.cols
    // Мелкие матрицы — CPU (vDSP), чтобы избежать overhead GPU-вызовов.
    // Большие — Metal GPU.
    let elements = m * k * n
    if elements >= 131072 {  // >= 128k флопов-элементов
        let data = MetalEngine.shared.multiply(a.data, b.data, m: m, k: k, n: n)
        return Matrix(data, rows: m)
    }
    var out = Matrix(rows: m, cols: n)
    if m > 0 && n > 0 && k > 0 {
        a.data.withUnsafeBufferPointer { pa in
            b.data.withUnsafeBufferPointer { pb in
                out.data.withUnsafeMutableBufferPointer { po in
                    vDSP_mmul(pa.baseAddress!, 1, pb.baseAddress!, 1,
                               po.baseAddress!, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
                }
            }
        }
    }
    return out
}

public func add(_ a: Matrix, _ b: Matrix) -> Matrix {
    // vDSP длины не проверяет: короткий b — чтение за границей буфера.
    precondition(a.data.count == b.data.count, "add: размеры матриц не совпадают")
    var out = a.copy()
    vDSP_vadd(a.data, 1, b.data, 1, &out.data, 1, vDSP_Length(a.data.count))
    return out
}

public func mul(_ a: Matrix, _ b: Matrix) -> Matrix {
    precondition(a.data.count == b.data.count, "mul: размеры матриц не совпадают")
    var out = a.copy()
    vDSP_vmul(a.data, 1, b.data, 1, &out.data, 1, vDSP_Length(a.data.count))
    return out
}

public func scalarMul(_ s: Float, _ a: Matrix) -> Matrix {
    var out = a.copy()
    vDSP_vsmul(a.data, 1, [s], &out.data, 1, vDSP_Length(a.data.count))
    return out
}

public func addScalar(_ s: Float, _ a: Matrix) -> Matrix {
    var out = a.copy()
    for i in 0..<out.data.count { out.data[i] += s }
    return out
}

public func relu(_ a: Matrix) -> Matrix {
    var out = a.copy()
    for i in 0..<out.data.count { if out.data[i] < 0 { out.data[i] = 0 } }
    return out
}

public func reluGrad(_ a: Matrix) -> Matrix {
    var out = Matrix(rows: a.rows, cols: a.cols)
    for i in 0..<a.data.count { out.data[i] = a.data[i] > 0 ? 1 : 0 }
    return out
}

public func tanhAct(_ a: Matrix) -> Matrix {
    var out = a.copy()
    for i in 0..<out.data.count { out.data[i] = tanhf(out.data[i]) }
    return out
}

public func tanhGrad(_ a: Matrix) -> Matrix {
    var out = Matrix(rows: a.rows, cols: a.cols)
    for i in 0..<a.data.count { let t = tanhf(a.data[i]); out.data[i] = 1 - t * t }
    return out
}

public func meanFlat(_ a: Matrix) -> Float {
    var s: Float = 0
    for v in a.data { s += v }
    return s / Float(a.data.count)
}

public func copyCols(_ a: Matrix, columns: Int, fill: Float = 0) -> Matrix {
    var out = Matrix(rows: a.rows, cols: columns, fill: fill)
    for r in 0..<a.rows { out[r, 0] = a[r, 0] }
    return out
}
