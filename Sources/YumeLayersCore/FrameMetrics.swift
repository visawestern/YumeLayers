import Foundation

/// Яркость пикселя (luma) из RGB HWC [0,1].
func lumaAt(_ f: [Float], _ i: Int) -> Float {
    0.299 * f[i * 3] + 0.587 * f[i * 3 + 1] + 0.114 * f[i * 3 + 2]
}

/// Максимальная поэлементная разность двух RGB-кадров (0…1).
func maxPixelDelta(_ a: [Float], _ b: [Float]) -> Float {
    let n = min(a.count, b.count)
    var m: Float = 0
    for i in 0..<n { m = max(m, abs(a[i] - b[i])) }
    return m
}
