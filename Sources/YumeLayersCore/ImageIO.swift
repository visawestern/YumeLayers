// YumeLayers — IO изображений и перекладка каналов (выделено из ConvTrainer.swift).
//
// Отдельный модуль для соблюдения лимита 600 строк/файл.
import Foundation
import CoreGraphics
import ImageIO

public func loadImageRGB(path: String, w: Int, h: Int) -> [Float]? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    return decodeImageRGB(img, w: w, h: h)
}

public func loadImageRGB(data: Data, w: Int, h: Int) -> [Float]? {
    guard let src = CGImageSourceCreateWithData(data as CFData, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    return decodeImageRGB(img, w: w, h: h)
}

public func decodeImageRGB(_ img: CGImage, w: Int, h: Int) -> [Float]? {
    var pixels = [UInt8](repeating: 0, count: w * h * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: cs,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    // CG рисует снизу вверх — переворачиваем в привычный top-down
    var out = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        let sy = h - 1 - y
        for x in 0..<w {
            let si = (sy * w + x) * 4
            let di = (y * w + x) * 3
            out[di] = Float(pixels[si]) / 255
            out[di + 1] = Float(pixels[si + 1]) / 255
            out[di + 2] = Float(pixels[si + 2]) / 255
        }
    }
    return out
}

public func listImages(in dir: String) -> [String] {
    let fm = FileManager.default
    guard let items = try? fm.contentsOfDirectory(atPath: dir) else { return [] }
    let exts: Set<String> = ["png", "jpg", "jpeg", "bmp", "tiff", "tif"]
    return items.filter { exts.contains(($0 as NSString).pathExtension.lowercased()) }
        .sorted()
        .map { (dir as NSString).appendingPathComponent($0) }
}

public func padToMultipleOf8(_ img: [Float], w: Int, h: Int) -> ([Float], Int, Int) {
    let pw = ((w + 7) / 8) * 8
    let ph = ((h + 7) / 8) * 8
    if pw == w && ph == h { return (img, w, h) }
    var out = [Float](repeating: 0, count: pw * ph * 3)
    for y in 0..<ph {
        let sy = min(y, h - 1)
        for x in 0..<pw {
            let sx = min(x, w - 1)
            let si = (sy * w + sx) * 3
            let di = (y * pw + x) * 3
            out[di] = img[si]; out[di + 1] = img[si + 1]; out[di + 2] = img[si + 2]
        }
    }
    return (out, pw, ph)
}

public func hwcToChw(_ hwc: [Float]) -> [Float] {
    precondition(hwc.count % 3 == 0, "hwcToChw: длина не кратна 3")
    let n = hwc.count / 3
    var out = [Float](repeating: 0, count: hwc.count)
    for c in 0..<3 {
        for i in 0..<n { out[c * n + i] = hwc[i * 3 + c] }
    }
    return out
}

public func chwToHwc(_ chw: [Float]) -> [Float] {
    precondition(chw.count % 3 == 0, "chwToHwc: длина не кратна 3")
    let n = chw.count / 3
    var out = [Float](repeating: 0, count: chw.count)
    for c in 0..<3 {
        for i in 0..<n { out[i * 3 + c] = chw[c * n + i] }
    }
    return out
}

public func savePairPNG(path: String, w: Int, h: Int, target: [Float], recon: [Float]) throws {
    let gap = 6
    let totalW = w * 2 + gap
    var out = [Float](repeating: 0.05, count: totalW * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            let si = (y * w + x) * 3
            let di = (y * totalW + x) * 3
            out[di] = target[si]; out[di + 1] = target[si + 1]; out[di + 2] = target[si + 2]
            let dj = (y * totalW + w + gap + x) * 3
            out[dj] = recon[si]; out[dj + 1] = recon[si + 1]; out[dj + 2] = recon[si + 2]
        }
    }
    try savePNG(path: path, width: totalW, height: h, rgb: out)
}

public func saveSinglePNG(path: String, w: Int, h: Int, rgb: [Float]) throws {
    try savePNG(path: path, width: w, height: h, rgb: rgb)
}

