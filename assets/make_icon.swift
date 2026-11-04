// Генератор иконки YumeLayers: три «слоя» (фон → объекты → свет) на градиенте.
// Запуск: swift assets/make_icon.swift  →  assets/AppIcon.png (1024×1024)
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let s = 1024
let colorSpace = CGColorSpaceCreateDeviceRGB()
guard let ctx = CGContext(data: nil, width: s, height: s, bitsPerComponent: 8,
                          bytesPerRow: 0, space: colorSpace,
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
    fatalError("context")
}

func rgb(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor {
    CGColor(red: r, green: g, blue: b, alpha: a)
}

func roundRect(_ rect: CGRect, _ radius: CGFloat) {
    ctx.addPath(CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil))
    ctx.fillPath()
}

// Фон: сине-фиолетовый градиент
let grad = CGGradient(colorsSpace: colorSpace,
                      colors: [rgb(0.10, 0.12, 0.40), rgb(0.50, 0.22, 0.78)] as CFArray,
                      locations: [0, 1])!
ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: CGFloat(s)),
                       end: CGPoint(x: CGFloat(s), y: 0), options: [])

// Мягкий «стеклянный» блик
ctx.setFillColor(rgb(1, 1, 1, 0.08))
ctx.fillEllipse(in: CGRect(x: 620, y: 180, width: 520, height: 420))

// Слои: фон → объекты → свет
ctx.setFillColor(rgb(0.30, 0.47, 0.95, 0.92))
roundRect(CGRect(x: 210, y: 448, width: 604, height: 420), 84)

ctx.setFillColor(rgb(0.96, 0.44, 0.74, 0.92))
roundRect(CGRect(x: 264, y: 370, width: 500, height: 348), 72)

ctx.setFillColor(rgb(1.00, 0.86, 0.45, 0.95))
roundRect(CGRect(x: 318, y: 292, width: 396, height: 276), 60)

// «Свет» — тёплый круг на верхнем слое
ctx.setFillColor(rgb(1, 1, 1, 0.95))
ctx.fillEllipse(in: CGRect(x: 596, y: 330, width: 118, height: 118))

// Итоговое изображение
let outURL = URL(fileURLWithPath: "assets/AppIcon.png")
let dest = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
guard CGImageDestinationFinalize(dest) else { fatalError("png write") }
print("Иконка: \(outURL.path)")
