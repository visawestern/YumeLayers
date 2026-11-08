// YumeLayers — инфраструктура патчей (Этап 1, шаг 4: обучение на кропах).
//
// Цель: обучать VAE на случайных кропах 128–256px из кадров 480p. Сеть
// полностью свёрточная, поэтому на инференсе применяется ко всему кадру
// 854×480 без переобучения. Это снимает VRAM-ограничение и проблему рабочего
// разрешения 80×56 с апскейлом ×10.
import Foundation

/// Случайный кроп кадра (HWC) внутри исходного размера.
///
/// Возвращает (кроп, x0, y0) в координатах исходного кадра. `nil`, если
/// запрошенный размер кропа больше кадра или вход некратен 3·w·h.
func randomCrop(rgb: [Float], w: Int, h: Int, cropW: Int, cropH: Int,
                       rng: inout SplitMix64) -> (crop: [Float], x0: Int, y0: Int)? {
    guard w >= cropW, h >= cropH, cropW > 0, cropH > 0,
          rgb.count == w * h * 3 else { return nil }
    let x0 = Int.random(in: 0...(w - cropW), using: &rng)
    let y0 = Int.random(in: 0...(h - cropH), using: &rng)
    var out = [Float](repeating: 0, count: cropW * cropH * 3)
    for y in 0..<cropH {
        for x in 0..<cropW {
            let si = ((y0 + y) * w + (x0 + x)) * 3
            let di = (y * cropW + x) * 3
            out[di] = rgb[si]; out[di + 1] = rgb[si + 1]; out[di + 2] = rgb[si + 2]
        }
    }
    return (out, x0, y0)
}
