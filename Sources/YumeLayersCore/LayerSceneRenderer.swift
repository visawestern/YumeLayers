// YumeLayers — рендерер слоёв: процедурные аниме-подобные сцены с ground-truth разметкой.
//
// Центральная идея проекта: сцена = композиция слоёв. Реальные кадры разметки
// слоёв не имеют, поэтому для обучения «раскладыванию по слоям» нужен источник
// пар «кадр ↔ идеальные слои». Этот рендерер — чистый Swift (замена src/dataset.py,
// Python запрещён), порождает детерминированные сцены:
//
//   фон RGB (3) + объекты RGB (3) + маска объектов (1) + карта света (1) = 8 каналов
//   (пространство слоёв зафиксировано в docs/CRITIQUE.md, п. «Зафиксированные решения»).
//
// Сборка (то, что должен выучить декодер VAE):
//   rgb = (фон·(1−маска) + объекты·маска) · свет   → кламп в [0,1].
//
// Все массивы HWC (индекс пикселя (y·w+x), каналы ×3), [0…1].
// Маска бинарная — композиция восстанавливается из слоёв точно.
import Foundation

// ---------- Сцена ----------

public struct LayeredScene {
    public let w: Int
    public let h: Int
    /// Фон: небо/солнце/облака/земля — без объектов, без света. 3·w·h, [0…1].
    public let background: [Float]
    /// Цвет объектов (фигуры «персонажей»). Нули вне маски. 3·w·h, [0…1].
    public let objects: [Float]
    /// Маска объектов: 1 внутри фигур, 0 — фон. w·h.
    public let mask: [Float]
    /// Карта освещённости [0…1]: направленный свет от солнца + виньетка + тени.
    public let light: [Float]
    /// Итоговый кадр: слои, собранные `composeLayers`. 3·w·h, [0…1].
    public let composite: [Float]

    public init(w: Int, h: Int, background: [Float], objects: [Float],
                mask: [Float], light: [Float], composite: [Float]) {
        self.w = w; self.h = h
        self.background = background
        self.objects = objects
        self.mask = mask
        self.light = light
        self.composite = composite
    }

    public static let layerChannels = 8

    /// 8-канальный ground-truth в раскладке свёрточного пайплайна: (канал, H·W).
    /// Порядок каналов: bgR bgG bgB | objR objG objB | mask | light.
    public func channelsCHW() -> [Float] {
        let n = w * h
        var out = [Float](repeating: 0, count: LayeredScene.layerChannels * n)
        for i in 0..<n {
            out[0 * n + i] = background[i * 3]
            out[1 * n + i] = background[i * 3 + 1]
            out[2 * n + i] = background[i * 3 + 2]
            out[3 * n + i] = objects[i * 3]
            out[4 * n + i] = objects[i * 3 + 1]
            out[5 * n + i] = objects[i * 3 + 2]
            out[6 * n + i] = mask[i]
            out[7 * n + i] = light[i]
        }
        return out
    }
}

// ---------- Сборка (обратная операция разложения) ----------

/// Чистая функция сборки кадра из слоёв. Декодер VAE должен приблизить её:
/// rgb = (фон·(1−маска) + объекты·маска)·свет, кламп [0,1].
public func composeLayers(w: Int, h: Int, background: [Float], objects: [Float],
                          mask: [Float], light: [Float]) -> [Float] {
    let n = w * h
    precondition(background.count >= n * 3 && objects.count >= n * 3
                 && mask.count >= n && light.count >= n)
    var rgb = [Float](repeating: 0, count: n * 3)
    for i in 0..<n {
        let m = mask[i]
        let l = light[i]
        let im = 1 - m
        var c = 0
        while c < 3 {
            let v = (background[i * 3 + c] * im + objects[i * 3 + c] * m) * l
            rgb[i * 3 + c] = min(1, max(0, v))
            c += 1
        }
    }
    return rgb
}

// ---------- Плавные переходы ----------

/// Гладкий порог: 0 при t≤e0, 1 при t≥e1, между — кубическая интерполяция.
/// Для мягких краёв облаков/теней (не влияет на точность композиции:
/// свет и фон сглаживаются ДО сборки).
@inline(__always)
func smoothstep(_ e0: Float, _ e1: Float, _ t: Float) -> Float {
    let x = min(1, max(0, (t - e0) / max(e1 - e0, 1e-6)))
    return x * x * (3 - 2 * x)
}

@inline(__always)
func clamp01(_ v: Float) -> Float { min(1, max(0, v)) }

// ---------- Персонаж ----------

/// Одна фигура: ноги на земле, три формы (волосы/голова/тело), одна маска.
/// Тип на уровне файла: вложенные типы нельзя объявлять в generic-функции.
private struct Figure {
    var cx: Float; var feetY: Float; var r: Float
    var hair: (Float, Float, Float)
    var skin: (Float, Float, Float)
    var cloth: (Float, Float, Float)
}

// ---------- Рендерер ----------

/// Порождает сцену с разметкой слоёв. Детерминирован при фиксированном RNG.
///
/// Содержимое:
/// - фон: небо-градиент, солнце (диск), 2–4 облака, полоса земли с своим градиентом;
/// - объекты: 1–3 «персонажа» (волосы + голова + тело), маска бинарная;
/// - свет: направленный от солнца + виньетка + мягкие тени под персонажами.
public func makeLayeredScene<R: RandomNumberGenerator>(w: Int, h: Int, rng: inout R) -> LayeredScene {
    precondition(w >= 16 && h >= 16, "сцена слишком мала для слоёв")
    let n = w * h
    var bg = [Float](repeating: 0, count: n * 3)
    var obj = [Float](repeating: 0, count: n * 3)
    var mask = [Float](repeating: 0, count: n)
    var light = [Float](repeating: 0, count: n)

    // --- Небо и солнце ---
    let top = (Float.random(in: 0.2...0.7, using: &rng),
               Float.random(in: 0.35...0.75, using: &rng),
               Float.random(in: 0.6...0.95, using: &rng))          // верх насыщеннее
    let horizon: (Float, Float, Float) = (min(1, top.0 + 0.25), min(1, top.1 + 0.2), 0.9) // у горизонта светлее
    let sunX = Float.random(in: 0.1...0.9, using: &rng) * Float(w)
    let sunY = Float.random(in: 0.04...0.28, using: &rng) * Float(h)
    let sunR = Float.random(in: 0.03...0.07, using: &rng) * Float(min(w, h))
    let sunColor: (Float, Float, Float) = (1.0, 0.97, 0.88)

    // Облака: центр, полуоси, плотность.
    let cloudCount = Int.random(in: 2...4, using: &rng)
    var clouds: [(cx: Float, cy: Float, ax: Float, ay: Float, a: Float)] = []
    for _ in 0..<cloudCount {
        clouds.append((Float.random(in: 0.05...0.95, using: &rng) * Float(w),
                       Float.random(in: 0.05...0.45, using: &rng) * Float(h),
                       Float.random(in: 0.06...0.22, using: &rng) * Float(w),
                       Float.random(in: 0.02...0.05, using: &rng) * Float(h),
                       Float.random(in: 0.35...0.8, using: &rng)))
    }

    // Земля: линия горизонта и два цвета.
    let gy = Int(Float.random(in: 0.62...0.78, using: &rng) * Float(h))  // y линии горизонта
    let groundA = (Float.random(in: 0.15...0.45, using: &rng),
                   Float.random(in: 0.35...0.65, using: &rng),
                   Float.random(in: 0.10...0.35, using: &rng))   // у горизонта
    let groundB = (groundA.0 * 0.6, groundA.1 * 0.75, groundA.2 * 0.5) // дальний план темнее

    for y in 0..<h {
        let fy = Float(y)
        for x in 0..<w {
            let fx = Float(x)
            let i = (y * w + x) * 3
            var r: Float, g: Float, b: Float
            if y < gy {
                // небо: вертикальный градиент
                let t = fy / Float(max(gy, 1))
                r = top.0 * (1 - t) + horizon.0 * t
                g = top.1 * (1 - t) + horizon.1 * t
                b = top.2 * (1 - t) + horizon.2 * t
                // солнце: мягкий диск
                let d = sqrt((fx - sunX) * (fx - sunX) + (fy - sunY) * (fy - sunY))
                let s = smoothstep(sunR, sunR * 0.6, d)   // 1 в центре, 0 снаружи
                if s > 0 {
                    r = r * (1 - s) + sunColor.0 * s
                    g = g * (1 - s) + sunColor.1 * s
                    b = b * (1 - s) + sunColor.2 * s
                }
                // облака: мягкие эллипсы, подмешиваются к небу
                for c in clouds {
                    let dx = (fx - c.cx) / c.ax
                    let dy = (fy - c.cy) / c.ay
                    let d = sqrt(dx * dx + dy * dy)
                    let f = c.a * (1 - smoothstep(0.65, 1.05, d))
                    if f > 0 {
                        r = r * (1 - f) + 0.98 * f
                        g = g * (1 - f) + 0.99 * f
                        b = b * (1 - f) + 1.0 * f
                    }
                }
            } else {
                // земля: свой градиент
                let t = (fy - Float(gy)) / Float(max(h - gy, 1))
                r = groundA.0 * (1 - t) + groundB.0 * t
                g = groundA.1 * (1 - t) + groundB.1 * t
                b = groundA.2 * (1 - t) + groundB.2 * t
            }
            bg[i] = clamp01(r); bg[i + 1] = clamp01(g); bg[i + 2] = clamp01(b)
        }
    }

    // --- Персонажи ---
    let figureCount = Int.random(in: 1...3, using: &rng)
    var figures: [Figure] = []
    for _ in 0..<figureCount {
        let r = Float.random(in: 0.030...0.065, using: &rng) * Float(min(w, h))
        let cx = Float.random(in: 0.15...0.85, using: &rng) * Float(w)
        // ноги на земле (чуть ниже горизонта, не упираясь в край)
        let feetLo = Float(gy) + 2 * r
        let feetHi = min(Float(h) - r, Float(gy) + Float(h - gy) * 0.8)
        let feetY = Float.random(in: feetLo...max(feetLo + 1, feetHi), using: &rng)
        figures.append(Figure(
            cx: cx, feetY: feetY, r: r,
            hair: (Float.random(in: 0.15...0.95, using: &rng),
                   Float.random(in: 0.15...0.85, using: &rng),
                   Float.random(in: 0.15...0.95, using: &rng)),
            skin: (0.96, 0.83, 0.73),
            cloth: (Float.random(in: 0.1...0.9, using: &rng),
                    Float.random(in: 0.1...0.9, using: &rng),
                    Float.random(in: 0.1...0.9, using: &rng))))
    }

    // Бинарная маска + цвета. Порядок отрисовки: волосы → тело → голова (голова сверху).
    for f in figures {
        let r = f.r
        let bodyCY = f.feetY - 2.1 * r           // центр эллипса тела
        let headCY = bodyCY - 2.55 * r           // центр головы
        let hairCY = headCY - 0.22 * r           // «шапка» волос чуть выше
        let axBody = r, ayBody = 2.2 * r
        let rHead = 0.95 * r
        let axHair = 1.25 * r, ayHair = 1.0 * r

        let x0 = max(0, Int(f.cx - max(axBody, axHair, rHead) - 1))
        let x1 = min(w - 1, Int(f.cx + max(axBody, axHair, rHead) + 1))
        let y0 = max(0, Int(hairCY - ayHair - 1))
        let y1 = min(h - 1, Int(f.feetY + 1))

        for y in y0...y1 {
            let fy = Float(y)
            for x in x0...x1 {
                let fx = Float(x)
                var color: (Float, Float, Float)? = nil
                // тело: эллипс
                let dxB = (fx - f.cx) / axBody
                let dyB = (fy - bodyCY) / ayBody
                if dxB * dxB + dyB * dyB <= 1 { color = f.cloth }
                // волосы: эллипс за головой
                if color == nil {
                    let dxH = (fx - f.cx) / axHair
                    let dyH = (fy - hairCY) / ayHair
                    if dxH * dxH + dyH * dyH <= 1 { color = f.hair }
                }
                // голова: круг поверх волос
                let dx = fx - f.cx, dy = fy - headCY
                if dx * dx + dy * dy <= rHead * rHead { color = f.skin }
                if let c = color {
                    let i = (y * w + x) * 3
                    obj[i] = c.0; obj[i + 1] = c.1; obj[i + 2] = c.2
                    mask[y * w + x] = 1
                }
            }
        }
    }

    // --- Свет ---
    // Направленный: ярче у солнца. dNorm — расстояние до солнца в долях диагонали.
    let diag = sqrt(Float(w * w + h * h))
    for y in 0..<h {
        let fy = Float(y)
        for x in 0..<w {
            let fx = Float(x)
            let dx = fx - sunX, dy = fy - sunY
            let dNorm = min(1, sqrt(dx * dx + dy * dy) / diag * 1.6)
            var l = 0.58 + 0.42 * (1 - dNorm)
            // виньетка: темнее к углам
            let vx = fx / Float(max(w - 1, 1)) - 0.5
            let vy = fy / Float(max(h - 1, 1)) - 0.5
            l *= 1 - 0.30 * (vx * vx + vy * vy) * 2
            light[y * w + x] = clamp01(l)
        }
    }
    // Мягкие тени на земле под персонажами: эллипс, свет −0.4·фактор.
    for f in figures {
        let shCY = min(Float(h - 1), f.feetY + 0.12 * f.r)
        let axS = 1.9 * f.r, ayS = 0.38 * f.r
        let x0 = max(0, Int(f.cx - axS - 1))
        let x1 = min(w - 1, Int(f.cx + axS + 1))
        let y0 = max(gy, Int(shCY - ayS - 1))
        let y1 = min(h - 1, Int(shCY + ayS + 1))
        for y in y0...y1 {
            for x in x0...x1 {
                let dx = (Float(x) - f.cx) / axS
                let dy = (Float(y) - shCY) / ayS
                let d = sqrt(dx * dx + dy * dy)
                let k = 1 - smoothstep(0.5, 1.0, d)   // 1 в центре тени
                if k > 0 {
                    let i = y * w + x
                    light[i] = clamp01(light[i] - 0.4 * k)
                }
            }
        }
    }

    let composite = composeLayers(w: w, h: h, background: bg, objects: obj, mask: mask, light: light)
    return LayeredScene(w: w, h: h, background: bg, objects: obj,
                        mask: mask, light: light, composite: composite)
}

/// Детерминированная обёртка: одна сцена — один seed.
public func makeLayeredScene(w: Int, h: Int, seed: UInt64) -> LayeredScene {
    var rng = SplitMix64(seed: seed)
    return makeLayeredScene(w: w, h: h, rng: &rng)
}

// ---------- Предпросмотр ----------

/// Сохраняет превью слоёв в PNG: <prefix>_composite/_background/_objects/_light.png.
/// Фон и объекты показаны с применённым светом (как в итоговом кадре),
/// свет — градациями серого.
public func saveLayerPreview(scene: LayeredScene, prefix: String) throws {
    let dir = (prefix as NSString).deletingLastPathComponent
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)

    let n = scene.w * scene.h
    // фон·свет и объекты·свет — для наглядности
    var bgLit = [Float](repeating: 0, count: n * 3)
    var objLit = [Float](repeating: 0, count: n * 3)
    var lightRGB = [Float](repeating: 0, count: n * 3)
    for i in 0..<n {
        let l = scene.light[i]
        for c in 0..<3 {
            bgLit[i * 3 + c] = clamp01(scene.background[i * 3 + c] * l)
            objLit[i * 3 + c] = clamp01(scene.objects[i * 3 + c] * l)
            lightRGB[i * 3 + c] = l
        }
    }
    try savePNG(path: prefix + "_composite.png", width: scene.w, height: scene.h, rgb: scene.composite)
    try savePNG(path: prefix + "_background.png", width: scene.w, height: scene.h, rgb: bgLit)
    try savePNG(path: prefix + "_objects.png", width: scene.w, height: scene.h, rgb: objLit)
    try savePNG(path: prefix + "_light.png", width: scene.w, height: scene.h, rgb: lightRGB)
}
