// YumeLayers — универсальная слоистая архитектура: композитор слоёв-экспертов.
//
// АРХИТЕКТУРНЫЙ ПРИНЦИП (не «три фиксированных модели», а факторизация):
//
//   Сцена = упорядоченный стек НЕЗАВИСИМЫХ слоёв. Каждый слой:
//     • rgb + alpha         — цветовой вклад (alpha-over композиция);
//     • lightMul/lightDelta — световой вклад (умножение / вычитание карты света);
//     • role                — КТО произвёл слой (эксперт: фон/объекты/свет/движение);
//     • group               — КУДА слой попадает в 8-канальном ground-truth.
//
//   Роль НЕ привязана к семантике: фон-эксперт МОЖЕТ породить слой с фигурой,
//   объектный эксперт — слой с небом, световой — фон. Композиция зависит только
//   от порядка слоёв в стеке, а не от того, кто их произвёл. Поэтому фон можно
//   ДОПОЛНЯТЬ любым числом объектных экспертов, комбинировать небо+земля+фигуры
//   в любом наборе и обучать каждый эксперт отдельно: модель на 30–70 ГБ никогда
//   не грузится в VRAM целиком — активен только текущий эксперт (остальные в RAM),
//   а обмен между экспертами — латенты/слои, а не веса.
//
//   8-канальный таргет (docs/CRITIQUE.md): фон RGB | объекты RGB | маска | свет.
//   Маска = вклад объектной группы в итоговый пиксель (1, если объектные слои
//   изменили цвет). Такое определение гарантирует ТОЧНУЮ обратную сборку:
//     composeLayers(flattened) == composite   (побитово, для любого рецепта).
//
//   Сборка (её должен выучить декодер): rgb = (фон·(1−m)+объекты·m)·свет.
//
// Чистый Swift, без Python. Все массивы HWC, [0…1], индекс (y·w+x).
import Foundation

// ---------- Модель слоёв ----------

public enum LayerKind: String, CaseIterable {
    case sky, ground, clouds, sun, figure, lightGradient, shadow
}

/// Кто произвёл слой. Это МЕТКА ПРОИСХОЖДЕНИЯ, а не ограничение на содержимое:
/// фон-эксперт может рисовать фигуры, объектный — небо.
public enum ExpertRole: String {
    case background, objects, lighting, motion
}

/// Канал ground-truth, в который попадает цвет слоя.
public enum ChannelGroup {
    case background, objects
}

public struct LayerSpec {
    public let kind: LayerKind
    public let count: Int
    public let role: ExpertRole
    public let group: ChannelGroup

    public init(_ kind: LayerKind, count: Int = 1, role: ExpertRole,
                group: ChannelGroup? = nil) {
        self.kind = kind
        self.count = max(1, count)
        self.role = role
        self.group = group ?? (kind == .figure ? .objects : .background)
    }
}

/// Рецепт сцены: ЛЮБОЙ набор слоёв в ЛЮБОМ порядке. Расширяемый:
/// `recipe.adding([...])` — фон дополняется эксперт-слоями объектов.
public struct LayerRecipe {
    public var specs: [LayerSpec]
    public init(_ specs: [LayerSpec] = []) { self.specs = specs }

    /// Дополнить сцену слоями (например, фон → + объектные эксперты).
    public func adding(_ more: [LayerSpec]) -> LayerRecipe {
        var r = self
        r.specs.append(contentsOf: more)
        return r
    }

    public func adding(_ spec: LayerSpec) -> LayerRecipe { adding([spec]) }

    /// Классическая сцена «небо + солнце + облака + земля + фигуры + свет + тени».
    public static func classic(figureCount: Int = 2) -> LayerRecipe {
        LayerRecipe([
            LayerSpec(.sky, role: .background),
            LayerSpec(.sun, role: .background),
            LayerSpec(.clouds, count: 3, role: .background),
            LayerSpec(.ground, role: .background),
            LayerSpec(.lightGradient, role: .lighting),
        ])
        .adding(LayerSpec(.figure, count: figureCount, role: .objects))
        .adding(LayerSpec(.shadow, count: figureCount, role: .lighting))
    }

    /// Только фон — стартовая точка для «дополнения объектами».
    public static func backdrop() -> LayerRecipe {
        LayerRecipe([
            LayerSpec(.sky, role: .background),
            LayerSpec(.sun, role: .background),
            LayerSpec(.clouds, count: 3, role: .background),
            LayerSpec(.ground, role: .background),
            LayerSpec(.lightGradient, role: .lighting),
        ])
    }
}

/// Один отрендеренный слой: цвет (rgb+alpha) ИЛИ свет (mul/delta).
public struct RenderedLayer {
    public let spec: LayerSpec
    public let rgb: [Float]?          // 3·n
    public let alpha: [Float]?        // n
    public let lightMul: [Float]?     // n
    public let lightDelta: [Float]?   // n
}

/// Сцена до «сплющивания» в 8 каналов: полный стек + свет + кадр.
public struct ComposedScene {
    public let w: Int
    public let h: Int
    public let recipe: LayerRecipe
    public let layers: [RenderedLayer]
    public let light: [Float]
    public let composite: [Float]

    /// 8-канальный ground-truth для обучения, с точной обратной сборкой.
    public func flattened() -> LayeredScene {
        flattenStack(w: w, h: h, layers: layers, light: light, composite: composite)
    }
}

// ---------- 8-канальный таргет и сборка ----------

public struct LayeredScene {
    public let w: Int
    public let h: Int
    /// Фон: все слои группы .background, без объектных, без света. 3·w·h, [0…1].
    public let background: [Float]
    /// Видимый вклад объектных слоёв. Нули вне маски. 3·w·h, [0…1].
    public let objects: [Float]
    /// Маска вклада объектов: 1 там, где они изменили кадр. w·h, {0,1}.
    public let mask: [Float]
    /// Карта освещённости [0…1].
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

    /// 8 каналов в раскладке свёрточного пайплайна: (канал, H·W).
    /// Порядок: bgR bgG bgB | objR objG objB | mask | light.
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

/// Чистая функция сборки кадра из 8-канального таргета. Декодер VAE
/// должен приблизить её: rgb = (фон·(1−маска) + объекты·маска)·свет.
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

// ---------- Композитор ----------

@inline(__always)
func smoothstep(_ e0: Float, _ e1: Float, _ t: Float) -> Float {
    let x = min(1, max(0, (t - e0) / max(e1 - e0, 1e-6)))
    return x * x * (3 - 2 * x)
}

@inline(__always)
func clamp01(_ v: Float) -> Float { min(1, max(0, v)) }

/// Alpha-over композиция цветовых слоёв в порядке стека.
public func compositeColorLayers(w: Int, h: Int, layers: [RenderedLayer],
                                 group: ChannelGroup? = nil) -> [Float] {
    let n = w * h
    var color = [Float](repeating: 0, count: n * 3)
    for layer in layers {
        if let g = group, layer.spec.group != g { continue }
        guard let rgb = layer.rgb, let alpha = layer.alpha else { continue }
        for i in 0..<n {
            let a = clamp01(alpha[i])
            if a <= 0 { continue }
            let inv = 1 - a
            color[i * 3]     = color[i * 3]     * inv + rgb[i * 3]     * a
            color[i * 3 + 1] = color[i * 3 + 1] * inv + rgb[i * 3 + 1] * a
            color[i * 3 + 2] = color[i * 3 + 2] * inv + rgb[i * 3 + 2] * a
        }
    }
    return color
}

/// Свет: произведение mul, минус delta, кламп [0.05…1].
public func composeLightLayers(w: Int, h: Int, layers: [RenderedLayer]) -> [Float] {
    let n = w * h
    var light = [Float](repeating: 1, count: n)
    for layer in layers {
        if let mul = layer.lightMul {
            for i in 0..<n { light[i] *= mul[i] }
        }
        if let delta = layer.lightDelta {
            for i in 0..<n { light[i] -= delta[i] }
        }
    }
    for i in 0..<n { light[i] = min(1, max(0.05, light[i])) }
    return light
}

/// Сплющивание стека в 8 каналов. Маска = «объектные слои изменили пиксель».
/// Это определение делает composeLayers(flattened) == composite побитово
/// для ЛЮБОГО рецепта, включая перекрытия и полупрозрачные слои поверх объектов.
public func flattenStack(w: Int, h: Int, layers: [RenderedLayer], light: [Float],
                         composite: [Float]) -> LayeredScene {
    let n = w * h
    let color = compositeColorLayers(w: w, h: h, layers: layers)
    let bgOnly = compositeColorLayers(w: w, h: h, layers: layers, group: .background)
    var objects = [Float](repeating: 0, count: n * 3)
    var mask = [Float](repeating: 0, count: n)
    for i in 0..<n {
        let differs = color[i * 3] != bgOnly[i * 3]
            || color[i * 3 + 1] != bgOnly[i * 3 + 1]
            || color[i * 3 + 2] != bgOnly[i * 3 + 2]
        if differs {
            mask[i] = 1
            objects[i * 3] = color[i * 3]
            objects[i * 3 + 1] = color[i * 3 + 1]
            objects[i * 3 + 2] = color[i * 3 + 2]
        }
    }
    return LayeredScene(w: w, h: h, background: bgOnly, objects: objects,
                        mask: mask, light: light, composite: composite)
}

// ---------- Параметры сцены (все случайности — ДО рендера, в фиксированном порядке) ----------

/// ВАЖНО: параметры генерируются в фиксированном порядке независимо от рецепта.
/// Благодаря этому `backdrop()` и `backdrop().adding(фигуры)` дают Побитово
/// одинаковый фон — добавление эксперта не «перемешивает» сцену.
private struct CloudParams { var cx, cy, ax, ay, a: Float }
private struct FigureParams {
    var cx, feetY, r: Float
    var hair: (Float, Float, Float)
    var skin: (Float, Float, Float)
    var cloth: (Float, Float, Float)
}
private struct SceneParams {
    var top: (Float, Float, Float)
    var horizon: (Float, Float, Float)
    var sunX, sunY, sunR: Float
    var gy: Int
    var groundA: (Float, Float, Float)
    var groundB: (Float, Float, Float)
    var clouds: [CloudParams]
    var figures: [FigureParams]
}

private func makeSceneParams<R: RandomNumberGenerator>(w: Int, h: Int, rng: inout R) -> SceneParams {
    let top = (Float.random(in: 0.2...0.7, using: &rng),
               Float.random(in: 0.35...0.75, using: &rng),
               Float.random(in: 0.6...0.95, using: &rng))
    let horizon: (Float, Float, Float) = (min(1, top.0 + 0.25), min(1, top.1 + 0.2), 0.9)
    let sunX = Float.random(in: 0.1...0.9, using: &rng) * Float(w)
    let sunY = Float.random(in: 0.04...0.28, using: &rng) * Float(h)
    let sunR = Float.random(in: 0.03...0.07, using: &rng) * Float(min(w, h))
    let gy = Int(Float.random(in: 0.62...0.78, using: &rng) * Float(h))
    let groundA = (Float.random(in: 0.15...0.45, using: &rng),
                   Float.random(in: 0.35...0.65, using: &rng),
                   Float.random(in: 0.10...0.35, using: &rng))
    let groundB = (groundA.0 * 0.6, groundA.1 * 0.75, groundA.2 * 0.5)

    var clouds: [CloudParams] = []
    for _ in 0..<8 {
        clouds.append(CloudParams(cx: Float.random(in: 0.05...0.95, using: &rng) * Float(w),
                                  cy: Float.random(in: 0.05...0.45, using: &rng) * Float(h),
                                  ax: Float.random(in: 0.06...0.22, using: &rng) * Float(w),
                                  ay: Float.random(in: 0.02...0.05, using: &rng) * Float(h),
                                  a: Float.random(in: 0.35...0.8, using: &rng)))
    }
    var figures: [FigureParams] = []
    for _ in 0..<8 {
        let r = Float.random(in: 0.030...0.065, using: &rng) * Float(min(w, h))
        let cx = Float.random(in: 0.15...0.85, using: &rng) * Float(w)
        let feetLo = Float(gy) + 2 * r
        let feetHi = min(Float(h) - r, Float(gy) + Float(h - gy) * 0.8)
        let feetY = Float.random(in: feetLo...max(feetLo + 1, feetHi), using: &rng)
        figures.append(FigureParams(
            cx: cx, feetY: feetY, r: r,
            hair: (Float.random(in: 0.15...0.95, using: &rng),
                   Float.random(in: 0.15...0.85, using: &rng),
                   Float.random(in: 0.15...0.95, using: &rng)),
            skin: (0.96, 0.83, 0.73),
            cloth: (Float.random(in: 0.1...0.9, using: &rng),
                    Float.random(in: 0.1...0.9, using: &rng),
                    Float.random(in: 0.1...0.9, using: &rng))))
    }
    return SceneParams(top: top, horizon: horizon, sunX: sunX, sunY: sunY, sunR: sunR,
                       gy: gy, groundA: groundA, groundB: groundB,
                       clouds: clouds, figures: figures)
}

// ---------- Отрисовка отдельных слоёв ----------

private func skyLayer(w: Int, h: Int, p: SceneParams, spec: LayerSpec) -> RenderedLayer {
    let n = w * h
    var rgb = [Float](repeating: 0, count: n * 3)
    let alpha = [Float](repeating: 1, count: n)
    for y in 0..<h {
        let t = Float(y) / Float(max(h - 1, 1))
        for x in 0..<w {
            let i = (y * w + x) * 3
            rgb[i] = clamp01(p.top.0 * (1 - t) + p.horizon.0 * t)
            rgb[i + 1] = clamp01(p.top.1 * (1 - t) + p.horizon.1 * t)
            rgb[i + 2] = clamp01(p.top.2 * (1 - t) + p.horizon.2 * t)
        }
    }
    return RenderedLayer(spec: spec, rgb: rgb, alpha: alpha, lightMul: nil, lightDelta: nil)
}

private func groundLayer(w: Int, h: Int, p: SceneParams, spec: LayerSpec) -> RenderedLayer {
    let n = w * h
    var rgb = [Float](repeating: 0, count: n * 3)
    var alpha = [Float](repeating: 0, count: n)
    for y in p.gy..<h {
        let t = (Float(y) - Float(p.gy)) / Float(max(h - p.gy, 1))
        for x in 0..<w {
            let i = (y * w + x) * 3
            rgb[i] = clamp01(p.groundA.0 * (1 - t) + p.groundB.0 * t)
            rgb[i + 1] = clamp01(p.groundA.1 * (1 - t) + p.groundB.1 * t)
            rgb[i + 2] = clamp01(p.groundA.2 * (1 - t) + p.groundB.2 * t)
            alpha[y * w + x] = 1
        }
    }
    return RenderedLayer(spec: spec, rgb: rgb, alpha: alpha, lightMul: nil, lightDelta: nil)
}

private func sunLayer(w: Int, h: Int, p: SceneParams, spec: LayerSpec) -> RenderedLayer {
    let n = w * h
    var rgb = [Float](repeating: 0, count: n * 3)
    var alpha = [Float](repeating: 0, count: n)
    for y in 0..<h {
        for x in 0..<w {
            let dx = Float(x) - p.sunX, dy = Float(y) - p.sunY
            let d = sqrt(dx * dx + dy * dy)
            let a = smoothstep(p.sunR, p.sunR * 0.6, d)
            if a > 0 {
                let i = (y * w + x) * 3
                rgb[i] = 1.0; rgb[i + 1] = 0.97; rgb[i + 2] = 0.88
                alpha[y * w + x] = a
            }
        }
    }
    return RenderedLayer(spec: spec, rgb: rgb, alpha: alpha, lightMul: nil, lightDelta: nil)
}

private func cloudLayer(w: Int, h: Int, c: CloudParams, spec: LayerSpec) -> RenderedLayer {
    let n = w * h
    var rgb = [Float](repeating: 0, count: n * 3)
    var alpha = [Float](repeating: 0, count: n)
    let x0 = max(0, Int(c.cx - c.ax * 1.1)), x1 = min(w - 1, Int(c.cx + c.ax * 1.1))
    let y0 = max(0, Int(c.cy - c.ay * 1.1)), y1 = min(h - 1, Int(c.cy + c.ay * 1.1))
    for y in y0...y1 {
        for x in x0...x1 {
            let dx = (Float(x) - c.cx) / c.ax
            let dy = (Float(y) - c.cy) / c.ay
            let d = sqrt(dx * dx + dy * dy)
            let a = c.a * (1 - smoothstep(0.65, 1.05, d))
            if a > 0 {
                let i = (y * w + x) * 3
                rgb[i] = 0.98; rgb[i + 1] = 0.99; rgb[i + 2] = 1.0
                alpha[y * w + x] = a
            }
        }
    }
    return RenderedLayer(spec: spec, rgb: rgb, alpha: alpha, lightMul: nil, lightDelta: nil)
}

private func figureLayer(w: Int, h: Int, f: FigureParams, spec: LayerSpec) -> RenderedLayer {
    let n = w * h
    var rgb = [Float](repeating: 0, count: n * 3)
    var alpha = [Float](repeating: 0, count: n)
    let r = f.r
    let bodyCY = f.feetY - 2.1 * r
    let headCY = bodyCY - 2.55 * r
    let hairCY = headCY - 0.22 * r
    let axBody = r, ayBody = 2.2 * r
    let rHead = 0.95 * r
    let axHair = 1.25 * r, ayHair = 1.0 * r
    let half = max(axBody, axHair, rHead)
    let x0 = max(0, Int(f.cx - half - 1)), x1 = min(w - 1, Int(f.cx + half + 1))
    let y0 = max(0, Int(hairCY - ayHair - 1)), y1 = min(h - 1, Int(f.feetY + 1))
    for y in y0...y1 {
        for x in x0...x1 {
            let fx = Float(x), fy = Float(y)
            var color: (Float, Float, Float)? = nil
            let dxB = (fx - f.cx) / axBody, dyB = (fy - bodyCY) / ayBody
            if dxB * dxB + dyB * dyB <= 1 { color = f.cloth }
            if color == nil {
                let dxH = (fx - f.cx) / axHair, dyH = (fy - hairCY) / ayHair
                if dxH * dxH + dyH * dyH <= 1 { color = f.hair }
            }
            let dx = fx - f.cx, dy = fy - headCY
            if dx * dx + dy * dy <= rHead * rHead { color = f.skin }
            if let c = color {
                let i = (y * w + x) * 3
                rgb[i] = c.0; rgb[i + 1] = c.1; rgb[i + 2] = c.2
                alpha[y * w + x] = 1
            }
        }
    }
    return RenderedLayer(spec: spec, rgb: rgb, alpha: alpha, lightMul: nil, lightDelta: nil)
}

private func lightGradientLayer(w: Int, h: Int, p: SceneParams, spec: LayerSpec) -> RenderedLayer {
    let n = w * h
    var mul = [Float](repeating: 1, count: n)
    let diag = sqrt(Float(w * w + h * h))
    for y in 0..<h {
        for x in 0..<w {
            let fx = Float(x), fy = Float(y)
            let dx = fx - p.sunX, dy = fy - p.sunY
            let dNorm = min(1, sqrt(dx * dx + dy * dy) / diag * 1.6)
            var l = 0.58 + 0.42 * (1 - dNorm)
            let vx = fx / Float(max(w - 1, 1)) - 0.5
            let vy = fy / Float(max(h - 1, 1)) - 0.5
            l *= 1 - 0.30 * (vx * vx + vy * vy) * 2
            mul[y * w + x] = clamp01(l)
        }
    }
    return RenderedLayer(spec: spec, rgb: nil, alpha: nil, lightMul: mul, lightDelta: nil)
}

private func shadowLayer(w: Int, h: Int, f: FigureParams, p: SceneParams,
                         spec: LayerSpec) -> RenderedLayer {
    let n = w * h
    var delta = [Float](repeating: 0, count: n)
    let shCY = min(Float(h - 1), f.feetY + 0.12 * f.r)
    let axS = 1.9 * f.r, ayS = 0.38 * f.r
    let x0 = max(0, Int(f.cx - axS - 1)), x1 = min(w - 1, Int(f.cx + axS + 1))
    let y0 = max(p.gy, Int(shCY - ayS - 1)), y1 = min(h - 1, Int(shCY + ayS + 1))
    for y in y0...y1 {
        for x in x0...x1 {
            let dx = (Float(x) - f.cx) / axS, dy = (Float(y) - shCY) / ayS
            let d = sqrt(dx * dx + dy * dy)
            let k = 1 - smoothstep(0.5, 1.0, d)
            if k > 0 { delta[y * w + x] = 0.4 * k }
        }
    }
    return RenderedLayer(spec: spec, rgb: nil, alpha: nil, lightMul: nil, lightDelta: delta)
}

// ---------- Универсальный рендер ----------

/// Собирает сцену по ЛЮБОМУ рецепту. Порядок слоёв = порядок в `recipe.specs`.
/// Слои с `count > 1` становятся отдельными слоями (каждый — свой эксперт).
public func makeComposedScene<R: RandomNumberGenerator>(
    w: Int, h: Int, recipe: LayerRecipe, rng: inout R
) -> ComposedScene {
    precondition(w >= 16 && h >= 16, "сцена слишком мала для слоёв")
    let p = makeSceneParams(w: w, h: h, rng: &rng)
    var layers: [RenderedLayer] = []
    for spec in recipe.specs {
        let k = min(spec.count, 8)   // параметров clouds/figures заготовлено 8
        switch spec.kind {
        case .sky:
            layers.append(skyLayer(w: w, h: h, p: p, spec: spec))
        case .ground:
            layers.append(groundLayer(w: w, h: h, p: p, spec: spec))
        case .sun:
            layers.append(sunLayer(w: w, h: h, p: p, spec: spec))
        case .clouds:
            for i in 0..<k { layers.append(cloudLayer(w: w, h: h, c: p.clouds[i], spec: spec)) }
        case .figure:
            for i in 0..<k { layers.append(figureLayer(w: w, h: h, f: p.figures[i], spec: spec)) }
        case .lightGradient:
            layers.append(lightGradientLayer(w: w, h: h, p: p, spec: spec))
        case .shadow:
            for i in 0..<k {
                layers.append(shadowLayer(w: w, h: h, f: p.figures[i], p: p, spec: spec))
            }
        }
    }
    let color = compositeColorLayers(w: w, h: h, layers: layers)
    let light = composeLightLayers(w: w, h: h, layers: layers)
    var composite = [Float](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) {
        composite[i * 3]     = clamp01(color[i * 3]     * light[i])
        composite[i * 3 + 1] = clamp01(color[i * 3 + 1] * light[i])
        composite[i * 3 + 2] = clamp01(color[i * 3 + 2] * light[i])
    }
    return ComposedScene(w: w, h: h, recipe: recipe, layers: layers,
                         light: light, composite: composite)
}

public func makeComposedScene(w: Int, h: Int, recipe: LayerRecipe, seed: UInt64) -> ComposedScene {
    var rng = SplitMix64(seed: seed)
    return makeComposedScene(w: w, h: h, recipe: recipe, rng: &rng)
}

/// Обратная совместимость: классическая сцена, сразу 8-канальный таргет.
public func makeLayeredScene<R: RandomNumberGenerator>(w: Int, h: Int, rng: inout R) -> LayeredScene {
    makeComposedScene(w: w, h: h, recipe: .classic(), rng: &rng).flattened()
}

public func makeLayeredScene(w: Int, h: Int, seed: UInt64) -> LayeredScene {
    makeComposedScene(w: w, h: h, recipe: .classic(), seed: seed).flattened()
}

// ---------- Предпросмотр ----------

/// Сохраняет превью: <prefix>_composite/_background/_objects/_light.png.
public func saveLayerPreview(scene: LayeredScene, prefix: String) throws {
    let dir = (prefix as NSString).deletingLastPathComponent
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    let n = scene.w * scene.h
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
