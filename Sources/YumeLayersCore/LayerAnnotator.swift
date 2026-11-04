// YumeLayers — разметка РЕАЛЬНЫХ кадров на 8 каналов ground-truth.
//
// Главный риск итераций 4–5: layered-VAE учился только на процедурной
// синтетике. Здесь он закрывается детерминированной факторизацией видеокадра:
//
//   фон      = временная медиана по стеку кадров (статичная часть сцены)
//   маска    = бинарная разница |кадр − фон| с порогом и дилатацией
//   свет     = низкочастотная карта освещённости, ограниченная снизу
//              максимальным каналом кадра (чтобы деление не клиппилось)
//   объекты  = (кадр / свет) внутри маски, нули вне её
//   фон/слой = (кадр / свет) вне маски, медиана внутри
//
// Ключевое свойство: composeLayers(...) == исходный кадр ТОЧНО (до ошибки
// Float). Это не «похожая картинка» и не приближение: таргет собирается
// обратно, поэтому loss композиции имеет смысл на настоящем видео, а
// QualityFloor сравнивает реконструкцию именно с реальным кадром.
//
// Почему свет ограничен снизу max(каналом): формула сборки умножает на свет,
// а сигмоида декодера держит слои в [0,1]. Если light < max(rgb), то
// rgb/light > 1 — и точная сборка невозможна в принципе. Поэтому берём
// max(сглаженная освещённость, max(rgb)): карта остаётся низкочастотной,
// но ни один пиксель не выходит за диапазон.
//
// Чистый Swift: Foundation + массивы Float. Никакого Python, никаких
// ML-фреймворков, никаких внешних моделей сегментации.
import Foundation

/// Параметры классической разметки. Все значения ограничены sane-диапазоном,
/// чтобы некорректный ввод не превратился в пустую или сплошную маску.
public struct LayerAnnotationParams {
    /// Порог |кадр − фон| по максимальному каналу, выше которого пиксель
    /// считается объектом.
    public var diffThreshold: Float
    /// Нижняя граница карты освещённости.
    public var minLight: Float
    /// Радиус бокс-размытия карты освещённости.
    public var blurRadius: Int
    /// Радиус дилатации бинарной маски (закрывает дыры внутри объектов).
    public var dilateRadius: Int

    public init(diffThreshold: Float = 0.08, minLight: Float = 0.70,
                blurRadius: Int = 6, dilateRadius: Int = 1) {
        self.diffThreshold = min(max(diffThreshold, 0.001), 1)
        self.minLight = min(max(minLight, 0.1), 1)
        self.blurRadius = max(0, blurRadius)
        self.dilateRadius = max(0, dilateRadius)
    }
}

/// Сепарабельное бокс-размытие одноканальной плоскости w×h с clamp по краю.
/// Детерминировано и не требует vDSP: для кадров доказательства концепции
/// (96×72…480p) прямое суммирование дешевле накладных расходов на BLAS.
public func boxBlur(_ x: [Float], w: Int, h: Int, radius: Int) -> [Float] {
    let n = w * h
    precondition(x.count >= n, "boxBlur: плоскость меньше w·h")
    let r = max(0, radius)
    if r == 0 { return Array(x.prefix(n)) }
    var tmp = [Float](repeating: 0, count: n)
    var out = [Float](repeating: 0, count: n)
    for y in 0..<h {
        let row = y * w
        for xx in 0..<w {
            var sum: Float = 0
            for dx in -r...r {
                let sx = min(w - 1, max(0, xx + dx))
                sum += x[row + sx]
            }
            tmp[row + xx] = sum / Float(2 * r + 1)
        }
    }
    for yy in 0..<h {
        for xx in 0..<w {
            var sum: Float = 0
            for dy in -r...r {
                let sy = min(h - 1, max(0, yy + dy))
                sum += tmp[sy * w + xx]
            }
            out[yy * w + xx] = sum / Float(2 * r + 1)
        }
    }
    return out
}

/// Попиксельная временная медиана по стеку RGB-кадров (HWC).
/// Для аниме это рабочая оценка статичного фона: движущийся персонаж
/// занимает конкретный пиксель лишь в меньшинстве кадров и медианой отсекается.
/// Ограничение честно зафиксировано: если объект закрывает пиксель более чем
/// в половине стека, медиана становится объектом — поэтому стек должен быть
/// достаточно длинным и/или с разреженным движением.
public func temporalMedianBackground(_ frames: [[Float]], w: Int, h: Int) -> [Float] {
    let n = w * h
    precondition(!frames.isEmpty, "temporalMedianBackground: пустой стек кадров")
    let count = frames.count
    var bg = [Float](repeating: 0, count: n * 3)
    var values = [Float](repeating: 0, count: count)
    for f in frames {
        precondition(f.count >= n * 3, "temporalMedianBackground: кадр меньше w·h·3")
    }
    for i in 0..<n {
        for c in 0..<3 {
            for f in 0..<count { values[f] = frames[f][i * 3 + c] }
            values.sort()
            if count % 2 == 1 {
                bg[i * 3 + c] = values[count / 2]
            } else {
                bg[i * 3 + c] = 0.5 * (values[count / 2 - 1] + values[count / 2])
            }
        }
    }
    return bg
}

/// Бинарная маска объектов по разнице с фоном + дилатацией.
/// Возвращает ровно {0, 1}: бинарность важна — формула сборки тогда
/// вырождается в точный выбор между фоном и объектами без интерполяции.
public func differenceMask(frame: [Float], background: [Float], w: Int, h: Int,
                           threshold: Float, dilate: Int) -> [Float] {
    let n = w * h
    precondition(frame.count >= n * 3 && background.count >= n * 3,
                 "differenceMask: неверный размер кадра")
    var raw = [Float](repeating: 0, count: n)
    for i in 0..<n {
        let d0 = abs(frame[i * 3] - background[i * 3])
        let d1 = abs(frame[i * 3 + 1] - background[i * 3 + 1])
        let d2 = abs(frame[i * 3 + 2] - background[i * 3 + 2])
        raw[i] = max(d0, max(d1, d2)) >= threshold ? 1 : 0
    }
    let r = max(0, dilate)
    if r == 0 { return raw }
    var out = [Float](repeating: 0, count: n)
    for y in 0..<h {
        for x in 0..<w {
            var hit = false
            for dy in -r...r {
                let sy = min(h - 1, max(0, y + dy))
                for dx in -r...r {
                    let sx = min(w - 1, max(0, x + dx))
                    if raw[sy * w + sx] == 1 { hit = true; break }
                }
                if hit { break }
            }
            out[y * w + x] = hit ? 1 : 0
        }
    }
    return out
}

/// Карта освещённости в [minLight, 1], при этом light[i] >= max(rgb[i]).
/// Второе условие — обязательное: оно гарантирует точную обратную сборку.
public func illuminationMap(frame: [Float], w: Int, h: Int,
                            minLight: Float, blurRadius: Int) -> [Float] {
    let n = w * h
    precondition(frame.count >= n * 3, "illuminationMap: неверный размер кадра")
    let lo = min(max(minLight, 0.1), 1)
    var mx = [Float](repeating: 0, count: n)
    for i in 0..<n {
        mx[i] = max(frame[i * 3], max(frame[i * 3 + 1], frame[i * 3 + 2]))
    }
    let blur = boxBlur(mx, w: w, h: h, radius: blurRadius)
    var light = [Float](repeating: 0, count: n)
    for i in 0..<n {
        let smooth = lo + (1 - lo) * min(1, max(0, blur[i]))
        let v = max(smooth, mx[i])
        light[i] = min(1, max(lo, v))
    }
    return light
}

/// Разметить стек реальных RGB-кадров в стек `LayeredScene` (8 каналов).
///
/// Фон оценивается ОДИН раз по всему стеку — так маска каждого кадра
/// действительно описывает движение, а не локальный шум. Каждая сцена
/// собирается обратно в исходный кадр формулой `composeLayers`.
public func annotateFrames(_ frames: [[Float]], w: Int, h: Int,
                           params: LayerAnnotationParams = LayerAnnotationParams())
    -> [LayeredScene] {
    let n = w * h
    precondition(!frames.isEmpty, "annotateFrames: пустой стек кадров")
    precondition(w > 0 && h > 0, "annotateFrames: неверный размер")
    let bgMed = temporalMedianBackground(frames, w: w, h: h)

    var scenes: [LayeredScene] = []
    scenes.reserveCapacity(frames.count)
    for frame in frames {
        let light = illuminationMap(frame: frame, w: w, h: h,
                                    minLight: params.minLight,
                                    blurRadius: params.blurRadius)
        let mask = differenceMask(frame: frame, background: bgMed, w: w, h: h,
                                  threshold: params.diffThreshold,
                                  dilate: params.dilateRadius)
        var bg = [Float](repeating: 0, count: n * 3)
        var obj = [Float](repeating: 0, count: n * 3)
        for i in 0..<n {
            let l = max(light[i], 1e-6)
            let r = min(1, max(0, frame[i * 3] / l))
            let g = min(1, max(0, frame[i * 3 + 1] / l))
            let b = min(1, max(0, frame[i * 3 + 2] / l))
            let mr = min(1, max(0, bgMed[i * 3] / l))
            let mg = min(1, max(0, bgMed[i * 3 + 1] / l))
            let mb = min(1, max(0, bgMed[i * 3 + 2] / l))
            if mask[i] == 1 {
                bg[i * 3] = mr; bg[i * 3 + 1] = mg; bg[i * 3 + 2] = mb
                obj[i * 3] = r; obj[i * 3 + 1] = g; obj[i * 3 + 2] = b
            } else {
                bg[i * 3] = r; bg[i * 3 + 1] = g; bg[i * 3 + 2] = b
            }
        }
        let composite = composeLayers(w: w, h: h, background: bg, objects: obj,
                                      mask: mask, light: light)
        scenes.append(LayeredScene(w: w, h: h, background: bg, objects: obj,
                                   mask: mask, light: light, composite: composite))
    }
    return scenes
}

/// Доля пикселей, отнесённых к объектам, — честная метрика разметки:
/// сплошная маска означает, что фон не выделился (видео без статики).
public func objectCoverage(_ scene: LayeredScene) -> Double {
    guard !scene.mask.isEmpty else { return 0 }
    let hits = scene.mask.reduce(0) { $0 + ($1 == 1 ? 1 : 0) }
    return Double(hits) / Double(scene.mask.count)
}
