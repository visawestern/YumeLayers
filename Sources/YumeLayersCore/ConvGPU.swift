// YumeLayers — свёрточный VAE на Metal: forward + backward + Adam целиком на GPU.
// Раскладка карт: (каналы, H*W), веса свёртки: (oc, ic*kh*kw). im2col/col2im — в шейдерах.
import Foundation

/// Детерминированный ГПСЧ для инициализации весов (splitmix64).
///
/// Раньше веса брались из `Float.random` без seed: два запуска с одним и тем же
/// датасетом давали разные модели, а тесты были невоспроизводимы. Для обучения
/// это ещё и мешает сравнивать изменения гиперпараметров.
/// Один свёрточный слой на GPU.
public final class ConvLayerGPU {
    let ic: Int
    let oc: Int
    let k: Int
    let stride: Int
    let pad: Int

    let W: GPUTensor      // oc × (ic*k*k)
    let b: GPUTensor      // oc × 1
    let mW: GPUTensor
    let vW: GPUTensor
    let mb: GPUTensor
    let vb: GPUTensor

    init(engine: MetalEngine, ic: Int, oc: Int, k: Int = 3, stride: Int = 1, pad: Int = 1,
         scale: Float? = nil, rng: inout SplitMix64) {
        self.ic = ic; self.oc = oc; self.k = k; self.stride = stride; self.pad = pad
        let fanIn = ic * k * k
        // He-инициализация: σ = √(2/fanIn) сохраняет дисперсию сигнала через ReLU.
        // Множитель 0.2 из прежней версии занижал веса в 5 раз — сигнал затухал
        // к латенту на порядки, и часть каналов умирала уже на первом слое.
        let s = scale ?? sqrt(2.0 / Float(fanIn))
        var wdata = [Float](repeating: 0, count: oc * fanIn)
        for i in 0..<wdata.count {
            // равномерное в [-s·√3, s·√3] имеет ту же дисперсию, что N(0, s²)
            let u = Float.random(in: -1...1, using: &rng)
            wdata[i] = u * s * 1.7320508
        }
        self.W = GPUTensor(engine: engine, data: wdata, rows: oc)
        self.b = GPUTensor(engine: engine, data: [Float](repeating: 0, count: oc), rows: oc)
        self.mW = GPUTensor(engine: engine, rows: oc, cols: fanIn)
        self.vW = GPUTensor(engine: engine, rows: oc, cols: fanIn)
        self.mb = GPUTensor(engine: engine, rows: oc, cols: 1)
        self.vb = GPUTensor(engine: engine, rows: oc, cols: 1)
        // Моменты Adam — такие же аккумуляторы, как eps ниже: приватный
        // буфер Metal не гарантирует нулевым, а adam_kernel читает m/v до
        // записи. Обнуляем явно, а не «по счастливой случайности» ОС.
        let s0 = GpuSession(engine)
        s0.begin()
        s0.zeroFill(self.mW)
        s0.zeroFill(self.vW)
        s0.zeroFill(self.mb)
        s0.zeroFill(self.vb)
        s0.end()
    }

    func forward(_ s: GpuSession, _ x: GPUTensor, ih: Int, iw: Int) -> (GPUTensor, ConvCache) {
        let oh = (ih + 2 * pad - k) / stride + 1
        let ow = (iw + 2 * pad - k) / stride + 1
        let cols = s.im2col(x, ih: ih, iw: iw, ic: ic, kh: k, kw: k, stride: stride, pad: pad)
        let pre = s.biasRow(s.matmul(W, cols), b)
        let cache = ConvCache(x: x, cols: cols, ih: ih, iw: iw, oh: oh, ow: ow)
        return (pre, cache)
    }

    func backward(_ s: GpuSession, _ cache: ConvCache, dOut: GPUTensor)
        -> (dx: GPUTensor, gW: GPUTensor, gb: GPUTensor) {
        let gW = s.gradW(dOut, cache.cols, oc: oc, kk: ic * k * k) // oc × (ic*k*k)
        let gb = s.biasRowGrad(dOut)                               // oc × 1
        let dx: GPUTensor
        if stride == 1 {
            dx = s.convBackInputS1(dOut, W, ic: ic, oc: oc,
                                   ih: cache.ih, iw: cache.iw, k: k, pad: pad)
        } else {
            let dcols = s.matmul(W, dOut, transA: true)            // (ic*k*k) × (oh*ow)
            dx = s.col2im(dcols, ic: ic, ih: cache.ih, iw: cache.iw,
                          kh: k, kw: k, stride: stride, pad: pad,
                          oh: cache.oh, ow: cache.ow)
        }
        return (dx, gW, gb)
    }

    func applyAdam(_ s: GpuSession, _ gW: GPUTensor, _ gb: GPUTensor, t: Int, lr: Float) {
        s.adamUpdate(W, gW, mW, vW, t: t, lr: lr)
        s.adamUpdate(b, gb, mb, vb, t: t, lr: lr)
    }

    var matrices: [Matrix] { [W.toMatrix(), b.toMatrix()] }
}

public struct ConvCache {
    let x: GPUTensor
    let cols: GPUTensor
    let ih: Int
    let iw: Int
    let oh: Int
    let ow: Int
}

/// Результат кодирования: активации энкодера и латент.
///
/// Отдельная структура нужна для интерполяции: там энкодер применяется к двум
/// опорным картинкам, а декодер — к произвольному промежуточному латенту.
struct EncoderOutput {
    let c1: ConvCache, p1: GPUTensor, a1: GPUTensor
    let c2: ConvCache, p2: GPUTensor, a2: GPUTensor
    let c3: ConvCache, p3: GPUTensor
    let mu: GPUTensor, logvar: GPUTensor, sigma: GPUTensor, eps: GPUTensor
    let z: GPUTensor
    let resE: ResidualBlockGPU.Cache?
}

/// Результат декодирования: активации декодера и реконструкция.
struct DecoderOutput {
    let u1: GPUTensor, c4: ConvCache, p4: GPUTensor, a4: GPUTensor
    let u2: GPUTensor, c5: ConvCache, p5: GPUTensor, a5: GPUTensor
    let u3: GPUTensor, c6: ConvCache, p6: GPUTensor
    let recon: GPUTensor
    let resD: ResidualBlockGPU.Cache?
}

/// Кэш прямого прохода VAE.
///
/// Латент вариационный: энкодер выдаёт 2·C каналов, они делятся на `mu` и
/// `logvar`, затем сэмплируется `z = mu + exp(0.5·logvar)·eps`.
public struct VAECache {
    let image: GPUTensor
    let c1: ConvCache, p1: GPUTensor, a1: GPUTensor
    let c2: ConvCache, p2: GPUTensor, a2: GPUTensor
    let c3: ConvCache, p3: GPUTensor
    let mu: GPUTensor, logvar: GPUTensor, sigma: GPUTensor, eps: GPUTensor
    let z: GPUTensor
    let u1: GPUTensor, c4: ConvCache, p4: GPUTensor, a4: GPUTensor
    let u2: GPUTensor, c5: ConvCache, p5: GPUTensor, a5: GPUTensor
    let u3: GPUTensor, c6: ConvCache, p6: GPUTensor
    let recon: GPUTensor
    let w: Int, h: Int
    let resE: ResidualBlockGPU.Cache?
    let resD: ResidualBlockGPU.Cache?

    /// Число каналов латента (не 2·C).
    var latentChannels: Int { mu.rows }
    var latentPlane: Int { mu.cols }
}

/// Байт видеопамяти на ОДИН кадр пачки: все активации прямого прохода, кэш
/// im2col и реконструкция. Считается по форме слоёв, а не «на глаз».
///
/// Доминируют im2col-буферы stride-1 слоёв декодера: для d3 это
/// `16·9·hw = 144·hw` float'ов, то есть ~230 МБ на кадр 854×480 только под
/// один слой. Прежняя формула учитывала лишь сам кадр (`3·hw` float'ов,
/// 4.9 МБ) и потому разрешала пачку 16 — это ~8.6 ГБ активаций при 8.18 ГБ
/// видеопамяти, то есть гарантированный OOM.
/// Бюджет памяти, эффективный размер пачки, сообщения об ошибках и потеря VAE
/// вынесены в `ConvVAEHelpers.swift` (см. convVAEBytesPerSample,
/// activationBudgetBytes, effectiveBatchSize, allocationFailureMessage,
/// isIncompatibleCheckpoint, vaeLoss).
/// Свёрточный VAE: `channels` → 16 → 32 → 2·C (mu, logvar) → z (C) → C → 32 → 16 → `channels`.
///
/// Сжатие 8× по каждой оси, латент (h/8, w/8). Веса — 6 слоёв × (W, b) = 12
/// матриц в checkpoint; энкодерный слой e3 вдвое шире остальных, потому что
/// содержит и средние, и логарифмы дисперсий.
///
/// Число каналов — параметр, а не константа:
///   • `channels = 3` — прежний путь reconstruct-RGB (всё обучение в GUI,
///     `convcheck`, `gpuheat` и старые чекпоинты продолжают работать);
///   • `channels = 8` — layered-путь из `LayeredScene`: фон RGB, объекты RGB,
///     маска, свет. Лосс по слоям + по композиции, см. `LayeredVAELoss.swift`.
/// Переход НЕ совместим по весам: e1.W и d3.W меняют форму, поэтому старый
/// 3-канальный чекпоинт честно отклоняется кодом 11 (`isIncompatibleCheckpoint`).
public final class ConvVAEGPU {
    /// Каналов латента (столько же у `mu` и у `logvar`).
    ///
    /// Степень сжатия = 3·hw / (C·hw/64) = 192/C. При C=64 это всего 3× —
    /// то есть «латент» по объёму почти равен исходному кадру, и вариационный
    /// регуляризатор ничего не ограничивает: сеть может просто запомнить кадры.
    /// При C=16 сжатие 12×, что уже сопоставимо с обычными VAE на латенте /8.
    /// Значение вынесено в константу: изменение здесь автоматически меняет
    /// ширину e3 (2·C) и вход d1 (C).
    public static let latentC = 16

    /// Во сколько раз латент меньше исходного RGB-кадра по числу значений.
    /// Оставлено как свойство: GUI, лог `ConvTrainer` и прежние тесты читают
    /// именно базовую конфигурацию (3 канала, C = 16).
    static var compressionRatio: Double { 192.0 / Double(latentC) }

    /// То же для 8-канального layered-входа: 8·hw / (C·hw/64) = 512/C.
    static var layeredCompressionRatio: Double { 512.0 / Double(latentC) }

    /// Сжатие для ПРОИЗВОЛЬНОЙ конфигурации: `channels·64 / C`.
    ///
    /// При 8 каналах и C=16 сжатие 32× — именно здесь был потолок 28.8 дБ:
    /// латент физически слишком узок, чтобы нести все 8 слоёв. C=32 даёт
    /// 16×, C=64 даёт 8×. Формула, а не константа, позволяет осознанно
    /// выбирать ёмкость под доступную VRAM.
    static func compressionRatio(channels: Int = 3,
                                 latentChannels: Int = ConvVAEGPU.latentC) -> Double {
        Double(max(channels, 1)) * 64.0 / Double(max(latentChannels, 1))
    }

    /// Число входных/выходных каналов: 3 (RGB) или 8 (слои).
    let channels: Int

    /// Число каналов латента ЭТОЙ модели. Не статическая константа: два VAE
    /// с разной ёмкостью живут в одном процессе, и `forward` обязан брать C
    /// у своего экземпляра, иначе 32-канальная модель соберёт 16-канальный
    /// латент и упадёт на `split2`.
    public let latentChannels: Int

    /// Кратность ширины свёрточных каналов этой модели (1 — базовая).
    /// Сохраняется в чекпоинт, чтобы реконструкции/переобучение строили
    /// модель той же архитектуры.
    public let width: Int

    let e1: ConvLayerGPU
    let e2: ConvLayerGPU
    let e3: ConvLayerGPU
    let d1: ConvLayerGPU
    let d2: ConvLayerGPU
    let d3: ConvLayerGPU
    /// Архитектура v2: остаточные блоки между e2-e3 и d1-d2.
    let residual: Bool
    let resE: ResidualBlockGPU?
    let resD: ResidualBlockGPU?

    /// - Parameter seed: seed инициализации весов. По умолчанию фиксирован,
    ///   чтобы обучение было воспроизводимым; тесты полагаются на это.
    /// - Parameter width: кратность ширины свёрточных слоёв (1 — базовая
    ///   архитектура; >1 — увеличенная ёмкость для ворот Фазы 1–2).
    init(engine: MetalEngine, channels: Int = 3,
         latentChannels: Int = ConvVAEGPU.latentC,
         width: Int = 1,
         residual: Bool = false,
         seed: UInt64 = 0x5EED_1234_ABCD_0001) {
        self.channels = channels
        // C кратно 8: split2/merge2 и im2col работают блоками, а чекпоинт
        // должен однозначно описывать форму. Некорректный ввод не тихо
        // округляется, а явно отвергается.
        precondition(latentChannels >= 1, "число каналов латента должно быть > 0")
        self.latentChannels = latentChannels
        let C = latentChannels
        let W = max(width, 1)
        self.width = W
        let f1 = 16 * W
        let f2 = 32 * W
        var rng = SplitMix64(seed: seed)
        e1 = ConvLayerGPU(engine: engine, ic: channels, oc: f1,
                          k: 3, stride: 2, pad: 1, rng: &rng)
        e2 = ConvLayerGPU(engine: engine, ic: f1, oc: f2, k: 3, stride: 2, pad: 1, rng: &rng)
        e3 = ConvLayerGPU(engine: engine, ic: f2, oc: 2 * C, k: 3, stride: 2, pad: 1, rng: &rng)
        d1 = ConvLayerGPU(engine: engine, ic: C,  oc: f2, k: 3, stride: 1, pad: 1, rng: &rng)
        d2 = ConvLayerGPU(engine: engine, ic: f2, oc: f1, k: 3, stride: 1, pad: 1, rng: &rng)
        // Выходной слой — намеренно мелкий: p6 должен стартовать около нуля,
        // иначе сигмоида насыщается и градиент к декодеру исчезает.
        // fanIn = ic·k·k = f1·9 и НЕ зависит от oc, так что масштаб один
        // и для 3-, и для 8-канальной модели.
        let outScale = 0.1 / sqrt(Float(f1 * 3 * 3))
        d3 = ConvLayerGPU(engine: engine, ic: f1, oc: channels,
                          k: 3, stride: 1, pad: 1, scale: outScale, rng: &rng)
        self.residual = residual
        if residual {
            self.resE = ResidualBlockGPU(engine: engine, channels: f2, rng: &rng)
            self.resD = ResidualBlockGPU(engine: engine, channels: f2, rng: &rng)
        } else {
            self.resE = nil
            self.resD = nil
        }
    }

    var layers: [ConvLayerGPU] {
        var l: [ConvLayerGPU] = [e1, e2]
        if let re = resE { l.append(re.c1); l.append(re.c2) }
        l.append(e3)
        l.append(d1)
        if let rd = resD { l.append(rd.c1); l.append(rd.c2) }
        l.append(d2)
        l.append(d3)
        return l
    }

    /// Прямой проход.
    ///
    /// - Parameters:
    ///   - deterministic: `true` → ε = 0, то есть `z = mu`. Используется для
    ///     валидации, реконструкций и численной проверки градиентов, чтобы
    ///     шум не менялся между прогонами.
    ///   - seed: seed гауссова шума; одинаковый seed → одинаковый `z`.
    func forward(_ s: GpuSession, image: GPUTensor, w: Int, h: Int,
                 deterministic: Bool = false, seed: UInt32 = 0) -> VAECache {
        let e = encode(s, image: image, w: w, h: h,
                       deterministic: deterministic, seed: seed)
        let d = decode(s, z: e.z, w: w, h: h)
        return VAECache(image: image,
                        c1: e.c1, p1: e.p1, a1: e.a1,
                        c2: e.c2, p2: e.p2, a2: e.a2,
                        c3: e.c3, p3: e.p3,
                        mu: e.mu, logvar: e.logvar, sigma: e.sigma, eps: e.eps, z: e.z,
                        u1: d.u1, c4: d.c4, p4: d.p4, a4: d.a4,
                        u2: d.u2, c5: d.c5, p5: d.p5, a5: d.a5,
                        u3: d.u3, c6: d.c6, p6: d.p6,
                        recon: d.recon, w: w, h: h,
                        resE: e.resE, resD: d.resD)
    }

    /// Кодирование кадра в латент (без декодирования).
    ///
    /// Выделено из `forward`, потому что интерполяция между двумя картинками
    /// устроена именно так: кодируем A и B отдельно, смешиваем их латенты
    /// и декодируем результат. Математика та же, что и в `forward`.
    func encode(_ s: GpuSession, image: GPUTensor, w: Int, h: Int,
                deterministic: Bool = false, seed: UInt32 = 0) -> EncoderOutput {
        precondition(w % 8 == 0 && h % 8 == 0, "ширина и высота должны делиться на 8")
        let h1 = h / 2, w1 = w / 2
        let h2 = h / 4, w2 = w / 4
        let h3 = h / 8, w3 = w / 8
        let C = latentChannels
        let plane = h3 * w3

        let (p1, c1) = e1.forward(s, image, ih: h, iw: w)
        let a1 = s.relu(p1)
        let (p2, c2) = e2.forward(s, a1, ih: h1, iw: w1)
        var a2 = s.relu(p2)
        var resECache: ResidualBlockGPU.Cache? = nil
        if let rb = resE {
            let (rbOut, rc) = rb.forward(s, a2, ih: h2, iw: w2)
            resECache = rc
            a2 = rbOut
        }
        let (p3, c3) = e3.forward(s, a2, ih: h2, iw: w2)

        // Вариационный латент: mu, logvar → σ → z = mu + σ·ε
        let (mu, logvar) = s.split2(p3, c: C, plane: plane)
        let sigma = s.expHalf(logvar)
        let eps: GPUTensor
        if deterministic {
            // ε = 0 без отдельного ядра: нулевой тензор в текущем command buffer.
            eps = GPUTensor(engine: s.engine, rows: C, cols: plane)
            s.zeroFill(eps)
        } else {
            eps = s.randn(seed: seed, rows: C, cols: plane)
        }
        let z = s.fma(mu, sigma, eps)
        return EncoderOutput(c1: c1, p1: p1, a1: a1,
                             c2: c2, p2: p2, a2: a2,
                             c3: c3, p3: p3,
                             mu: mu, logvar: logvar, sigma: sigma, eps: eps, z: z,
                             resE: resECache)
    }

    /// Декодирование латента в картинку (без кодирования).
    ///
    /// Принимает любой латент размером C × (h/8 × w/8) — в том числе
    /// промежуточный, полученный смешиванием латентов A и B.
    func decode(_ s: GpuSession, z: GPUTensor, w: Int, h: Int) -> DecoderOutput {
        precondition(w % 8 == 0 && h % 8 == 0, "ширина и высота должны делиться на 8")
        let h1 = h / 2, w1 = w / 2
        let h2 = h / 4, w2 = w / 4
        let h3 = h / 8, w3 = w / 8

        let u1 = s.upsample2x(z, h: h3, w: w3)
        let (p4, c4) = d1.forward(s, u1, ih: h2, iw: w2)
        var a4 = s.relu(p4)
        var resDCache: ResidualBlockGPU.Cache? = nil
        if let rb = resD {
            let (rbOut, rc) = rb.forward(s, a4, ih: h2, iw: w2)
            resDCache = rc
            a4 = rbOut
        }
        let u2 = s.upsample2x(a4, h: h2, w: w2)
        let (p5, c5) = d2.forward(s, u2, ih: h1, iw: w1)
        let a5 = s.relu(p5)
        let u3 = s.upsample2x(a5, h: h1, w: w1)
        let (p6, c6) = d3.forward(s, u3, ih: h, iw: w)
        let recon = s.sigmoid(p6)
        return DecoderOutput(u1: u1, c4: c4, p4: p4, a4: a4,
                             u2: u2, c5: c5, p5: p5, a5: a5,
                             u3: u3, c6: c6, p6: p6,
                             recon: recon, resD: resDCache)
    }

    /// Возвращает (градиенты входов, градиенты слоёв в порядке layers).
    ///
    /// Потеря: `MSE(recon, target) + β·KL`, где KL усреднён по элементам
    /// латента. Градиент по `z` идёт через перепараметризацию, по `mu`/`logvar`
    /// — плюс вклад KL-регуляризатора.
    func backward(_ s: GpuSession, _ cache: VAECache, target: GPUTensor,
                  beta: Float = 1.0,
                  dReconOverride: GPUTensor? = nil) -> ([GPUTensor], [GPUTensor]) {
        let w = cache.w, h = cache.h
        let n = Float(channels * w * h)
        // dL/d(recon) = 2(recon-target)/N, затем через сигмоиду.
        // `dReconOverride` подставляет градиент layered-лосса (слои + сборка),
        // не ломая ни один из прежних вызовов backward.
        let dRecon = dReconOverride ?? s.smul(2.0 / n, s.sub(cache.recon, target))
        let dP6 = s.sigmoidGrad(cache.p6, dRecon)

        let (dx6, gW6, gb6) = d3.backward(s, cache.c6, dOut: dP6)
        let dA5 = s.downsample2x(dx6, h: h / 2, w: w / 2)
        let dP5 = s.reluGrad(cache.a5, dA5)
        let (dx5, gW5, gb5) = d2.backward(s, cache.c5, dOut: dP5)
        var dA4 = s.downsample2x(dx5, h: h / 4, w: w / 4)
        var resDGrads: [GPUTensor] = []
        if let rb = resD, let rc = cache.resD {
            let (dPrev, gr) = rb.backward(s, rc, dOut: dA4)
            dA4 = dPrev
            resDGrads = gr
        }
        let dP4 = s.reluGrad(cache.a4, dA4)
        let (dx4, gW4, gb4) = d1.backward(s, cache.c4, dOut: dP4)

        // ---- Вариационный узел ----
        let dz = s.downsample2x(dx4, h: h / 8, w: w / 8)
        let M = Float(max(cache.mu.n, 1))
        let (dmu, dlogvar) = s.vaeGrad(dz: dz, mu: cache.mu, logvar: cache.logvar,
                                       eps: cache.eps,
                                       betaN: beta / M, betaHalfN: 0.5 * beta / M)
        let dP3 = s.merge2(dmu, dlogvar, c: cache.latentChannels, plane: cache.latentPlane)

        let (dx3, gW3, gb3) = e3.backward(s, cache.c3, dOut: dP3)
        var dBeforeResE = dx3
        var resEGrads: [GPUTensor] = []
        if let rb = resE, let rc = cache.resE {
            let (dPrev, gr) = rb.backward(s, rc, dOut: dx3)
            dBeforeResE = dPrev
            resEGrads = gr
        }
        let dP2 = s.reluGrad(cache.a2, dBeforeResE)
        let (dx2, gW2, gb2) = e2.backward(s, cache.c2, dOut: dP2)
        let dP1 = s.reluGrad(cache.a1, dx2)
        let (dx1, gW1, gb1) = e1.backward(s, cache.c1, dOut: dP1)

        var gAll: [GPUTensor] = [gW1, gb1, gW2, gb2]
        gAll += resEGrads
        gAll += [gW3, gb3, gW4, gb4]
        gAll += resDGrads
        gAll += [gW5, gb5, gW6, gb6]
        return ([dx1, dx2, dx3, dx4, dx5, dx6], gAll)
    }

    /// Обучение на 8-канальном layered-таргете.
    ///
    /// Отличается от обычного `backward` ТОЛЬКО градиентом по реконструкции:
    /// вместо `MSE(recon, target)` считается
    /// `MSE(8 слоёв) + MSE(собранный RGB, кадр)`. Всё остальное — сигмоида,
    /// свёртки, перепараметризация и KL — тот же самый граф, так что
    /// численная проверка градиентов проходит через оба пути одинаково.
    func backwardLayered(_ s: GpuSession, _ cache: VAECache,
                         target: LayeredTargetGPU,
                         layerWeight: Float = 1,
                         composeWeight: Float = 1,
                         beta: Float = 1) -> ([GPUTensor], [GPUTensor]) {
        precondition(channels == LayeredScene.layerChannels,
                     "backwardLayered: нужна 8-канальная модель, а не \(channels)")
        let dRecon = s.layeredGrad(recon: cache.recon, target: target,
                                   layerWeight: layerWeight,
                                   composeWeight: composeWeight)
        return backward(s, cache, target: target.channels, beta: beta,
                        dReconOverride: dRecon)
    }

    /// KL(N(mu, σ²)‖N(0,1)), усреднённая по элементам латента (Double для логов).
    static func klMean(_ cache: VAECache) -> Double {
        let mu = cache.mu.toArray()
        let lv = cache.logvar.toArray()
        guard !mu.isEmpty else { return 0 }
        var sum = 0.0
        for i in 0..<mu.count {
            let l = Double(min(max(lv[i], -10), 10))
            sum += 0.5 * (Double(mu[i]) * Double(mu[i]) + exp(l) - 1 - l)
        }
        return sum / Double(mu.count)
    }

    func applyAdam(_ s: GpuSession, grads: [GPUTensor], t: Int, lr: Float) {
        let ls = layers
        for (i, layer) in ls.enumerated() {
            layer.applyAdam(s, grads[i * 2], grads[i * 2 + 1], t: t, lr: lr)
        }
    }

    var matrices: [Matrix] {
        var out: [Matrix] = []
        for l in layers { out.append(l.W.toMatrix()); out.append(l.b.toMatrix()) }
        return out
    }

    /// Моменты Adam в порядке: для каждого слоя — `mW`, `vW`, `mb`, `vb`.
    ///
    /// Порядок совпадает с порядком весов (`W0, b0, W1, b1, …`): для веса с
    /// номером `p` моменты лежат под номерами `2p` (m) и `2p+1` (v).
    func optimizerMoments() -> [Matrix] {
        var out: [Matrix] = []
        for l in layers {
            out.append(l.mW.toMatrix())
            out.append(l.vW.toMatrix())
            out.append(l.mb.toMatrix())
            out.append(l.vb.toMatrix())
        }
        return out
    }

    /// Загрузка моментов Adam (порядок: `mW, vW, mb, vb` на слой).
    ///
    /// Без этого продолжение обучения стартует с нулевых моментов, и первая
    /// коррекция смещения Adam сносит модель (шаг ±lr по каждому весу).
    func loadOptimizerMoments(_ moments: [Matrix]) throws {
        let needed = layers.count * 4
        guard moments.count == needed else {
            throw NSError(domain: "Yume", code: 13,
                          userInfo: [NSLocalizedDescriptionKey:
                            "состояние оптимизатора: нужно \(needed) моментов, найдено \(moments.count)"])
        }
        for (i, layer) in layers.enumerated() {
            let mW = moments[i * 4], vW = moments[i * 4 + 1]
            let mb = moments[i * 4 + 2], vb = moments[i * 4 + 3]
            guard mW.rows == layer.W.rows, mW.cols == layer.W.cols,
                  vW.rows == layer.W.rows, vW.cols == layer.W.cols,
                  mb.rows == layer.b.rows, mb.cols == layer.b.cols,
                  vb.rows == layer.b.rows, vb.cols == layer.b.cols else {
                throw NSError(domain: "Yume", code: 13,
                              userInfo: [NSLocalizedDescriptionKey:
                                "состояние оптимизатора: несовпадение формы слоя \(i)"])
            }
            layer.mW.copyFrom(GPUTensor(engine: layer.mW.engine, data: mW.data, rows: mW.rows))
            layer.vW.copyFrom(GPUTensor(engine: layer.vW.engine, data: vW.data, rows: vW.rows))
            layer.mb.copyFrom(GPUTensor(engine: layer.mb.engine, data: mb.data, rows: mb.rows))
            layer.vb.copyFrom(GPUTensor(engine: layer.vb.engine, data: vb.data, rows: vb.rows))
        }
    }

    /// Загрузка весов из checkpoint. Формат: W0,b0,W1,b1,...
    ///
    /// Два прохода: сначала валидируются формы ВСЕХ слоёв, и только потом
    /// копируются веса. Иначе при несовместимом чекпоинте часть слоёв
    /// успевала перезаписаться до первой ошибки — модель оставалась в
    /// невалидной смеси старых и случайных весов под лог «старт с новых».
    func load(from mats: [Matrix]) throws {
        guard mats.count >= layers.count * 2 else {
            throw NSError(domain: "Yume", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "checkpoint: нужно \(layers.count * 2) матриц, найдено \(mats.count)"])
        }
        for (i, layer) in layers.enumerated() {
            let wm = mats[i * 2], bm = mats[i * 2 + 1]
            guard wm.rows == layer.oc, wm.cols == layer.ic * layer.k * layer.k,
                  bm.data.count == layer.oc else {
                throw NSError(domain: "Yume", code: 11,
                              userInfo: [NSLocalizedDescriptionKey: "checkpoint: несовпадение формы слоя \(i)"])
            }
        }
        for (i, layer) in layers.enumerated() {
            let wm = mats[i * 2], bm = mats[i * 2 + 1]
            layer.W.copyFrom(GPUTensor(engine: layer.W.engine, data: wm.data, rows: wm.rows))
            layer.b.copyFrom(GPUTensor(engine: layer.b.engine, data: bm.data, rows: bm.rows))
        }
    }
}
