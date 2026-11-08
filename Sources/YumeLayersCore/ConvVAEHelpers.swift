// YumeLayers — чистая вспомогательная математика сёрточного VAE.
//
// Выделено из `ConvGPU.swift`, чтобы держать файлы в пределе 600 строк и
// функции атомарными (см. scripts/lint-codeguide.py и testCodeGuideLimits).
// Эти функции — чистые (без Metal/UI): их можно покрывать юнит-тестами.
import Foundation

/// Детерминированный ГПСЧ с перемешиванием. Единый для обучения, чтобы
/// воспроизводить прогоны (SplitMix64, тройной xor-shift).
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

/// Seed шума VAE для шага обучения.
///
/// Шум ОБЯЗАН меняться от шага к шагу. С фиксированным seed (как было:
/// `forward` без `deterministic` подставлял `randn(seed: 0)`) модель
/// превращается в автоэнкодер с постоянным смещением: декодер выучивает
/// ровно один вектор шума и не умеет работать с остальными. Тогда же
/// латент перестаёт быть латентом VAE — оценка через ε = 0 (так работает
/// интерполяция) даёт совсем другое качество, чем показывает обучение.
///
/// Смешивание сделано на 64 битах, чтобы seed зависел и от шага, и от
/// позиции кадра в пачке, но оставался воспроизводимым.
public func trainingNoiseSeed(step: Int, batchIndex: Int) -> UInt32 {
    let a = UInt64(bitPattern: Int64(step)) &* 0x9E37_79B9_7F4A_7C15
    let b = UInt64(bitPattern: Int64(batchIndex)) &* 0xC2B2_AE3D_27D4_EB4F
    return UInt32(truncatingIfNeeded: (a ^ (b &+ 0x1656_67B1_9E37_79F9)) >> 32)
}

/// Множитель скорости обучения на первых шагах после продолжения с
/// чекпоинта БЕЗ состояния оптимизатора.
///
/// Моменты Adam при таком продолжении нулевые, поэтому первый шаг равен
/// ±lr по каждому весу — для обученной модели это разрушительно. Мягкий
/// разгон (warmup) ограничивает величину первых шагов.
public func resumeWarmupFactor(step: Int, warmupSteps: Int) -> Float {
    guard warmupSteps > 0 else { return 1 }
    let s = max(step, 1)
    return Float(min(1.0, Double(s) / Double(warmupSteps)))
}

/// Байты активаций на один кадр пачки (с учётом im2col-буферов).
///
/// Доминируют im2col-буферы stride-1 слоёв декодера: для d3 это
/// `16·9·hw = 144·hw` float'ов, то есть ~230 МБ на кадр 854×480 только под
/// один слой. Прежняя формула учитывала лишь сам кадр (`3·hw` float'ов,
/// 4.9 МБ) и потому разрешала пачку 16 — это ~8.6 ГБ активаций при 8.18 ГБ
/// видеопамяти, то есть гарантированный OOM.
public func convVAEBytesPerSample(w: Int, h: Int, channels: Int = 3,
                                  layered: Bool = false,
                                  latentChannels: Int = ConvVAEGPU.latentC) -> Int {
    let hw = w * h
    // Ёмкость латента теперь параметр. При C=32 буферы mu/logvar/sigma/eps/z,
    // u1 и im2col декодера d1 вдвое больше — бюджет обязан это видеть, иначе
    // effectiveBatchSize снова разрешит невлезающую пачку (история §C2 AUDIT).
    let C = max(latentChannels, 1)
    let ic = max(channels, 1)
    var f = 0
    // e1: channels → 16, stride 2
    f += 16 * hw / 4 * 2            // p1, a1
    f += (ic * 9) * hw / 4          // cols1 (im2col растёт с числом входных каналов)
    // e2: 16 → 32, stride 2
    f += 32 * hw / 16 * 2           // p2, a2
    f += (16 * 9) * hw / 16         // cols2
    // e3: 32 → 2C, stride 2
    f += (2 * C) * hw / 64          // p3
    f += 5 * C * hw / 64            // mu, logvar, sigma, eps, z
    // d1: C → 32, stride 1
    f += C * hw / 16                // u1
    f += (C * 9) * hw / 16          // cols4
    f += 32 * hw / 16 * 2           // p4, a4
    // d2: 32 → 16, stride 1
    f += 32 * hw / 4                // u2
    f += (32 * 9) * hw / 4          // cols5
    f += 16 * hw / 4 * 2            // p5, a5
    // d3: 16 → channels, stride 1
    f += 16 * hw                    // u3
    f += (16 * 9) * hw              // cols6
    f += ic * hw * 2                // p6, recon
    // layered: таргет 8n + градиент 8n + композит 3n поверх активаций
    return f * 4 + (layered ? layeredActivationBytes(w: w, h: h) : 0)
}

/// Доля рабочего набора видеопамяти под активации обучения.
///
/// Бюджет делится явно: 60% — пул кадров (`gpuFramePoolCapacity`), 30% —
/// активации, 10% — запас на фрагментацию и служебные буферы. Раньше два
/// бюджета считались независимо (60% и «сколько влезет»), и вместе они
/// превышали рабочий набор: на 8.18 ГБ пул занимал 4.9 ГБ, а пачка из 16
/// кадров требовала ещё 8.6 ГБ — гарантированный OOM.
public let activationBudgetFraction = 0.30

/// Байт под активации одного шага обучения.
public func activationBudgetBytes(workingSet: UInt64) -> Int {
    Int(Double(workingSet) * activationBudgetFraction)
}

/// Сколько кадров в пачке влезает в `budgetBytes` с учётом активаций.
/// Ровно один кадр разрешён всегда: иначе обучение невозможно в принципе,
/// и лучше упасть на явной проверке размера, чем молча не работать.
public func effectiveBatchSize(requested: Int, w: Int, h: Int,
                               budgetBytes: Int,
                               channels: Int = 3,
                               layered: Bool = false,
                               latentChannels: Int = ConvVAEGPU.latentC) -> Int {
    let per = max(convVAEBytesPerSample(w: w, h: h, channels: channels,
                                        layered: layered,
                                        latentChannels: latentChannels), 1)
    let fits = max(1, budgetBytes / per)
    return max(1, min(max(requested, 1), fits))
}

/// Сообщение при нехватке видеопамяти.
///
/// Metal возвращает `nil` из `makeBuffer`, если памяти не хватило; без этого
/// текста пользователь видел только падение на разыменовании `nil` и не понимал,
/// что уменьшать. Формат проверяется тестом.
public func allocationFailureMessage(bytes: Int, availableBytes: Int, what: String) -> String {
    let mb = max(bytes, 0) / 1_048_576
    let av = max(availableBytes, 0) / 1_048_576
    return "не хватило видеопамяти: нужно \(mb) МБ под \(what), свободно ~\(av) МБ. "
         + "Уменьшите разрешение кадра или размер пачки."
}

/// Ошибка «чекпоинт от другой архитектуры» — не битый файл, а модель других
/// размеров (например, латент до перехода на вариационный путь: e3 был 64
/// канала вместо 2·C). Такую ситуацию можно пережить, начав обучение с новых
/// весов, поэтому её надо отличать от настоящих ошибок чтения.
public func isIncompatibleCheckpoint(_ error: Error) -> Bool {
    let e = error as NSError
    // 10 — неверное число матриц, 11 — несовпадение формы слоя (см. ConvVAEGPU.load).
    return e.domain == "Yume" && (e.code == 10 || e.code == 11)
}

/// Полная потеря VAE: `MSE(recon, target) + β·KL`. Считается по CPU-копиям.
/// Должна совпадать с формулой в `ConvVAEGPU.backward`, иначе численная
/// проверка градиентов показывает расхождение там, где его нет.
public func vaeLoss(cache: VAECache, target: [Float], beta: Float = 1.0) -> Double {
    let r = cache.recon.toArray()
    let n = min(r.count, target.count)
    guard n > 0 else { return .infinity }
    var mse = 0.0
    for i in 0..<n {
        let d = Double(r[i] - target[i])
        mse += d * d
    }
    mse /= Double(n)
    return mse + Double(beta) * ConvVAEGPU.klMean(cache)
}
