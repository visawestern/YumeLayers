// YumeLayers — хелперы E2E-тестов сети движения (вынесены из TestGroup22.swift
// для соблюдения лимита 120 строк на функцию).
import Foundation
import Metal

/// MSE между двумя латентами (поэлементно, нормировано на размерность).
func motionMSE(_ a: [Float], _ b: [Float]) -> Double {
    var total = 0.0
    for i in 0..<min(a.count, b.count) {
        let d = Double(a[i] - b[i])
        total += d * d
    }
    return total / Double(max(a.count, 1))
}

/// Сдвиг пары на базовый индекс (окна внутри конкретного сегмента).
func motionBuildPairs(_ base: Int, _ size: Int, _ window: Int, _ stride: Int) -> [MotionPair] {
    var out: [MotionPair] = []
    for p in motionPairs(frameCount: size, window: window, stride: stride) {
        out.append(MotionPair(a: p.a + base, b: p.b + base, k: p.k + base))
    }
    return out
}

/// Нормализация латентов по диапазону индексов (as в trainMotionMLP).
func motionNormalizeNet(_ net: MotionMLPGPU, latents: [Int: [Float]],
                        range: Range<Int>) {
    var sum = 0.0
    var sumSq = 0.0
    var nElem = 0
    for i in range {
        if let z = latents[i] {
            for v in z {
                sum += Double(v)
                sumSq += Double(v) * Double(v)
                nElem += 1
            }
        }
    }
    guard nElem > 0 else { return }
    let mean = Float(sum / Double(nElem))
    let variance = max((sumSq / Double(nElem)) - Double(mean) * Double(mean), 1e-6)
    net.latentMean = mean
    net.latentScale = max(Float(sqrt(variance)), 1e-3)
}

/// Цикл обучения сети движения на случайных парах.
func motionTrainLoop(net: MotionMLPGPU, session: GpuSession,
                     pairs: [MotionPair], latents: [Int: [Float]],
                     mseBuf: MTLBuffer?, steps: Int, lr: Float,
                     seed: UInt64) {
    var rng = SplitMix64(seed: seed)
    for step in 1...steps {
        let pair = pairs[Int(rng.next() % UInt64(pairs.count))]
        _ = runMotionStep(net: net, session: session, pair: pair,
                          latents: latents, mseBuf: mseBuf,
                          step: step, lr: lr)
    }
}

/// MSE окна (a,b): lerp против сети движения для всех промежуточных кадров.
func motionEvalWindow(net: MotionMLPGPU, latents: [Int: [Float]],
                      a: Int, b: Int) -> (lerp: Double, net: Double) {
    var l = 0.0
    var n = 0.0
    var total = 0
    for k in (a + 1)..<b {
        guard let zA = latents[a], let zB = latents[b], let zK = latents[k] else { continue }
        let t = Double(k - a) / Double(b - a)
        l += motionMSE(interpolateScalars(zA, zB, t: t), zK)
        n += motionMSE(predictMotionLatent(net: net, zA: zA, zB: zB, t: t), zK)
        total += 1
    }
    return (l / Double(max(total, 1)), n / Double(max(total, 1)))
}

/// Каскадное деление пополам: предсказание середины, рекурсивно до листьев.
func motionCascadeWindow(net: MotionMLPGPU, latents: [Int: [Float]],
                         a: Int, b: Int) -> Double {
    guard let zA = latents[a], let zB = latents[b] else { return 0 }
    var predicted: [Int: [Float]] = [a: zA, b: zB]
    func cascade(_ x: Int, _ y: Int) {
        guard y - x > 1 else { return }
        let m = (x + y) / 2
        if predicted[m] == nil {
            predicted[m] = predictMotionLatent(net: net,
                                               zA: predicted[x]!,
                                               zB: predicted[y]!,
                                               t: 0.5)
        }
        cascade(x, m)
        cascade(m, y)
    }
    cascade(a, b)
    var total = 0.0
    var count = 0
    for k in (a + 1)..<b {
        guard let zK = latents[k], let pred = predicted[k] else { continue }
        total += motionMSE(pred, zK)
        count += 1
    }
    return total / Double(max(count, 1))
}
