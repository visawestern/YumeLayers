// YumeLayers — обучающий цикл для 8-канальной факторизации по экспертам.
//
// Отличие от `ConvTrainer.trainConvVAE`: вход не RGB-кадр, а 8 каналов
// ground-truth из `LayeredScene` (фон | объекты | маска | свет). Лосс —
// по слоям И по точной сборке кадра (LayeredVAELoss.swift). Это первый
// рабочий контур, где декодер учится не «рисовать картинку», а выдавать
// слои, из которых картинка собирается без потерь.
//
// ТЕПЛО. Каждый command buffer сопровождается `GpuDutyGuard`: GPU занята
// не более 90% времени, 10% остаётся WindowServer и на остывание. Раньше
// короткие прогоны (в том числе тесты) гнали шаги без пауз и держали
// очередь GPU целиком — именно с этого началась принудительная
// перезагрузка 4 ноября 2026. Аварийная температура добавляет паузу
// поверх скважности.
import Foundation

/// Итог короткого layered-прогона. Все поля — наблюдаемые, без «успеха
/// по умолчанию»: `acceptedByFloor` считается из реальных PSNR/SSIM.
public struct LayeredTrainReport {
    public let steps: Int
    public let firstLoss: Double
    public let lastLoss: Double
    public let psnr: Float
    public let ssim: Float
    public let acceptedByFloor: Bool
    public let compositeChannels: Int
    /// Фактический потолок скважности, с которым работал прогон.
    public let duty: Double
}

/// Обучить 8-канальный VAE на процедурной сцене с ground-truth слоями.
///
/// Разрешение принудительно округляется вверх до кратного 8 (латент /8).
/// Это не «демо», а тот же граф, что пойдёт на реальных кадрах: заменить
/// `makeLayeredScene` на аннотированный кадр — и цикл не меняется.
///
/// - Parameter powerPercent: потолок мощности GPU в процентах; даже 100
///   упирается в жёсткие 90% (`safeDutyCap`).
public func trainLayeredVAE(steps: Int = 60,
                            w: Int = 32, h: Int = 24,
                            seed: UInt64 = 2026,
                            lr: Float = 0.01,
                            beta: Float = 0.01,
                            layerWeight: Float = 1,
                            composeWeight: Float = 1,
                            powerPercent: Int = defaultGpuPowerPercent)
    -> LayeredTrainReport {

    let pw = ((max(w, 8) + 7) / 8) * 8
    let ph = ((max(h, 8) + 7) / 8) * 8
    let total = max(steps, 1)

    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let sensor = GpuSensor()
    let dutyGuard = GpuDutyGuard(powerPercent: powerPercent)

    // Один и тот же таргет живёт в VRAM весь прогон: гонять 8·w·h float'ов
    // на каждом шаге дороже, чем посчитать по ним градиент.
    let target = LayeredTarget(scene: makeLayeredScene(w: pw, h: ph, seed: seed))
    let targetGPU = LayeredTargetGPU(engine: engine, target: target)
    let x = GPUTensor(engine: engine, data: target.channelsCHW,
                      rows: LayeredScene.layerChannels)
    let vae = ConvVAEGPU(engine: engine, channels: LayeredScene.layerChannels)

    /// Детерминированный замер потери + собранного кадра. Пауза по скважности
    /// применяется и здесь: замер — такая же GPU-работа, как шаг обучения.
    func evaluate() -> (loss: Double, recon: [Float]) {
        session.begin()
        let cache = vae.forward(session, image: x, w: pw, h: ph, deterministic: true)
        let gpuMs = session.end()
        let recon = cache.recon.toArray()
        let value = layeredLoss(recon: recon, target: target,
                                layerWeight: layerWeight,
                                composeWeight: composeWeight)
                  + Double(beta) * ConvVAEGPU.klMean(cache)
        dutyGuard.wait(gpuMs: gpuMs)
        return (value, recon)
    }

    let first = evaluate()
    for step in 1...total {
        session.begin()
        let cache = vae.forward(session, image: x, w: pw, h: ph,
                                deterministic: false, seed: UInt32(step))
        let (_, grads) = vae.backwardLayered(session, cache, target: targetGPU,
                                             layerWeight: layerWeight,
                                             composeWeight: composeWeight,
                                             beta: beta)
        vae.applyAdam(session, grads: grads, t: step, lr: lr)
        let gpuMs = session.end()
        dutyGuard.wait(gpuMs: gpuMs)

        // Раз в 10 шагов смотрим на датчик: аварийная температура важнее
        // любого расписания, обучение обязано уступить GPU системе.
        if step % 10 == 0 {
            let extra = dutyGuard.emergencyPauseMs(tempC: sensor.read()?.temperatureC)
            if extra > 0 { Thread.sleep(forTimeInterval: extra / 1000) }
        }
    }
    let last = evaluate()

    // Пол качества — по СОБРАННОМУ кадру, а не по 8 каналам по отдельности.
    let quality = composedQuality(recon: last.recon, target: target)
    let floor = QualityFloor()
    let accepted = floor.accepts(reference: target.compositeHWC,
                                 reconstruction: composeHWC(w: pw, h: ph,
                                                            recon: last.recon))
    return LayeredTrainReport(steps: total,
                              firstLoss: first.loss, lastLoss: last.loss,
                              psnr: quality.psnr, ssim: quality.ssim,
                              acceptedByFloor: accepted,
                              compositeChannels: 3,
                              duty: dutyGuard.duty)
}
