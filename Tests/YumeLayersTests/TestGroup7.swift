// YumeLayers — группа тестов (выделено из main.swift).
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import Metal
import AVFoundation
import UniformTypeIdentifiers
import Accelerate
import VideoToolbox
import IOKit

func testLayeredTrainer() {
    section("Layered trainer: сходимость + ворота качества")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let report = trainLayeredVAE(steps: 60, w: 32, h: 24, seed: 2026)
    expect(report.firstLoss.isFinite && report.lastLoss.isFinite,
           "потери конечны на обоих концах обучения")
    expect(report.lastLoss < report.firstLoss,
           String(format: "потеря упала: %.5f → %.5f", report.firstLoss, report.lastLoss))
    expect(report.psnr.isFinite, "PSNR собранного кадра конечен")
    expect(report.ssim.isFinite && report.ssim >= 0 && report.ssim <= 1,
           "SSIM собранного кадра в [0,1]")
    // Ворота обязаны быть честными: короткий прогон на 32×24 не может
    // выдавать 30 дБ, и «принят» должен совпадать с формальным порогом.
    expectEqual(report.acceptedByFloor,
                report.psnr >= 30 && report.ssim >= 0.90,
                "ворота качества согласованы с порогом 30 дБ / 0.90")
    expect(report.compositeChannels == 3, "выход сборки — RGB, 3 канала")
}

// MARK: - Потолок GPU: изменяемый параметр (до физического предела 98%)


func testDutyCap() {
    section("Предохранитель GPU: параметр мощности соблюдается")

    expectClose(safeDutyCap, 0.98, 1e-9, "физический потолок доли работы = 98%")
    expectEqual(defaultGpuPowerPercent, 90, "мощность по умолчанию = 90%")
    expectClose(effectiveDutyCeiling(powerPercent: 98), 0.98, 1e-9,
                "запрошенные 98% не режутся до 90%")
    expectClose(effectiveDutyCeiling(powerPercent: 100), 0.98, 1e-9,
                "100% упираются только в физический предел 98%")
    expectClose(effectiveDutyCeiling(powerPercent: 200), 0.98, 1e-9,
                "некорректные 200% тоже упираются в 98%")
    expectClose(effectiveDutyCeiling(powerPercent: 85), 0.85, 1e-9,
                "85% проходят как есть")
    expect(effectiveDutyCeiling(powerPercent: 85) < safeDutyCap,
           "85% остаётся ниже потолка")
    expect(effectiveDutyCeiling(powerPercent: 100) <= maxSafeDuty,
           "предохранитель не слабее физического предела")

    let guard90 = GpuDutyGuard()
    expectClose(guard90.duty, 0.90, 1e-9, "guard по умолчанию: доля работы 90%")
    let pause = guard90.pauseMs(gpuMs: 100)
    expect(pause > 0, "guard: после шага 100 мс есть пауза")
    expectClose(100.0 / (100.0 + pause), 0.90, 1e-6,
                "guard: фактическая занятость GPU = 90%")
    expectEqual(guard90.pauseMs(gpuMs: 0), 0, "guard: нулевой шаг → без паузы")
    expectEqual(guard90.emergencyPauseMs(tempC: 70), 0, "guard: 70 °C без аварии")
    expect(guard90.emergencyPauseMs(tempC: emergencyTempC + 1) > 0,
           "guard: аварийная температура → принудительное остывание")

    let guard98 = GpuDutyGuard(powerPercent: 98)
    expectClose(guard98.duty, 0.98, 1e-9, "guard: запрошенные 98% соблюдаются")
    let pause98 = guard98.pauseMs(gpuMs: 100)
    expectClose(100.0 / (100.0 + pause98), 0.98, 1e-6,
                "guard: фактическая занятость GPU = 98%")
    expect(Params().gpuPowerPercent == 90, "Params по умолчанию = 90%")
}

/// `runGpuHeat` — диагностический прогрев, но он гоняет тот же 856×480
/// forward+backward, что и обучение, и поэтому не имеет права обходить потолок.
/// До исправления там стоял `ceiling: auto ? maxSafeDuty : 1.0`, то есть 98–100%.

func testHeatDutyCap() {
    section("Прогрев GPU: физический потолок без обходных путей")

    expectClose(clampedGpuDuty(1.0), safeDutyCap, 1e-9,
                "gpuheat: ручной режим 100% упирается в 98%")
    expectClose(clampedGpuDuty(0.99), safeDutyCap, 1e-9,
                "gpuheat: 99% упирается в 98%")
    expectClose(clampedGpuDuty(0.90), 0.90, 1e-9,
                "gpuheat: ровно 90% проходят как есть")
    expectClose(clampedGpuDuty(0.75), 0.75, 1e-9,
                "gpuheat: 75% проходят как есть")
    let autoDuty = clampedGpuDuty(-1)
    expect(autoDuty.isFinite && autoDuty >= 0.05 && autoDuty <= safeDutyCap,
           "gpuheat: авто-режим (duty ≤ 0) остаётся в коридоре")
    let nanDuty = clampedGpuDuty(Double.nan)
    expect(nanDuty.isFinite && nanDuty <= safeDutyCap,
           "gpuheat: NaN не превращается в полную нагрузку")

    // Сам регулятор с новым потолком не должен выходить за 98% ни при
    // каких показаниях датчика, включая «холодно» (там он разгоняется).
    var power = PowerController(ceiling: safeDutyCap, startDuty: 0.5)
    expectClose(power.ceiling, safeDutyCap, 1e-9,
                "PowerController: потолок = 98%")
    for temp in stride(from: 40.0, through: 100.0, by: 5.0) {
        let stats = GpuStats(temperatureC: temp, coreClockMHz: 400,
                             activityPercent: 90, powerW: 30)
        let d = power.update(stats: stats)
        expect(d <= safeDutyCap + 1e-9,
               "PowerController при \(Int(temp)) °C не выше 98%")
    }
    expect(power.duty <= safeDutyCap + 1e-9,
           "PowerController после 13 замеров не выше 98%")

    // Пауза при 98% обязана быть положительной: 2% времени уходят системе
    // и это не даёт повториться watchdog-перезагрузке.
    let guard90 = GpuDutyGuard(powerPercent: 100)
    expect(guard90.pauseMs(gpuMs: 150) > 0,
           "даже запрос 100% оставляет паузу на остывание")
}

// MARK: - Итерация 6: реальные кадры → слои → обучение под ключ


func testLayerAnnotation() {
    section("Разметка реальных кадров: точная факторизация 8 каналов")
    let w = 16, h = 16
    let n = w * h

    // Статичный фон + движущийся квадрат: классическое отделение объектов
    // от фона, но без единой строки на Python и без ML-модели.
    func backdrop() -> [Float] {
        var f = [Float](repeating: 0, count: n * 3)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 3
                let t = Float(y) / Float(h - 1)
                f[i] = 0.20 + 0.50 * t
                f[i + 1] = 0.30 + 0.40 * t
                f[i + 2] = 0.60 - 0.30 * t
            }
        }
        return f
    }
    func shot(_ sx: Int) -> [Float] {
        var f = backdrop()
        for y in 4..<8 {
            for x in sx..<(sx + 4) {
                let i = (y * w + x) * 3
                f[i] = 0.95; f[i + 1] = 0.10; f[i + 2] = 0.10
            }
        }
        return f
    }
    // Позиции объекта подобраны так, чтобы ни один пиксель не был закрыт
    // объектом более чем в 2 из 5 кадров: иначе временная медиана ПО ПРАВУ
    // становится объектом, и требовать от неё чистый фон было бы ошибкой
    // самого теста, а не алгоритма.
    let frames = [shot(0), shot(4), shot(8), shot(12), shot(2)]
    let params = LayerAnnotationParams(diffThreshold: 0.08, minLight: 0.70,
                                       blurRadius: 2, dilateRadius: 1)

    let bg = temporalMedianBackground(frames, w: w, h: h)
    expect(maxAbsDiff(bg, backdrop()) < 0.02,
           "временная медиана восстановила статичный фон")

    let scenes = annotateFrames(frames, w: w, h: h, params: params)
    expectEqual(scenes.count, frames.count, "по одной сцене на кадр")
    var totalMask = 0
    for i in 0..<scenes.count {
        let sc = scenes[i]
        let got = composeLayers(w: w, h: h, background: sc.background,
                                objects: sc.objects, mask: sc.mask, light: sc.light)
        expect(maxAbsDiff(got, frames[i]) < 1e-5,
               "кадр \(i): compose(слои) == исходный RGB")
        expect(sc.mask.allSatisfy { $0 == 0 || $0 == 1 },
               "кадр \(i): маска бинарна")
        totalMask += sc.mask.filter { $0 == 1 }.count
        let lo = params.minLight - 1e-6
        expect(sc.light.allSatisfy { $0 >= lo && $0 <= 1 + 1e-6 },
               "кадр \(i): свет в [minLight, 1]")
        var objectsClean = true
        for p in 0..<n where sc.mask[p] == 0 {
            let a = sc.objects[p * 3]
            let b = sc.objects[p * 3 + 1]
            let c = sc.objects[p * 3 + 2]
            if a != 0 || b != 0 || c != 0 { objectsClean = false; break }
        }
        expect(objectsClean, "кадр \(i): объекты нулевые вне маски")
    }
    expect(totalMask > 0, "движущийся объект обнаружен")

    let flat = [Float](repeating: 0.5, count: n)
    expect(maxAbsDiff(boxBlur(flat, w: w, h: h, radius: 3), flat) < 1e-6,
           "boxBlur константы не меняет её")
    let empty = differenceMask(frame: frames[0], background: frames[0],
                               w: w, h: h, threshold: 0.08, dilate: 1)
    expect(empty.allSatisfy { $0 == 0 }, "нулевая разница → пустая маска")

    let light = illuminationMap(frame: frames[0], w: w, h: h,
                                minLight: params.minLight, blurRadius: 2)
    var lightSafe = true
    for i in 0..<n {
        let r = frames[0][i * 3]
        let g = frames[0][i * 3 + 1]
        let b = frames[0][i * 3 + 2]
        let mx = max(r, max(g, b))
        if light[i] < mx - 1e-6 { lightSafe = false; break }
    }
    expect(lightSafe, "свет >= max канала: деление не выходит за [0,1]")

    let single = annotateFrames([frames[0]], w: w, h: h, params: params)
    expect(single[0].mask.allSatisfy { $0 == 0 },
           "один кадр → маска пуста")
    let one = composeLayers(w: w, h: h, background: single[0].background,
                            objects: single[0].objects, mask: single[0].mask,
                            light: single[0].light)
    expect(maxAbsDiff(one, frames[0]) < 1e-5,
           "один кадр: compose всё равно == RGB")
}


func testLayeredCapacityAndSchedule() {
    section("Ёмкость латента и расписание lr")
    let base: Float = 0.01
    let total = 100
    expectClose(Double(cosineLR(step: 0, totalSteps: total, baseLR: base)),
                Double(base), 1e-6, "lr на старте = baseLR")
    expectClose(Double(cosineLR(step: total, totalSteps: total, baseLR: base)),
                Double(base * 0.05), 1e-6, "lr в конце = 5% от baseLR")
    var prev = Double(base) + 1
    var monotonic = true
    for s in stride(from: 0, through: total, by: 10) {
        let v = Double(cosineLR(step: s, totalSteps: total, baseLR: base))
        if v > prev + 1e-9 { monotonic = false }
        prev = v
    }
    expect(monotonic, "lr монотонно убывает")
    expectClose(Double(cosineLR(step: total * 2, totalSteps: total, baseLR: base)),
                Double(base * 0.05), 1e-6, "шаг вне расписания не повышает lr")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let vae16 = ConvVAEGPU(engine: engine, channels: 8)
    let vae32 = ConvVAEGPU(engine: engine, channels: 8, latentChannels: 32)
    expectEqual(vae16.latentChannels, ConvVAEGPU.latentC, "по умолчанию C = 16")
    expectEqual(vae32.latentChannels, 32, "layered-путь: C = 32")
    expectEqual(vae32.e3.oc, 64, "e3 отдаёт 2·C каналов")
    expectEqual(vae32.d1.ic, 32, "d1 принимает C каналов")
    expectClose(ConvVAEGPU.compressionRatio(channels: 8, latentChannels: 32), 16, 1e-9,
                "сжатие 8 каналов при C=32 = 16×")

    let session = GpuSession(engine)
    let data = [Float](repeating: 0.5, count: 8 * 16 * 16)
    let x = GPUTensor(engine: engine, data: data, rows: 8)
    session.begin()
    let cache = vae32.forward(session, image: x, w: 16, h: 16, deterministic: true)
    _ = session.end()
    expectEqual(cache.z.rows, 32, "латент реально 32 канала")
    expectEqual(cache.recon.rows, 8, "выход по-прежнему 8 каналов")

    let b16 = convVAEBytesPerSample(w: 64, h: 48, channels: 8, layered: true)
    let b32 = convVAEBytesPerSample(w: 64, h: 48, channels: 8, layered: true,
                                    latentChannels: 32)
    expect(b32 > b16, "бюджет VRAM учитывает ёмкость латента")
    expectThrows("чекпоинт C=16 отвергнут моделью C=32") {
        try vae32.load(from: vae16.matrices)
    }
}


func testLayeredDatasetTraining() {
    section("Обучение на датасете сцен + чекпоинт")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let scenes = [makeLayeredScene(w: 16, h: 16, seed: 101),
                  makeLayeredScene(w: 16, h: 16, seed: 202),
                  makeLayeredScene(w: 16, h: 16, seed: 303)]
    let ckpt = NSTemporaryDirectory() + "yume_layered_" + UUID().uuidString + ".safetensors"
    let report = trainLayeredVAE(steps: 30, w: 16, h: 16,
                                 scenes: scenes, latentChannels: 32,
                                 checkpointPath: ckpt)
    expectEqual(report.sceneCount, 3, "тренер увидел все 3 сцены")
    expect(report.lastLoss < report.firstLoss, "потеря по датасету упала")
    expect(report.psnr.isFinite && report.ssim.isFinite, "качество конечно")
    expectClose(Double(report.duty), 0.90, 1e-9, "потолок GPU = 90%")
    do {
        let mats = try loadState(path: ckpt)
        expectEqual(mats.count, 12, "чекпоинт: 12 матриц")
        expectEqual(mats[4].rows, 64, "чекпоинт: e3 = 2·C = 64")
        try? FileManager.default.removeItem(atPath: ckpt)
    } catch {
        expect(false, "чекпоинт не читается: \(error)")
    }
}


func testLayerPipeline() {
    section("Конвейер под ключ: кадры → слои → обучение → чекпоинт → PNG")
    let w = 16, h = 16
    let n = w * h
    func frame(_ sx: Int) -> [Float] {
        var f = [Float](repeating: 0, count: n * 3)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 3
                let t = Float(y) / Float(h - 1)
                f[i] = 0.2 + 0.5 * t
                f[i + 1] = 0.3 + 0.4 * t
                f[i + 2] = 0.6 - 0.3 * t
            }
        }
        for y in 4..<8 {
            for x in sx..<(sx + 4) {
                let i = (y * w + x) * 3
                f[i] = 0.9; f[i + 1] = 0.1; f[i + 2] = 0.1
            }
        }
        return f
    }
    let frames = [frame(0), frame(4), frame(8), frame(12)]

    // Моменты времени: чистая функция, проверяется без видеофайла.
    let times = sampleVideoTimes(count: 4, duration: 10)
    expectEqual(times.count, 4, "4 момента времени")
    expect(times[0] < times[1] && times[1] < times[2] && times[2] < times[3],
           "времена монотонно возрастают")
    expect(times[0] >= 0 && times[3] <= 10, "внутри длительности ролика")
    expectClose(sampleVideoTimes(count: 1, duration: 10, windowSeconds: 8,
                                 startSeconds: 0)[0], 4, 1e-9,
                "один кадр = середина заданного отрывка")
    // Ключевое для 30-минутного ролика: НЕ разбрасывать кадры по всему
    // фильму, иначе временная медиана фона превращается в кашу из сцен.
    let longVideo = sampleVideoTimes(count: 16, duration: 1800)
    expect(longVideo[15] - longVideo[0] <= 6.0001,
           "стек берётся из короткого непрерывного отрывка, а не из всего фильма")

    let params = LayerAnnotationParams(diffThreshold: 0.08, minLight: 0.70,
                                       blurRadius: 1, dilateRadius: 0)
    let ds = makeLayerDataset(frames: frames, w: w, h: h, params: params)
    expectEqual(ds.scenes.count, frames.count, "датасет: по сцене на кадр")
    expect(ds.compositeError < 1e-5, "датасет: обратная сборка точна")
    expect(ds.meanCoverage > 0 && ds.meanCoverage < 1,
           "датасет: маска не пустая и не сплошная")

    // Путь decodeImageRGB, который использует AVAssetImageGenerator:
    // проверяем через PNG в памяти, чтобы не тащить видеофайл в юнит-тесты.
    if let png = encodePNG(width: w, height: h, rgb: frames[0]),
       let back = loadImageRGB(data: png, w: w, h: h) {
        expect(maxAbsDiff(back, frames[0]) < 0.02,
               "PNG → decodeImageRGB сохраняет цвет кадра")
    } else {
        expect(false, "PNG → decodeImageRGB: декодирование не вернуло кадр")
    }

    guard gHasMetal else { skip("Metal недоступен"); return }
    let tmp = NSTemporaryDirectory() + "yume_pipe_" + UUID().uuidString
    let ckpt = tmp + ".safetensors"
    do {
        let r = try runLayerPipeline(frames: frames, w: w, h: h,
                                     frameCount: frames.count, steps: 12,
                                     latentChannels: 32,
                                     checkpointPath: ckpt,
                                     previewPrefix: tmp + "_preview")
        expectEqual(r.frameCount, frames.count, "pipeline: все кадры дошли")
        expectEqual(r.train.sceneCount, frames.count, "pipeline: тренер учился на всех сценах")
        expectEqual(r.train.latentChannels, 32, "pipeline: ёмкость латента C=32")
        expect(r.train.checkpointSaved, "pipeline: чекпоинт записан")
        expect(r.train.lastLoss < r.train.firstLoss, "pipeline: потеря упала")
        expectEqual(r.previewPaths.count, 7, "pipeline: 7 PNG-превью")
        expect(r.previewPaths.allSatisfy { FileManager.default.fileExists(atPath: $0) },
               "pipeline: превью реально на диске")
        expect(r.train.reconstructionHWC.count == n * 3,
               "pipeline: реконструкция той же размерности, что кадр")
        let mats = try loadState(path: ckpt)
        expectEqual(mats.count, 12, "pipeline: чекпоинт содержит 12 матриц")
        expectEqual(mats[4].rows, 64, "pipeline: e3 = 2·C = 64")
        for path in r.previewPaths { try? FileManager.default.removeItem(atPath: path) }
        try? FileManager.default.removeItem(atPath: ckpt)
    } catch {
        expect(false, "pipeline завершился ошибкой: \(error)")
    }
}


func testLayeredBatching() {
    section("Мини-батч: один command buffer на несколько сцен")
    guard gHasMetal else { skip("Metal недоступен"); return }
    // Корректная эквивалентность: ОДИНАКОВОЕ число обновлений Adam.
    // Батч 2 по [S, S] (сцена продублирована) с 20 обновлениями должен дать
    // ту же траекторию, что одиночный батч по [S] с теми же 20 обновлениями:
    // среднее из двух ТОЖДЕСТВЕННЫХ градиентов равно самому градиенту.
    // (Это изолирует проверку накопления/нормировки от разных расписаний лr:
    // сравнение «батч 4 за 8 шагов vs одиночный за 32» было бы некорректно.)
    let S = makeLayeredScene(w: 16, h: 16, seed: 99)
    let dupe = [S, S]
    let updates = 20
    let rBatch = trainLayeredVAE(steps: updates, w: 16, h: 16,
                                 scenes: dupe, latentChannels: 32,
                                 cosineSchedule: true, batchSize: 2)
    let rSingle = trainLayeredVAE(steps: updates, w: 16, h: 16,
                                  scenes: [S], latentChannels: 32,
                                  cosineSchedule: true, batchSize: 1)
    expectEqual(rBatch.sceneCount, 2, "батч-тренер видел обе сцены")
    expectEqual(rSingle.sceneCount, 1, "одиночный видел одну сцену")
    expect(rBatch.lastLoss < rBatch.firstLoss, "батч: потеря упала")
    expect(rSingle.lastLoss < rSingle.firstLoss, "одиночный: потеря упала")
    // Допуск не жёсткий: шаги батча берут разные сэмплы z (seed на forward),
    // так что дублированная пара не побитово равна одиночной, но статистика
    // одинаковых градиентов обязана сойтись.
    expectClose(rBatch.lastLoss, rSingle.lastLoss, 0.02,
                String(format: "батч 2 по дупликату == одиночный: %.5f vs %.5f",
                       rBatch.lastLoss, rSingle.lastLoss))
    expectClose(rBatch.duty, 0.90, 1e-9, "батч сохраняет потолок 90%")
    expect(rBatch.psnr.isFinite && rBatch.ssim.isFinite, "качество конечно")
}


func testLayeredOverlap() {
    section("Double-buffering CPU↔GPU: overlap даёт те же числа, что синхронный")
    guard gHasMetal else { skip("Metal недоступен"); return }
    // Overlap только перекрывает кодирование следующего шага со счётом
    // текущего на GPU. Математика не меняется: те же сцены, те же seed'ы
    // (step*batch+b), тот же порядок очереди command buffer'ов, те же
    // обновления Adam. Значит траектория обучения обязана совпасть — это
    // и есть доказательство, что перекрытие не ломает зависимости параметров.
    let S = makeLayeredScene(w: 24, h: 24, seed: 7)
    let steps = 40
    let sync = trainLayeredVAE(steps: steps, w: 24, h: 24,
                               scenes: [S], latentChannels: 32,
                               cosineSchedule: true, overlap: false)
    let ovl = trainLayeredVAE(steps: steps, w: 24, h: 24,
                              scenes: [S], latentChannels: 32,
                              cosineSchedule: true, overlap: true)
    expect(ovl.steps == steps, "overlap отработал заявленное число шагов")
    expectClose(ovl.firstLoss, sync.firstLoss, 1e-6,
                String(format: "firstLoss совпадает: %.6f vs %.6f",
                       ovl.firstLoss, sync.firstLoss))
    expectClose(ovl.lastLoss, sync.lastLoss, 1e-6,
                String(format: "lastLoss совпадает: %.6f vs %.6f",
                       ovl.lastLoss, sync.lastLoss))
    expectClose(Double(ovl.psnr), Double(sync.psnr), 1e-4,
                String(format: "PSNR совпадает: %.2f vs %.2f",
                       ovl.psnr, sync.psnr))
    expectClose(Double(ovl.ssim), Double(sync.ssim), 1e-5,
                String(format: "SSIM совпадает: %.5f vs %.5f",
                       ovl.ssim, sync.ssim))
}


func testDutyPacer() {
    section("Пейсер доли работы: пакетные паузы без потери 90%")
    let pacer = GpuDutyPacer(powerPercent: 90, maxPauseMs: 2000, minSleepMs: 2.0)
    expectClose(pacer.duty, 0.90, 1e-9, "пейсер наследует жёсткие 90%")
    // Короткий шаг: пауза копится в долг, но НЕ будит планировщик каждый раз.
    _ = pacer.wait(gpuMs: 1.0)
    expect(pacer.pendingPauseMs > 0, "короткий шаг копит долг паузы")
    expectEqual(pacer.sleepCalls, 0, "на коротком шаге сна ещё нет")
    // Накопим до порога и проверим, что суммарная доля работы всё та же.
    for _ in 0..<200 { _ = pacer.wait(gpuMs: 1.0) }
    _ = pacer.sleepDebt()
    expectEqual(pacer.pendingPauseMs, 0, "после сброса долга нет")
    expect(pacer.sleepCalls < 20,
           "201 шаг обошёлся \(pacer.sleepCalls) снами вместо 201")
    expectClose(pacer.measuredDuty, 0.90, 0.02,
                "фактическая доля работы осталась ~90%")
    expectEqual(pacer.emergency(tempC: 70), 0, "при 70 °C аварии нет")
    expectEqual(pacer.emergencyPauses, 0, "нормальная температура не считается аварией")
    // Отдельный пейсер с паузой 1 мс, чтобы тест не спал 2 секунды.
    let hot = GpuDutyPacer(powerPercent: 90, maxPauseMs: 1, minSleepMs: 2)
    expect(hot.emergency(tempC: emergencyTempC) > 0, "при 86 °C пауза применена сразу")
    expectEqual(hot.emergencyPauses, 1, "авария учтена один раз")
}


func testFrameStabilization() {
    section("Стабилизация камеры: сдвиг, обрезка, точная сборка")
    let w = 32, h = 32
    func base() -> [Float] {
        var f = [Float](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 3
                // НЕпериодический узор: на периодической «шахматке» сдвиг
                // на целый период даёт такой же ноль невязки, и поиск сдвига
                // становится неоднозначным. Хэш гарантирует единственный минимум.
                var q = UInt64(x) &* 73856093
                q ^= UInt64(y) &* 19349663
                q = q &* 2654435761
                let v = Float((q >> 24) & 255) / 255
                f[i] = v; f[i + 1] = v * 0.6; f[i + 2] = 0.3
            }
        }
        return f
    }
    let b = base()
    /// Кадр со сдвинутой «камерой»: пиксель (x,y) берёт базу из (x+s, y).
    func shifted(_ s: Int) -> [Float] {
        var f = [Float](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            for x in 0..<w {
                let sx = min(w - 1, max(0, x + s))
                let si = (y * w + sx) * 3
                let di = (y * w + x) * 3
                f[di] = b[si]; f[di + 1] = b[si + 1]; f[di + 2] = b[si + 2]
            }
        }
        return f
    }
    let frames = [shifted(-3), shifted(-1), shifted(0), shifted(2), shifted(4)]
    let params = LayerAnnotationParams(diffThreshold: 0.08, minLight: 0.70,
                                       blurRadius: 1, dilateRadius: 0)
    let naive = makeLayerDataset(frames: frames, w: w, h: h, params: params)
    let stab = makeLayerDataset(frames: frames, w: w, h: h, params: params,
                                stabilize: true, maxShift: 6)
    expect(naive.meanCoverage > 0.2,
           "без выравнивания сдвиг камеры раздувает маску")
    expect(stab.alignment != nil, "отчёт стабилизации заполнен")
    expectEqual(stab.alignment?.droppedFrames ?? -1, 0,
                "однородный отрывок: ни один кадр не выброшен")
    expect(stab.meanCoverage < naive.meanCoverage,
           String(format: "выравнивание снизило долю объектов: %.1f%% → %.1f%%",
                  naive.meanCoverage * 100, stab.meanCoverage * 100))
    expect(stab.w % 8 == 0 && stab.h % 8 == 0,
           "размер после обрезки кратен 8 (латент /8)")
    expect(stab.compositeError < 1e-5,
           "после перестановки пикселей сборка всё ещё точна")
    expectEqual(stab.scenes.count, frames.count, "все кадры остались в датасете")
}

// MARK: - Извлечение кадров: многопоточный ffmpeg в RAM


