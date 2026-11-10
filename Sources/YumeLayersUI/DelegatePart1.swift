import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

extension YumeAppDelegate {
    func reconData(for name: String) -> Data? {
        reconMemory[name]
    }

    func renderCollagePage() {
        framesChip.setCount(framesStore.count)
        guard collageThumbs.count == 10 else { return }
        let total = framesStore.count
        collageEmpty.isHidden = total > 0
        let pages = max(1, (total + pageSize - 1) / pageSize)
        carouselPage = min(max(0, carouselPage), pages - 1)
        let start = carouselPage * pageSize
        let end = min(start + pageSize, total)

        for (slot, btn) in collageThumbs.enumerated() {
            let off = frameOffset(forSlot: slot)
            let idx = start + off
            if idx < end {
                if showRecon {
                    // В режиме реконструкции ВСЕ миниатюры — реконструкции
                    // тех же номеров кадров, что и в оригинале (из памяти).
                    let rd = reconData(for: framesStore[idx].name)
                    btn.image = rd.map { NSImage(data: $0) } ?? nil
                    btn.toolTip = "Реконструкция кадра \(idx + 1)"
                    btn.isEnabled = true
                    btn.contentTintColor = nil
                    thumbBadges[slot].isHidden = true
                } else {
                    btn.image = NSImage(data: framesStore[idx].jpeg)
                    btn.toolTip = "Кадр \(idx + 1)"
                    btn.isEnabled = true
                    btn.contentTintColor = nil
                    // Галочка — ТОЛЬКО после обработки моделью.
                    thumbBadges[slot].isHidden = !processedFrames[idx]
                }
            } else {
                btn.image = nil
                btn.isEnabled = false
                thumbBadges[slot].isHidden = true
            }
        }

        if total > 0 {
            let sel = min(max(0, selectedFrameIndex), total - 1)
            selectedFrameIndex = sel
            if showRecon {
                // Реконструкция с тем же номером кадра (из памяти).
                let reconData = reconData(for: framesStore[sel].name)
                if let rd = reconData, let img = NSImage(data: rd) {
                    collageBig.image = img
                    collageBig.toolTip = "Реконструкция кадра \(sel + 1)"
                    collageBigBadge.isHidden = false
                } else {
                    collageBig.image = nil
                    collageBig.toolTip = "Реконструкция кадра \(sel + 1) появится после обучения"
                    collageBigBadge.isHidden = true
                }
            } else {
                collageBig.image = NSImage(data: framesStore[sel].jpeg)
                collageBig.toolTip = "Кадр \(sel + 1)"
                collageBigBadge.isHidden = !processedFrames[sel]
            }
        } else {
            collageBig.image = nil
            collageBigBadge.isHidden = true
        }

        renderPageDots(pages: pages)
        carouselPrev.isEnabled = carouselPage > 0
        carouselNext.isEnabled = carouselPage < pages - 1
        carouselFirst.isEnabled = carouselPage > 0
        carouselLast.isEnabled = carouselPage < pages - 1
        updateShotButtons()
    }

    func updatePagination() {
        framesChip.setCount(framesStore.count)
        let total = framesStore.count
        let pages = max(1, (total + pageSize - 1) / pageSize)
        carouselPage = min(max(0, carouselPage), pages - 1)
        renderPageDots(pages: pages)
        carouselPrev.isEnabled = carouselPage > 0
        carouselNext.isEnabled = carouselPage < pages - 1
        carouselFirst.isEnabled = carouselPage > 0
        carouselLast.isEnabled = carouselPage < pages - 1
    }

    @objc func thumbClicked(_ sender: NSButton) {
        let off = frameOffset(forSlot: sender.tag)
        let idx = carouselPage * pageSize + off
        guard idx >= 0, idx < framesStore.count else { return }
        selectedFrameIndex = idx
        renderCollagePage()
    }

    func makeScreenshot() -> String? {        guard let content = window.contentView else { return nil }
        content.layoutSubtreeIfNeeded()
        let rep = content.bitmapImageRepForCachingDisplay(in: content.bounds)!
        content.cacheDisplay(in: content.bounds, to: rep)
        guard let data = rep.representation(using: .png, properties: [:]) else { return nil }
        let path = "/tmp/yume_window_shot.png"
        do {
            try data.write(to: URL(fileURLWithPath: path))
            return path
        } catch {
            return nil
        }
    }

    @objc func startTraining() {
        startTraining(maxFrames: 0, resume: false)
    }

    func startTraining(maxFrames: Int, resume: Bool) {
        log("startTraining(maxFrames=\(maxFrames), resume=\(resume))")
        log("guards: isTraining=\(isTraining) isReconstructing=\(isReconstructing) isExtracting=\(isExtracting) sheet=\(window.attachedSheet != nil)")
        guard !isTraining, !isReconstructing, !isExtracting, window.attachedSheet == nil else { return }
        window.makeFirstResponder(nil)
        var p = Params()
        // Размер обучения читается из полей как есть. Кадры в памяти
        // остаются полноразмерными: VAE декодирует их сразу в этот размер.
        p.width = widthCtl.intValue(def: 80)
        p.height = heightCtl.intValue(def: 56)
        guard p.width >= 8, p.height >= 8, p.width <= 8192, p.height <= 8192 else {
            log("Недопустимый размер обучения: \(p.width)×\(p.height) — допустимо 8…8192")
            return
        }
        p.epochs = max(1, epochsCtl.intValue(def: 20))
        p.steps = max(1, stepsCtl.intValue(def: 20))
        p.batch = max(1, batchCtl.intValue(def: 1))
        p.lr = Float(lrCtl.doubleValue(def: 0.0005))
        p.pauseMs = max(0, pauseCtl.intValue(def: 250))
        // Поле читается как есть; ограничиваем только физическим пределом,
        // чтобы интерфейс не обещал недостижимое.
        p.gpuPowerPercent = min(max(powerCtl.intValue(def: defaultGpuPowerPercent), 10),
                                Int(safeDutyCap * 100))
        p.resume = resumeSwitch.state == .on
        if resume { p.resume = true }   // API: продолжить с чекпоинта без диалога
        p.adaptiveThrottle = adaptiveSwitch.state == .on
        p.modelWidth = max(1, requestedModelWidth)
        p.latentChannels = max(1, requestedLatentChannels)
        p.lossKind = requestedLossKind
        p.edgeWeight = requestedEdgeWeight
        p.residual = requestedResidual
        p.groupNorm = requestedGroupNorm
        p.klBeta = requestedKlBeta
        p.convPath = (convPathCtl.stringValue as NSString).expandingTildeInPath

        guard p.lr.isFinite, p.lr > 0, p.lr <= 1,
              p.epochs <= 100_000, p.steps <= 1_000_000, p.batch <= 16 else {
            log("Недопустимые параметры: lr должен быть (0, 1], батч 1…16, эпох ≤ 100000, шагов ≤ 1000000")
            return
        }

        // Подготовка чекпоинта: при resume=false старый чекпоинт удаляется
        // (с бэкапом), и обучение всегда начинается с новых весов — без
        // диалогов. При resume=true файл сохраняется для дообучения.
        switch prepareCheckpointForTraining(convPath: p.convPath, resume: p.resume) {
        case .keep:
            log("Режим: продолжение с чекпоинта")
        case .fresh:
            log("Чекпоинт отсутствует — обучение начнётся с новых весов")
        case .removed(let backup):
            log("Старый чекпоинт удалён — обучение с нуля (бэкап: \(backup))")
        }

        startTrainingInternal(p, maxFrames: maxFrames)
    }

    func startTrainingInternal(_ pIn: Params, maxFrames: Int = 0) {
        guard !isTraining, !isReconstructing else { return }
        // Нельзя запускать обучение, пока идёт раскадровка: оба процесса
        // будут мешать друг другу (кнопка «Стоп» извлечения оставалась бы
        // активной, а GPU был бы перегружен).
        guard !isExtracting else {
            log("Обучение нельзя запустить во время извлечения кадров — дождитесь завершения")
            return
        }
        guard let p = resolveTrainingFrames(pIn, maxFrames: maxFrames) else { return }
        lastParams = p
        configureTrainingUI(p)
        // Окно движения считается на main-потоке (чтение UI-контролов из фона
        // может заблокировать/уронить приложение), затем передаётся в фон.
        let motionWindow = currentInterpolationWindow()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try trainConvVAE(
                    p: p,
                    onProgress: { ep, loss, psnr, previewPath in
                        DispatchQueue.main.async {
                            self.lastEpoch = ep
                            let line = String(format: "эпоха %d: loss %.6f  psnr %.2f дБ", ep, loss, psnr)
                            self.log(line)
                            self.epochCard.setValue("\(ep + 1)/\(p.epochs)")
                            self.lossCard.setValue(String(format: "%.4f", loss))
                            self.psnrCard.setValue(String(format: "%.2f", psnr))
                            self.chart.add(loss: loss, psnr: psnr)
                            self.api.updateProgress(epoch: ep + 1, loss: loss, psnr: psnr,
                                                   progress: Double(ep + 1) / Double(p.epochs))
                            if let path = previewPath {
                                // Обучение НЕ меняет список оригиналов (галочки/добавления
                                // относятся только к обработанным моделью кадрам отдельного списка).
                                self.api.setPreview(path: path)
                            }
                        }
                    },
                    onStep: { [weak self] report in
                        DispatchQueue.main.async {
                            guard let self = self else { return }
                            self.lastTrainStepAt = Date()
                            self.lastTrainReport = report
                            // Полосу прогресса двигают только настоящие шаги.
                            // Служебные отчёты (подготовка, догрузка окна VRAM)
                            // приходят с done = 0, и раньше из-за них полоса
                            // откатывалась на 0% каждые 1024 кадра.
                            if report.done > 0 {
                                let pv = Double(report.done) / Double(max(report.total, 1))
                                self.setTrainProgress(pv, done: report.done, total: report.total)
                            }
                            self.trainFrameLabel.stringValue = trainLiveText(report)
                            self.refreshTrainLiveStatus()
                            // Периодический лог в «Журнал» (каждые 200 шагов),
                            // чтобы было видно, что обучение идёт и на каком кадре.
                            // Служебные отчёты (подготовка, догрузка VRAM) приходят
                            // с done = 0 — их не пишем, иначе на каждое окно в
                            // 1024 кадра журнал получает 64 строки «шаг 0».
                            if shouldLogTrainStep(done: report.done, total: report.total) {
                                let pct = Double(report.done) / Double(max(report.total, 1)) * 100
                                let frameInfo = report.framesTotal > 0
                                    ? " · кадр \(report.frame) из \(report.framesTotal)" : ""
                                self.log(String(format: "обучение: шаг %d/%d (%.0f%%) · эпоха %d/%d%@",
                                               report.done, report.total, pct,
                                               report.epoch, report.epochs, frameInfo))
                            }
                        }
                    }
                )
                // Итоговое обучение движения: после VAE учим сеть воспроизводить
                // реальные промежуточные кадры между опорами (если есть кадры).
                // Окно вычисляется на main-потоке ДО погружения в фон
                // (чтение UI-контролов из фона = deadlock/краш).
                self.trainMotionAfterVae(p: p, window: motionWindow)
                DispatchQueue.main.async {
                    self.finishTraining(ok: true)
                }
            } catch is CancellationError {
                // Штатная остановка пользователем (кнопка «Стоп»/API) — НЕ ошибка:
                // не показываем красный статус «Ошибка», а просто завершаем обучение.
                DispatchQueue.main.async {
                    self.finishTraining(ok: false)
                }
            } catch {
                DispatchQueue.main.async {
                    self.log("Ошибка: \(error.localizedDescription)")
                    self.finishTraining(ok: false)
                }
            }
        }
    }

    /// Длина окна движения под текущие настройки интерполяции (секунды × fps).
    func currentInterpolationWindow() -> Int {
        let seconds = interpSecondsCtl.doubleValue(def: 5)
        let fps = interpFpsCtl.intValue(def: 24)
        return max(8, min(128, Int((seconds * Double(fps)).rounded())))
    }

    /// Обучение сети движения после успешного VAE. Ошибки не фатальны:
    /// реконструкции/завершение обучения не отменяются, пишем в журнал.
    /// Функция вызывается из фоновых потоков, поэтому UI-лог оборачивается
    /// в main.async (прямой доступ к NSTextView из фона = SIGSEGV).
    func trainMotionAfterVae(p: Params, steps: Int = 300, window: Int? = nil) {
        guard let frames = p.memoryFrames, !frames.isEmpty else {
            DispatchQueue.main.async { self.log("Движение: нет кадров в памяти — сеть движения не обучалась") }
            return
        }
        let convPath = p.convPath
        let motionPath = motionCheckpointPath(for: convPath)
        let pw = max(8, ((max(p.width, 8) + 7) / 8) * 8)
        let ph = max(8, ((max(p.height, 8) + 7) / 8) * 8)
        // Окно обучения = целевой длине интервала A→B: сеть выучивает динамику
        // именно этого масштаба (тест MotionTrainingImprovesPrediction).
        let w = max(8, min(128, window ?? currentInterpolationWindow()))
        do {
            DispatchQueue.main.async {
                self.log("=== Запуск обучения движения (сеть предсказания промежуточных кадров, окно \(w) кадров) ===")
            }
            try trainMotionMLPFromCheckpoint(memoryFrames: frames,
                                             vaeCheckpointPath: convPath,
                                             pw: pw, ph: ph,
                                             steps: steps, lr: 0.0005,
                                             window: w,
                                             maxEncodedFrames: 128,
                                             objectivePath: motionPath,
                                             onProgress: { done, total in
                DispatchQueue.main.async {
                    self.setTrainProgress(Double(done) / Double(max(total, 1)),
                                          done: done, total: total)
                }
            })
        } catch {
            DispatchQueue.main.async {
                self.log("⚠️ Движение: сеть движения не обучена (\(error.localizedDescription))")
            }
        }
    }

    @objc func stopTraining() {
        TrainingControl.shared.cancel()
        setStatus("Остановка…", ok: false)
        log("=== Остановка по запросу ===")
    }

    func autoTuneParams(frames: Int) {
        let batch = max(1, batchCtl.intValue(def: 1))
        let resume = resumeSwitch.state == .on
        let currentSteps = max(1, stepsCtl.intValue(def: 20))
        let currentEpochs = max(1, epochsCtl.intValue(def: 20))
        let currentLr = Float(lrCtl.doubleValue(def: 0.0005))
        // Чистая протестированная логика: resume=true → шаги/эпохи не
        // пересчитываются, но LR авто-снижается (если не задан вручную).
        let res = autoTuneValues(frames: frames, batch: batch, resume: resume,
                                 currentSteps: currentSteps, currentEpochs: currentEpochs,
                                 currentLr: currentLr,
                                 stepsEditedByUser: stepsEditedByUser,
                                 epochsEditedByUser: epochsEditedByUser,
                                 lrEditedByUser: lrEditedByUser)
        var applied = res.changed
        if res.changed {
            stepsCtl.setValue("\(res.steps)")
            epochsCtl.setValue("\(res.epochs)")
        }
        if let lr = res.lr {
            lrCtl.setValue(String(format: "%.6f", lr))
            applied = true
        }
        if applied {
            log("Авто-настройка под \(frames) кадров (батч \(batch), resume=\(resume)): \(res.epochs) эпох × \(res.steps) шагов, LR=\(lrCtl.stringValue)")
        } else if resume {
            log("Авто-расчёт: продолжение с чекпоинта — параметры не пересчитываются (все заданы вручную)")
        }
    }

    @objc func autoTuneNow() {
        guard !framesStore.isEmpty else {
            log("Авто-расчёт: нет кадров — сначала извлеките видео")
            return
        }
        autoTuneParams(frames: framesStore.count)
    }

    @objc func resetParams() {
        epochsEditedByUser = false
        stepsEditedByUser = false
        lrEditedByUser = false
        epochsCtl.setValue("20")
        stepsCtl.setValue("20")
        batchCtl.setValue("1")
        widthCtl.setValue("80")
        heightCtl.setValue("56")
        lrCtl.setValue("0.0003")
        pauseCtl.setValue("250")
        powerCtl.setValue("90")
        resumeSwitch.state = .off
        adaptiveSwitch.state = .on
        convPathCtl.setValue("~/YumeLayers/checkpoints/conv.safetensors")
        log("Параметры сброшены к умолчаниям")
    }

    func finishTraining(ok: Bool) {
        isTraining = false
        api.setTrainingActive(false, totalEpochs: 0)
        trainButton.isEnabled = true
        stopButton.isEnabled = false
        // Прогресс и живой индикатор обучения закончены — их место занимает
        // строка реконструкций.
        stopTrainHeartbeat()
        trainSpinner.stopAnimation(nil)
        trainLiveBlock.isHidden = true
        lastTrainStepAt = nil
        lastTrainReport = nil
        trainProgressBlock.isHidden = true
        setStatus(ok ? "Обучение завершено" : "Обучение прервано", ok: ok)
        log(ok ? "=== Завершено ===" : "=== Прервано ===")
        if ok {
            // После успешного обучения — реконструкции (контрольная выборка или все).
            let p = lastParams
            // Реконструкции идут в фоне: держим «Стоп» активной, чтобы можно было
            // прервать их (реальная отмена в runReconsOn через TrainingControl).
            TrainingControl.shared.reset()
            self.isReconstructing = true
            self.trainButton.isEnabled = false
            self.stopButton.isEnabled = true
            // Прогресс реконструкций — отдельной строкой под кнопками.
            self.reconsProgressBlock.isHidden = false
            self.reconsProgressBar.doubleValue = 0
            self.reconsPercentLabel.stringValue = "0%"
            self.setStatus("Реконструкции…", ok: true)
            self.log("=== Запуск реконструкций ===")
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let recon = try generateReconstructions(p: p) { dn, total in
                        DispatchQueue.main.async {
                            self.setReconsProgress(Double(dn) / Double(max(total, 1)))
                        }
                    }
                    DispatchQueue.main.async {
                        self.isReconstructing = false
                        self.trainButton.isEnabled = true
                        self.stopButton.isEnabled = false
                        self.reconsProgressBlock.isHidden = true
                        // Реконструкции — в памяти (без записи на диск).
                        self.reconMemory = recon
                        // Галочка ✓ ставится ТОЛЬКО здесь: кадр реально прошёл
                        // через модель. Раньше processedFrames никогда не
                        // становился true, и бейдж не появлялся вовсе.
                        let marked = markProcessedFrames(
                            names: self.framesStore.map { $0.name },
                            reconNames: Set(recon.keys),
                            processed: &self.processedFrames)
                        if marked > 0 {
                            self.log("Отмечено обработанных кадров: \(marked)")
                        }
                        self.log("Реконструкции: \(recon.count) кадров готовы (в памяти)")
                        self.renderCollagePage()
                        self.setStatus(TrainingControl.shared.isCancelled ? "Реконструкции остановлены" : "Обучение завершено. Реконструкции готовы", ok: !TrainingControl.shared.isCancelled)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.isReconstructing = false
                        self.trainButton.isEnabled = true
                        self.stopButton.isEnabled = false
                        self.reconsProgressBlock.isHidden = true
                        self.log("Ошибка реконструкций: \(error.localizedDescription)")
                    }
                }
            }
        }
    }

    func log(_ s: String) {
        api.appendLog(s)
        let t = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let line = "[\(t)] \(s)\n"
        logView.string += line
        // ограничение длины журнала (защита от разрастания)
        let lines = logView.string.components(separatedBy: "\n")
        if lines.count > 1500 {
            logView.string = lines.suffix(1500).joined(separator: "\n") + "\n"
        }
        logView.scrollToEndOfDocument(nil)
        // Дублируем в файл, чтобы логи переживали перезапуски.
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("YumeLayers", isDirectory: true)
            .appendingPathComponent("train.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            try? line.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    func appendCoreLog(_ s: String) {
        api.appendLog(s)
        logView.string += s + "\n"
        let lines = logView.string.components(separatedBy: "\n")
        if lines.count > 1500 {
            logView.string = lines.suffix(1500).joined(separator: "\n") + "\n"
        }
        logView.scrollToEndOfDocument(nil)
    }

    /// Определяет обучающие кадры и контрольную выборку для запуска.
    private func resolveTrainingFrames(_ pIn: Params, maxFrames: Int) -> Params? {
        var p = pIn
        if maxFrames > 0 {
            let limit = min(maxFrames, framesStore.count)
            guard limit > 0 else {
                log("Обучение: нет кадров для лимита \(maxFrames)")
                return nil
            }
            p.memoryFrames = Array(framesStore.prefix(limit))
            let names = Set(p.memoryFrames!.map { $0.name })
            p.controlFrames = controlFramesStore.filter { names.contains($0.name) }
            log("Обучение ограничено первыми \(limit) кадрами из \(framesStore.count)")
        } else {
            p.memoryFrames = framesStore.isEmpty ? nil : framesStore
            // Контрольная выборка (1 кадр/сек) для реконструкций на итоговой модели.
            p.controlFrames = controlFramesStore.isEmpty ? nil : controlFramesStore
        }
        return p
    }

    /// Настраивает UI и логирует параметры старта обучения.
    private func configureTrainingUI(_ p: Params) {
        TrainingControl.shared.reset()
        isTraining = true
        api.setTrainingActive(true, totalEpochs: p.epochs)
        trainButton.isEnabled = false
        stopButton.isEnabled = true
        // Прогресс — рядом с кнопкой «Обучить VAE»; реконструкции ещё не идут.
        trainProgressBlock.isHidden = false
        trainProgressBar.doubleValue = 0
        trainPercentLabel.stringValue = "0%"
        trainEtaLabel.stringValue = "…"
        // Живой индикатор: сразу показываем, что цикл запущен, и ждём первый шаг.
        trainLiveBlock.isHidden = false
        trainStateLabel.stringValue = "Обучение идёт"
        trainStateLabel.textColor = .labelColor
        trainFrameLabel.stringValue = "Ожидание первого шага…"
        lastTrainStepAt = Date()
        lastTrainReport = nil
        trainSpinner.startAnimation(nil)
        startTrainHeartbeat()
        reconsProgressBlock.isHidden = true
        reconsProgressBar.doubleValue = 0
        reconsPercentLabel.stringValue = "0%"
        trainStartTime = Date()
        chart.reset()
        epochCard.setValue("0")
        lossCard.setValue("0")
        psnrCard.setValue("0")
        videoEmptyState.isHidden = false
        // Кадры оригинала НЕ сбрасываются при старте обучения.
        lastEpoch = 0
        setStatus("Обучение…", ok: true)
        log("=== Запуск обучения ===")

        log(String(format: "Кадр %d×%d, эпох %d, шагов %d, батч %d, lr %.5f, пауза %d мс",
                   p.width, p.height, p.epochs, p.steps, p.batch, p.lr, p.pauseMs))
        // Термо-регулятор держит температуру GPU, а не просто «увеличивает паузы
        // между эпохами»: пауза применяется на КАЖДОМ шаге.
        log(String(format: "Термо-регулятор: потолок нагрузки %d%%, рабочая %.0f °C, стоп-порог %.0f °C",
                   p.gpuPowerPercent, defaultTargetTempC, defaultMaxTempC))
        log(p.adaptiveThrottle
            ? "Авто-пауза при перегреве: включена (снижает нагрузку при подходе к стоп-порогу)"
            : "Авто-пауза при перегреве: выключена (нагрузка держится постоянной)")
        log("API: http://127.0.0.1:17851 (только локальные запросы)")
    }

}
