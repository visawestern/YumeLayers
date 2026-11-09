import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

extension YumeAppDelegate {
    func frameOffset(forSlot slot: Int) -> Int {
        frameOffsetForSlot(slot)
    }

    func referenceRow(_ label: String, _ button: NSButton, _ lbl: NSTextField) -> NSStackView {
        let tag = NSTextField(labelWithString: label)
        let row = NSStackView(views: [tag, button, lbl])
        row.orientation = .horizontal
        row.spacing = 6
        return row
    }

    @objc func chooseReferenceA() { pickReference(isA: true) }
    @objc func chooseReferenceB() { pickReference(isA: false) }

    func pickReference(isA: Bool) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .tiff, .bmp, .heic, .gif, .image]
        panel.message = isA ? "Опорная картинка A — начало ролика" : "Опорная картинка B — конец ролика"
        panel.prompt = "Выбрать"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if isA {
            interpImageAPath = url.path
            interpALabel.stringValue = url.lastPathComponent
            interpALabel.toolTip = url.path
        } else {
            interpImageBPath = url.path
            interpBLabel.stringValue = url.lastPathComponent
            interpBLabel.toolTip = url.path
        }
        updateInterpolateButtons()
    }

    @objc func runInterpolation() {
        startInterpolation(imageA: interpImageAPath,
                           imageB: interpImageBPath,
                           seconds: interpSecondsCtl.doubleValue(def: 5),
                           fps: interpFpsCtl.intValue(def: 24),
                           outputPath: nil)
    }

    @objc func stopInterpolation() {
        InterpolationControl.shared.cancel()
        setStatus("Остановка интерполяции…", ok: false)
        log("=== Остановка интерполяции по запросу ===")
    }

    func updateInterpolateButtons() {
        let ready = !interpImageAPath.isEmpty && !interpImageBPath.isEmpty
            && !isInterpolating && convPathExists()
        interpRunButton.isEnabled = ready
        interpStopButton.isEnabled = isInterpolating
    }

    func convPathExists() -> Bool {
        let p = (convPathCtl.stringValue as NSString).expandingTildeInPath
        return FileManager.default.fileExists(atPath: bestCheckpointPath(for: p))
    }

    @objc func mediaTabChanged(_ sender: NSSegmentedControl) {
        let showResult = sender.selectedSegment == 1
        videoMediaCard?.isHidden = showResult
        resultMediaCard?.isHidden = !showResult
    }

    func defaultInterpolationOutputPath() -> String {
        let dir = (NSHomeDirectory() as NSString).appendingPathComponent("YumeLayers/outputs")
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let df = DateFormatter()
        df.dateFormat = "dd.MM.yyyy, HH-mm"
        return (dir as NSString).appendingPathComponent("interp_\(df.string(from: Date())).mp4")
    }

    func startInterpolation(imageA: String, imageB: String,
                                    seconds: Double, fps: Int, outputPath: String?) {
        guard !isInterpolating else {
            setStatus("Интерполяция уже идёт", ok: false)
            return
        }
        guard !imageA.isEmpty, !imageB.isEmpty else {
            setStatus("Нужны обе опорные картинки (A и B)", ok: false)
            return
        }
        let checkpoint = bestCheckpointPath(for: (convPathCtl.stringValue as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: checkpoint) else {
            setStatus("Нет файла модели — сначала обучите VAE", ok: false)
            return
        }
        lastInterpolationModel = checkpoint
        let out = outputPath ?? defaultInterpolationOutputPath()
        InterpolationControl.shared.reset()
        isInterpolating = true
        api.setInterpolatingActive(true)
        api.setLastInterpolation(nil)
        if interpProgressBlock == nil {
            interpProgressBlock = makeProgressBlock(bar: interpProgressBar,
                                                   percent: interpPercentLabel, eta: nil)
            interpProgressBlock.widthAnchor.constraint(equalToConstant: 360).isActive = true
        }
        interpProgressBlock.isHidden = false
        interpProgressBar.doubleValue = 0
        interpPercentLabel.stringValue = "0%"
        interpResultLabel.stringValue = "Готовим кадры…"
        updateInterpolateButtons()
        setStatus("Интерполяция A→B…", ok: true)
        log("=== Интерполяция A→B: \(imageA) → \(imageB), \(seconds) с × \(fps) fps ===")
        log("   Модель: \(checkpoint)")

        let request = InterpolationRequest(imageAPath: imageA,
                                           imageBPath: imageB,
                                           seconds: seconds,
                                           fps: fps,
                                           outputPath: out,
                                           modelWidth: widthCtl.intValue(def: 80),
                                           modelHeight: heightCtl.intValue(def: 56),
                                           checkpointPath: checkpoint)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let result = try interpolateVideo(request) { done, total in
                    DispatchQueue.main.async {
                        guard let self = self else { return }
                        let ratio = Double(done) / Double(max(total, 1))
                        self.interpProgressBar.doubleValue = ratio
                        self.interpPercentLabel.stringValue = "\(Int(ratio * 100))%"
                        self.api.updateInterpolationProgress(ratio)
                    }
                }
                DispatchQueue.main.async { self?.finishInterpolation(result) }
            } catch {
                DispatchQueue.main.async {
                    self?.isInterpolating = false
                    self?.api.setInterpolatingActive(false)
                    self?.interpResultLabel.stringValue = "Ошибка: \(error.localizedDescription)"
                    self?.setStatus("Интерполяция не удалась", ok: false)
                    self?.log("⚠️ Интерполяция: \(error.localizedDescription)")
                    self?.updateInterpolateButtons()
                }
            }
        }
    }

    func finishInterpolation(_ result: InterpolationResult) {
        isInterpolating = false
        api.setInterpolatingActive(false)
        api.updateInterpolationProgress(1)
        interpProgressBar.doubleValue = 1
        interpPercentLabel.stringValue = "100%"
        interpResultLabel.stringValue =
            "Готово: \(result.frameCount) кадров, \(result.width)×\(result.height), \(result.fps) fps — \(result.path)\n"
            + "Модель: \((lastInterpolationModel as NSString).lastPathComponent)"
        api.setLastInterpolation(["path": result.path,
                                  "frames": result.frameCount,
                                  "fps": result.fps,
                                  "seconds": result.seconds,
                                  "width": result.width,
                                  "height": result.height,
                                  "model": lastInterpolationModel,
                                  "ok": true])
        updateInterpolateButtons()
        setStatus("Видео A→B готово", ok: true)
        log("✅ Интерполяция готова: \(result.frameCount) кадров, \(result.width)×\(result.height), \(result.fps) fps")
        log("   Файл: \(result.path)")

        // Показываем результат в плеере «Готовое видео».
        let url = URL(fileURLWithPath: result.path)
        guard FileManager.default.fileExists(atPath: result.path) else { return }
        resultLooper?.disableLooping()
        resultQueuePlayer.pause()
        resultQueuePlayer.removeAllItems()
        let item = AVPlayerItem(url: url)
        resultLooper = AVPlayerLooper(player: resultQueuePlayer, templateItem: item)
        resultEmptyState.isHidden = true
        mediaTabsControl.selectedSegment = 1
        videoMediaCard?.isHidden = true
        resultMediaCard?.isHidden = false
    }

    func saveReconstructionsToDisk(dir: String?) {
        let baseDir = dir ?? (NSHomeDirectory() as NSString).appendingPathComponent("YumeLayers/outputs/recons")
        try? FileManager.default.createDirectory(atPath: baseDir, withIntermediateDirectories: true)
        var saved = 0
        for (name, data) in reconMemory {
            let path = (baseDir as NSString).appendingPathComponent(name)
            try? data.write(to: URL(fileURLWithPath: path))
            saved += 1
        }
        log("Реконструкции сохранены: \(saved) файлов в \(baseDir)")
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.mainMenu = buildMenu()

        _ = api.start()
        // Вся отладка core-кода (yumeLog) — в журнал приложения, а не только в консоль.
        YumeLogBus.shared.subscribe { [weak self] line in
            DispatchQueue.main.async { self?.appendCoreLog(line) }
        }
        configureTrainCallbacks()
        configureSetParamCallback()
        configureMiscCallbacks()
        configureTestFramesCallback()
        configureViewProviders()
        configureStateProvider()
        configureWindow()
    }

    /// Регистрирует /api/set: универсальный сеттер параметров интерфейса.
    private func configureSetParamCallback() {
        // /api/set: универсальный сеттер всех параметров интерфейса. Каждый
        // ключ обновляет тот же элемент управления, что и ручной ввод в UI,
        // поэтому следующий вызов /api/train использует эти значения.
        api.onSetParam = { [weak self] params in
            guard let self = self else { return (false, [], params.keys.map { "\($0)" }) }
            var applied: [String] = []
            var unknown: [String] = []
            DispatchQueue.main.sync {
                for (key, value) in params {
                    // Текстовый параметр: число/строка/bool → строка поля.
                    func textParam(_ ctl: EditableValueControl, _ markUserEdited: inout Bool) -> Bool {
                        guard let s = paramStringValue(value) else { return false }
                        ctl.setValue(s)
                        markUserEdited = true
                        return true
                    }
                    switch key {
                    case "lr":
                        if textParam(self.lrCtl, &self.lrEditedByUser) { applied.append(key) } else { unknown.append(key) }
                    case "epochs":
                        if textParam(self.epochsCtl, &self.epochsEditedByUser) { applied.append(key) } else { unknown.append(key) }
                    case "steps":
                        if textParam(self.stepsCtl, &self.stepsEditedByUser) { applied.append(key) } else { unknown.append(key) }
                    case "batch":
                        if let s = paramStringValue(value) { self.batchCtl.setValue(s); applied.append(key) } else { unknown.append(key) }
                    case "width":
                        if let s = paramStringValue(value) { self.widthCtl.setValue(s); applied.append(key) } else { unknown.append(key) }
                    case "height":
                        if let s = paramStringValue(value) { self.heightCtl.setValue(s); applied.append(key) } else { unknown.append(key) }
                    case "modelWidth":
                        if let s = paramStringValue(value), let w = Int(s), w >= 1 {
                            self.requestedModelWidth = w
                            applied.append(key)
                        } else { unknown.append(key) }
                    case "latentChannels":
                        if let s = paramStringValue(value), let c = Int(s), c >= 1 {
                            self.requestedLatentChannels = c
                            applied.append(key)
                        } else { unknown.append(key) }
                    case "lossKind":
                        if let s = paramStringValue(value), s == "mse" || s == "l1" {
                            self.requestedLossKind = s
                            applied.append(key)
                        } else { unknown.append(key) }
                    case "edgeWeight":
                        if let s = paramStringValue(value), let f = Float(s), f >= 0 {
                            self.requestedEdgeWeight = f
                            applied.append(key)
                        } else { unknown.append(key) }
                    case "residual":
                        if let b = paramBoolValue(value) {
                            self.requestedResidual = b
                            applied.append(key)
                        } else { unknown.append(key) }
                    case "groupNorm":
                        if let b = paramBoolValue(value) {
                            self.requestedGroupNorm = b
                            applied.append(key)
                        } else { unknown.append(key) }
                    case "pauseMs":
                        if let s = paramStringValue(value) { self.pauseCtl.setValue(s); applied.append(key) } else { unknown.append(key) }
                    case "gpuPowerPercent":
                        if let s = paramStringValue(value) { self.powerCtl.setValue(s); applied.append(key) } else { unknown.append(key) }
                    case "extractLimit":
                        if let s = paramStringValue(value) { self.extractLimitCtl.setValue(s); applied.append(key) } else { unknown.append(key) }
                    case "convPath":
                        if let s = paramStringValue(value) { self.convPathCtl.setValue(s); applied.append(key) } else { unknown.append(key) }
                    case "resume":
                        if let b = paramBoolValue(value) { self.resumeSwitch.state = b ? .on : .off; applied.append(key) } else { unknown.append(key) }
                    case "adaptiveThrottle":
                        if let b = paramBoolValue(value) { self.adaptiveSwitch.state = b ? .on : .off; applied.append(key) } else { unknown.append(key) }
                    case "selectedVideos":
                        if let arr = value as? [String] {
                            self.setSelectedVideos(arr.map { URL(fileURLWithPath: $0) })
                            applied.append(key)
                        } else { unknown.append(key) }
                    default:
                        unknown.append(key)
                    }
                }
            }
            return (unknown.isEmpty, applied, unknown)
        }
    }

    /// Регистрирует /api/train, /api/stop, /api/extract.
    private func configureTrainCallbacks() {
        // API /api/train запускает ТУ ЖЕ функцию, что и кнопка «Обучить VAE»
        // (та же логика бэкапа/диалога и те же параметры из интерфейса).
        api.onTrain = { [weak self] maxFrames, resume in
            DispatchQueue.main.async { self?.startTraining(maxFrames: maxFrames, resume: resume) }
        }
        api.onStop = { [weak self] in
            DispatchQueue.main.async { self?.stopTraining() }
        }
        api.onExtract = { [weak self] paths, maxFrames in
            DispatchQueue.main.async {
                guard let self = self, !paths.isEmpty else { return }
                let urls = paths.map { URL(fileURLWithPath: $0) }
                // Та же регистрация, что и при выборе в UI: список, поле, показ.
                self.setSelectedVideos(urls)
                // Тот же запуск извлечения, что и у кнопки «Вытащить кадры».
                self.startExtraction(urls: self.videoURLs, append: false, maxFrames: maxFrames ?? 0)
            }
        }
    }

    /// Регистрирует сброс/автоподбор/интерполяцию/слойный контур.
    private func configureMiscCallbacks() {
        api.onResetParams = { [weak self] in
            DispatchQueue.main.async { self?.resetParams() }
        }
        api.onAutoTune = { [weak self] in
            DispatchQueue.main.async { self?.autoTuneNow() }
        }
        api.onInterpolate = { [weak self] imageA, imageB, seconds, fps, out in
            DispatchQueue.main.async {
                self?.startInterpolation(imageA: imageA, imageB: imageB,
                                         seconds: seconds, fps: fps, outputPath: out)
            }
        }
        api.onInterpolateStop = {
            InterpolationControl.shared.cancel()
        }
        api.onSaveRecons = { [weak self] dir in
            DispatchQueue.main.async { self?.saveReconstructionsToDisk(dir: dir) }
        }
        api.onLayeredTrain = { [weak self] video, w, h, steps, latentChannels, checkpoint in
            DispatchQueue.global(qos: .userInitiated).async {
                guard let self = self else { return }
                do {
                    let report = try runLayerPipeline(
                        videoPath: video,
                        w: w, h: h,
                        frameCount: 24,
                        steps: steps,
                        latentChannels: latentChannels,
                        lr: 0.01,
                        checkpointPath: checkpoint,
                        previewPrefix: nil)
                    let dict: [String: Any] = [
                        "source": report.sourcePath,
                        "width": report.width, "height": report.height,
                        "frames": report.frameCount,
                        "psnr": Double(report.train.psnr),
                        "ssim": Double(report.train.ssim),
                        "acceptedByFloor": report.train.acceptedByFloor,
                        "sceneCount": report.train.sceneCount,
                        "latentChannels": report.train.latentChannels,
                        "checkpoint": report.checkpointPath ?? "",
                        "previews": report.previewPaths,
                        "ok": true,
                    ]
                    self.api.setLastLayered(dict)
                    self.log("✅ Слойный VAE: PSNR \(String(format: "%.2f", report.train.psnr)) дБ, SSIM \(String(format: "%.3f", report.train.ssim)), принят=\(report.train.acceptedByFloor)")
                } catch {
                    self.api.setLastLayered(["ok": false, "error": error.localizedDescription])
                    self.log("⚠️ Слойный контур: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Регистрирует /api/test-frames: проверка кадров в памяти.
    private func configureTestFramesCallback() {
        // /api/test-frames: проверка текущих кадров в памяти (JPEG-сигнатура,
        // декодирование, имена). count — сколько кадров проверить (0 — все).
        api.testFramesProvider = { [weak self] count in
            guard let self = self else { return "{\"error\":\"no window\"}" }
            var report = ""
            DispatchQueue.main.sync {
                func decodeJPEG(_ data: Data) -> Bool {
                    guard let src = CGImageSourceCreateWithData(data as CFData, nil)
                    else { return false }
                    return CGImageSourceCreateImageAtIndex(src, 0, nil) != nil
                }
                let r = frameValidation(frames: self.framesStore, limit: count,
                                        decode: decodeJPEG)
                self.log("Проверка кадров: \(r.checked) проверено, \(r.passed) ок, \(r.failed) провалов")
                for issue in r.issues.prefix(10) {
                    self.log("  \(issue)")
                }
                let obj: [String: Any] = [
                    "ok": true,
                    "total": self.framesStore.count,
                    "requested": r.requested,
                    "checked": r.checked,
                    "passed": r.passed,
                    "failed": r.failed,
                    "issues": Array(r.issues.prefix(20)),
                ]
                report = (try? String(data: JSONSerialization.data(withJSONObject: obj),
                                      encoding: .utf8)) ?? "{}"
            }
            return report
        }
    }

    /// Регистрирует screenshotProvider и layoutProvider.
    private func configureViewProviders() {
        api.screenshotProvider = { [weak self] in
            guard let self = self else { return nil }
            var path: String? = nil
            DispatchQueue.main.sync { path = self.makeScreenshot() }
            return path
        }
        api.layoutProvider = { [weak self] in
            guard let self = self else { return "{}" }
            var json = ""
            DispatchQueue.main.sync {
                var parts: [String] = []
                for (name, view) in self.layoutViews {
                    let r = view.convert(view.bounds, to: nil)
                    parts.append("\"\(name)\":{\"x\":\(Int(r.minX)),\"y\":\(Int(r.minY)),\"w\":\(Int(r.width)),\"h\":\(Int(r.height))}")
                }
                json = "{" + parts.joined(separator: ",") + "}"
            }
            return json
        }
        api.framesProvider = { [weak self] in
            guard let self = self else { return ["count":0,"names":[]] }
            var info: [String: Any] = [:]
            DispatchQueue.main.sync {
                info["count"] = self.framesStore.count
                info["names"] = self.framesStore.map { $0.name }
            }
            return info
        }
        api.paramsProvider = { [weak self] in
            guard let self = self else { return [:] }
            var dict: [String: Any] = [:]
            DispatchQueue.main.sync {
                dict["width"] = self.widthCtl.stringValue
                dict["height"] = self.heightCtl.stringValue
                dict["vaeWidth"] = self.widthField.stringValue
                dict["vaeHeight"] = self.heightField.stringValue
                dict["epochs"] = self.epochsCtl.stringValue
                dict["steps"] = self.stepsCtl.stringValue
                dict["batch"] = self.batchCtl.stringValue
                dict["lr"] = self.lrCtl.stringValue
                dict["extractLimit"] = self.extractLimitCtl.stringValue
                dict["modelWidth"] = self.requestedModelWidth
                dict["latentChannels"] = self.requestedLatentChannels
                dict["lossKind"] = self.requestedLossKind
                dict["edgeWeight"] = self.requestedEdgeWeight
                dict["residual"] = self.requestedResidual
                dict["groupNorm"] = self.requestedGroupNorm
                dict["resume"] = self.resumeSwitch.state == .on
                dict["adaptiveThrottle"] = self.adaptiveSwitch.state == .on
            }
            return dict
        }
        api.frameProvider = { [weak self] idx in
            guard let self = self else { return nil }
            var result: [String: Any]? = nil
            DispatchQueue.main.sync {
                guard idx >= 0 && idx < self.framesStore.count else { result = nil; return }
                let f = self.framesStore[idx]
                let b64 = f.jpeg.base64EncodedString()
                result = ["index": idx, "name": f.name, "source": f.source, "jpegBase64": b64]
            }
            return result
        }
    }

    /// Регистрирует stateProvider: снимок состояния окна.
    private func configureStateProvider() {
        api.stateProvider = { [weak self] in
            guard let self = self else { return "{}" }
            var json = ""
            DispatchQueue.main.sync {
                let dict: [String: Any] = [
                    "windowVisible": self.window?.isVisible ?? false,
                    "windowMiniaturized": self.window?.isMiniaturized ?? false,
                    "contentSubviews": self.window?.contentView?.subviews.count ?? 0,
                    "selectedVideos": self.videoURLs.map { $0.path },
                    "framesCount": self.framesStore.count,
                    "processedCount": self.processedFrames.filter { $0 }.count,
                    "extractLimit": self.extractLimitCtl.stringValue,
                    "epochs": self.epochsCtl.stringValue,
                    "steps": self.stepsCtl.stringValue,
                    "batch": self.batchCtl.stringValue,
                    "width": self.widthCtl.stringValue,
                    "height": self.heightCtl.stringValue,
                    "lr": self.lrCtl.stringValue,
                    "pauseMs": self.pauseCtl.stringValue,
                    "gpuPowerPercent": self.powerCtl.stringValue,
                    "resume": self.resumeSwitch.state == .on,
                    "adaptiveThrottle": self.adaptiveSwitch.state == .on,
                    "training": self.isTraining,
                    "extracting": self.isExtracting,
                    "thermal": self.thermalStateName(),
                    "gpuBufferAllocations": MetalEngine.shared.totalBufferAllocations,
                    "gpuStagingAllocations": MetalEngine.shared.totalStagingAllocations,
                ]
                if let data = try? JSONSerialization.data(withJSONObject: dict),
                   let s = String(data: data, encoding: .utf8) {
                    json = s
                } else {
                    json = "{}"
                }
            }
            return json
        }
    }

    /// Настраивает окно, контент и обсервер ресайза.
    private func configureWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1360, height: 840),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.backgroundColor = .windowBackgroundColor
        window.appearance = NSAppearance(named: .aqua)
        window.title = "YumeLayers"
        window.subtitle = "Студия обучения"
        window.titlebarAppearsTransparent = true
        window.toolbarStyle = .unified
        
        window.minSize = NSSize(width: 1280, height: 780)
        window.isReleasedWhenClosed = false
        window.contentView = buildUI()
        // Диагностика запуска: если buildUI() оборвался (например, исключение
        // в NSGridView), окно останется пустым — эта строка сразу это покажет
        // в журнале и в /api/log.
        log("Интерфейс собран: контейнеров \(window.contentView?.subviews.count ?? 0), элементов учёта \(layoutViews.count)")
        maximizeOnLaunch()
        showMainWindow()
        // Пагинатор: при изменении ширины окна пересчитываем видимое число
        // кружков от ширины центральной колонки. Ширина колонки при этом
        // не меняется — меняется только количество показанных страниц.
        NotificationCenter.default.addObserver(
            forName: NSWindow.didResizeNotification, object: window, queue: .main
        ) { [weak self] _ in
            guard let self = self else { return }
            let pages = max(1, (self.framesStore.count + self.pageSize - 1) / self.pageSize)
            self.renderPageDots(pages: pages)
        }
        log("Готово. Добавьте видео → извлеките кадры → начните обучение.")
    }

}
