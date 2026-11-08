import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

extension YumeAppDelegate {
    func setReconsProgress(_ p: Double) {
        let clamped = min(max(0, p), 1)
        reconsProgressBar.doubleValue = clamped
        reconsPercentLabel.stringValue = String(format: "%d%%", Int(clamped * 100))
    }

    func updateETALabel(_ label: NSTextField, elapsed: TimeInterval, done: Int, total: Int) {
        guard done > 0, total > 0 else {
            label.stringValue = "—"
            return
        }
        let remaining = elapsed / Double(done) * Double(max(0, total - done))
        label.stringValue = "≈ " + formatDuration(remaining)
    }

    func formatDuration(_ seconds: TimeInterval) -> String {
        let s = Int(seconds.rounded())
        if s < 60 { return "\(s) с" }
        let m = s / 60
        if m < 60 { return "\(m) мин" }
        return "\(m / 60) ч \(m % 60) мин"
    }

    func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    func renderVideoList() {
        for v in videoListStack.arrangedSubviews { videoListStack.removeArrangedSubview(v); v.removeFromSuperview() }
        for (i, url) in videoURLs.enumerated() {
            let name = NSTextField(labelWithString: url.lastPathComponent)
            name.font = NSFont.systemFont(ofSize: 11)
            name.textColor = .secondaryLabelColor
            name.lineBreakMode = .byTruncatingMiddle
            name.setContentHuggingPriority(.defaultLow, for: .horizontal)

            let del = NSButton(title: "", target: self, action: #selector(removeVideo(_:)))
            del.tag = i
            del.isBordered = false
            del.image = NSImage(systemSymbolName: "xmark.circle.fill", accessibilityDescription: "Удалить из списка")
            del.contentTintColor = .tertiaryLabelColor
            del.widthAnchor.constraint(equalToConstant: 20).isActive = true
            del.heightAnchor.constraint(equalToConstant: 20).isActive = true

            let row = NSStackView(views: [name, del])
            row.orientation = .horizontal
            row.spacing = 8
            row.alignment = .centerY
            videoListStack.addArrangedSubview(row)
        }
    }

    @objc func removeVideo(_ sender: NSButton) {
        guard sender.tag >= 0, sender.tag < videoURLs.count else { return }
        videoURLs.remove(at: sender.tag)
        saveSelection()
        if videoURLs.isEmpty {
            folderField.stringValue = ""
            extractButton.isEnabled = false
            videoEmptyState.isHidden = false
        } else {
            folderField.stringValue = videoURLs.count == 1
                ? videoURLs[0].path
                : "Выбрано MP4: \(videoURLs.count) (порядок сохранится)"
            selectVideo(videoURLs[0])
        }
        renderVideoList()
    }

    func selectVideo(_ url: URL) {
        videoEmptyState.isHidden = true
        looper?.disableLooping()
        queuePlayer.pause()
        queuePlayer.removeAllItems()
        let item = AVPlayerItem(url: url)
        looper = AVPlayerLooper(player: queuePlayer, templateItem: item)
        // Без автозапуска: воспроизведение — по кнопке плеера.
        setStatus("Видео выбрано. Нажмите «Вытащить кадры»", ok: true)
    }

    func saveSelection() {
        let paths = videoURLs.map { $0.path }
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("YumeLayers", isDirectory: true)
            .appendingPathComponent("last_video.json")
        if let data = try? JSONSerialization.data(withJSONObject: ["videos": paths]) {
            try? data.write(to: url)
        }
    }

    func restoreSelection() {
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("YumeLayers", isDirectory: true)
            .appendingPathComponent("last_video.json")
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let arr = obj["videos"] as? [String], !arr.isEmpty else { return }
        let urls = arr.map { URL(fileURLWithPath: $0) }
        setSelectedVideos(urls)
    }

    @objc func extractFrames() {
        guard !videoURLs.isEmpty else { return }
        // Лимит для теста: 0 — все кадры, N — первые N кадров подряд.
        let limit = max(0, extractLimitCtl.intValue(def: 0))
        // Если в редакторе уже есть кадры — спрашиваем: обнулить или добавить.
        if framesStore.isEmpty {
            startExtraction(urls: videoURLs, append: false, maxFrames: limit)
            return
        }
        let alert = NSAlert()
        alert.messageText = "В редакторе уже есть кадры (\(framesStore.count))"
        alert.informativeText = "Обнулить и извлечь заново или добавить новые кадры к текущим?"
        alert.addButton(withTitle: "Добавить к текущим")
        alert.addButton(withTitle: "Обнулить")
        alert.addButton(withTitle: "Отмена")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            startExtraction(urls: videoURLs, append: true, maxFrames: limit)
        case .alertSecondButtonReturn:
            startExtraction(urls: videoURLs, append: false, maxFrames: limit)
        default:
            break
        }
    }

    @objc func stopExtraction() {
        guard isExtracting else { return }
        extractCancelled = true
        currentExtractor?.cancel()
        setStatus("Остановка извлечения…", ok: false)
        log("=== Остановка извлечения ===")
    }

    @objc func carouselPagePrev() { changeCarouselPage(-1) }
    @objc func carouselPageNext() { changeCarouselPage(1) }
    @objc func carouselPageFirst() {
        let pages = max(1, (framesStore.count + pageSize - 1) / pageSize)
        carouselPage = edgePage(current: carouselPage, pages: pages, toStart: true)
        selectedFrameIndex = carouselPage * pageSize
        renderCollagePage()
    }

    @objc func carouselPageLast() {
        let pages = max(1, (framesStore.count + pageSize - 1) / pageSize)
        carouselPage = edgePage(current: carouselPage, pages: pages, toStart: false)
        selectedFrameIndex = carouselPage * pageSize
        renderCollagePage()
    }

    @objc func sourceSwitchChanged(_ sender: NSSegmentedControl) {
        showRecon = sender.selectedSegment == 1
        renderCollagePage()
    }

    func changeCarouselPage(_ d: Int) {
        let total = framesStore.count
        let pages = max(1, (total + pageSize - 1) / pageSize)
        carouselPage = min(max(0, carouselPage + d), pages - 1)
        selectedFrameIndex = carouselPage * pageSize
        renderCollagePage()
    }

    @objc func pageDotClicked(_ sender: NSButton) {
        let pages = max(1, (framesStore.count + pageSize - 1) / pageSize)
        let visible = paginatorVisiblePages(current: carouselPage, pages: pages, limit: fittedDotCount())
        guard sender.tag >= 0, sender.tag < visible.count else { return }
        carouselPage = visible[sender.tag]
        selectedFrameIndex = carouselPage * pageSize
        renderCollagePage()
    }

    func fittedDotCount() -> Int {
        let dot = paginatorDotWidth(pageNumber: carouselPage + 1)
        let rowWidth = paginatorRowWidth(columnWidth: Double(centerColumnWidth()))
        let n = dotCapacity(columnWidth: rowWidth, dotWidth: dot)
        return max(1, min(pageDots.count, n))
    }

    func centerColumnWidth() -> CGFloat {
        let content = window?.contentView?.bounds.width ?? 1360
        return max(320, (content - 442) / 2)
    }

    func renderPageDots(pages: Int) {
        let limit = fittedDotCount()
        let visible = paginatorVisiblePages(current: carouselPage, pages: pages, limit: limit)
        for (i, b) in pageDots.enumerated() {
            if i < visible.count {
                let page = visible[i]
                b.isHidden = false
                b.isEnabled = true
                let active = page == carouselPage
                let label = "\(page + 1)"
                // Ширина овала — из той же чистой функции, по которой
                // считается вместимость: цифры номера не обрезаются.
                let width = CGFloat(paginatorDotWidth(pageNumber: page + 1))
                if i < dotWidthConstraints.count {
                    dotWidthConstraints[i].constant = width
                }
                b.layer?.cornerRadius = 14   // овал: в 28 pt — круг, шире — овал
                b.layer?.backgroundColor = active
                    ? NSColor.systemBlue.cgColor
                    : NSColor.clear.cgColor
                let name: NSColor = active ? .white : .secondaryLabelColor
                b.attributedTitle = NSAttributedString(string: label,
                                                       attributes: [.foregroundColor: name])
            } else {
                b.isHidden = true
                b.isEnabled = false
                b.layer?.backgroundColor = NSColor.clear.cgColor
            }
        }
    }

    func startExtraction(urls: [URL], append: Bool, maxFrames: Int = 0) {
        guard !isExtracting, !isTraining, !isReconstructing, !urls.isEmpty else { return }
        isExtracting = true
        extractCancelled = false
        api.setExtractingActive(true)
        extractButton.isEnabled = false
        stopExtractButton.isEnabled = true
        trainButton.isEnabled = false
        // Прогресс — рядом с кнопкой «Вытащить кадры».
        extractProgressBlock.isHidden = false
        extractProgressBar.doubleValue = 0
        extractPercentLabel.stringValue = "0%"
        extractEtaLabel.stringValue = "…"
        setStatus("Извлечение кадров…", ok: true)
        if !append { clearCarousel() }
        let initialFrameCount = framesStore.count
        extractedCount = 0
        skippedBlackTotal = 0
        extractStartTime = Date()
        resetExtractBatch()
        startExtractFlushTimer()
        log("=== Извлечение кадров ===")
        for (i, u) in urls.enumerated() {
            log("Видео \(i + 1)/\(urls.count): \(u.lastPathComponent)")
        }

        log("extract: urls=\(urls.count), append=\(append), maxFrames=\(maxFrames) (первые N кадров)")

        DispatchQueue.global(qos: .userInitiated).async {
            // Лимит — общий на всю операцию: раздаём его видео по порядку.
            // 0 означает «все кадры», поэтому отдельный массив нужен, чтобы
            // после исчерпания лимита не запускать оставшиеся видео.
            var videoLimits: [Int] = []
            if maxFrames > 0 {
                var remaining = maxFrames
                for u in urls {
                    guard remaining > 0 else { videoLimits.append(0); continue }
                    let count = FrameExtractor(
                        url: u,
                        options: FrameExtractor.Options(maxFrames: remaining)
                    ).estimatedCount()
                    let take = min(remaining, count)
                    videoLimits.append(take)
                    remaining -= take
                }
            } else {
                videoLimits = urls.map { FrameExtractor(url: $0).estimatedCount() }
            }
            let totalAll = videoLimits.reduce(0, +)
            guard totalAll > 0 else {
                DispatchQueue.main.async {
                    self.finishExtraction(.failure(FrameExtractError.unreadableAsset))
                }
                return
            }
            DispatchQueue.main.async { self.extractTotalAll = totalAll }

            let sem = DispatchSemaphore(value: 0)
            var lastResult: Result<Void, Error> = .success(())
            var videoIndex = 0
            var globalDone = 0
            // При добавлении продолжаем нумерацию с текущего количества кадров.
            var startIndex = initialFrameCount

            for (i, u) in urls.enumerated() {
                if DispatchQueue.main.sync(execute: { self.extractCancelled }) {
                    lastResult = .failure(FrameExtractError.cancelled); break
                }
                let videoLimit = maxFrames > 0 ? videoLimits[i] : 0
                if maxFrames > 0 && videoLimit <= 0 { continue }
                videoIndex = i
                DispatchQueue.main.async {
                    self.setStatus("Извлечение: видео \(i + 1)/\(urls.count)", ok: true)
                }
                let extractor = FrameExtractor(url: u, options: FrameExtractor.Options(maxFrames: videoLimit, startIndex: startIndex))
                DispatchQueue.main.sync {
                    self.currentExtractor = extractor
                    if self.extractCancelled { extractor.cancel() }
                }
                // 1 кадр в секунду видео = каждый round(fps)-й кадр этого видео.
                let fps = extractor.frameRate()
                extractor.run(
                    onFrame: { [weak self] idx, _, ordinal, jpeg in
                        guard let self = self else { return }
                        globalDone += 1
                        // Только складываем в буфер: интерфейс забирает
                        // кадры пачками по таймеру, а не на каждый кадр.
                        let name = String(format: "frame_%04d.jpg", ordinal)
                        self.enqueueExtractedFrame(
                            name: name, jpeg: jpeg, source: u.lastPathComponent,
                            // Контрольная выборка: первый кадр каждой секунды видео.
                            isControl: isControlFrame(index: idx, fps: fps))
                    },
                    progress: { [weak self] p in
                        self?.storeExtractProgress(p, done: Int(p * Double(totalAll)))
                    },
                    completion: { result in
                        DispatchQueue.main.async { self.skippedBlackTotal += extractor.skippedLeadingBlack }
                        lastResult = result
                        sem.signal()
                    })
                sem.wait()
                if case .failure = lastResult { break }
                startIndex = initialFrameCount + globalDone
            }

            _ = videoIndex
            DispatchQueue.main.async {
                self.finishExtraction(lastResult)
            }
        }
    }

    func finishExtraction(_ result: Result<Void, Error>) {
        stopExtractFlushTimer()
        // Забираем остаток буфера, чтобы ни один кадр не потерялся.
        flushExtractBatch(force: true)
        isExtracting = false
        currentExtractor = nil
        api.setExtractingActive(false)
        extractButton.isEnabled = true
        stopExtractButton.isEnabled = false
        trainButton.isEnabled = true
        // Прогресс извлечения закончен — результат остаётся в строке статуса.
        extractProgressBlock.isHidden = true
        switch result {
        case .success:
            setStatus("Извлечено кадров: \(extractedCount)", ok: true)
            log("=== Извлечение завершено: \(extractedCount) кадров в строгом порядке ===")
            log("Кадры хранятся в памяти (без записи на диск)")
            if skippedBlackTotal > 0 {
                log("Пропущено чёрных кадров в начале видео: \(skippedBlackTotal)")
            }
            autoTuneParams(frames: framesStore.count)
            // Остаёмся на первой странице: в коллаже первые 10 кадров,
            // остальные доступны через пагинацию.
            if !framesStore.isEmpty {
                carouselPage = 0
                selectedFrameIndex = min(pageSize - 1, framesStore.count - 1)
                renderCollagePage()
            }
        case .failure(FrameExtractError.cancelled):
            setStatus("Извлечение остановлено", ok: false)
            log("=== Извлечение остановлено: \(extractedCount) кадров ===")
        case .failure(let e):
            setStatus("Извлечение прервано", ok: false)
            log("Ошибка извлечения: \(e.localizedDescription)")
        }
    }

    func enqueueExtractedFrame(name: String, jpeg: Data, source: String,
                                       isControl: Bool) {
        pendingExtractLock.lock()
        pendingExtractFrames.append(PendingExtractFrame(name: name, jpeg: jpeg,
                                                        source: source,
                                                        isControl: isControl))
        pendingExtractLock.unlock()
    }

    func storeExtractProgress(_ p: Double, done: Int) {
        pendingExtractLock.lock()
        pendingExtractProgress = p
        pendingExtractDone = done
        pendingExtractLock.unlock()
    }

    func resetExtractBatch() {
        pendingExtractLock.lock()
        pendingExtractFrames.removeAll()
        pendingExtractProgress = 0
        pendingExtractDone = 0
        pendingExtractLock.unlock()
        extractTotalAll = 0
        lastExtractFlush = Date.distantPast
        flushedFirstExtractBatch = false
    }

    func startExtractFlushTimer() {
        stopExtractFlushTimer()
        lastExtractFlush = Date()
        let timer = Timer(timeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.flushExtractBatch()
        }
        RunLoop.main.add(timer, forMode: .common)
        extractFlushTimer = timer
    }

    func stopExtractFlushTimer() {
        extractFlushTimer?.invalidate()
        extractFlushTimer = nil
    }

    func flushExtractBatch(force: Bool = false) {
        let now = Date()
        let elapsedMs = now.timeIntervalSince(lastExtractFlush) * 1000

        pendingExtractLock.lock()
        let pending = pendingExtractFrames.count
        let due = force || extractBatch.shouldFlush(pending: pending,
                                                    sinceLastFlushMs: elapsedMs,
                                                    flushedFirstBatch: flushedFirstExtractBatch)
        var batch: [PendingExtractFrame] = []
        if due && pending > 0 {
            batch = pendingExtractFrames
            pendingExtractFrames.removeAll(keepingCapacity: true)
        }
        let progress = pendingExtractProgress
        let done = pendingExtractDone
        pendingExtractLock.unlock()

        if !batch.isEmpty {
            lastExtractFlush = now
            flushedFirstExtractBatch = true
            appendExtractedFrames(batch)
        }
        // Прогресс обновляем на тике таймера, а не на каждый кадр.
        setExtractProgress(progress, done: done, total: extractTotalAll)
        api.updateExtractProgress(progress)
    }

    func appendExtractedFrames(_ batch: [PendingExtractFrame]) {
        let totalBefore = framesStore.count
        for item in batch {
            framesStore.append(MemoryFrame(name: item.name, jpeg: item.jpeg,
                                           source: item.source))
            processedFrames.append(false)
            if item.isControl {
                controlFramesStore.append(MemoryFrame(name: item.name, jpeg: item.jpeg,
                                                      source: item.source))
            }
        }
        extractedCount += batch.count
        if selectedFrameIndex < 0 && !framesStore.isEmpty {
            selectedFrameIndex = 0
        }
        // Пока заполняется первая страница — показываем кадры по мере
        // поступления. Дальше только пагинация: остальные кадры грузятся
        // по клику, а не на каждый извлечённый кадр.
        if extractBatch.shouldRenderCollage(page: carouselPage,
                                            totalBefore: totalBefore,
                                            totalAfter: framesStore.count,
                                            pageSize: pageSize) {
            renderCollagePage()
        } else {
            updatePagination()
        }
    }

    func clearCarousel() {
        reconMemory.removeAll()
        framesStore.removeAll()
        controlFramesStore.removeAll()
        processedFrames.removeAll()
        selectedFrameIndex = -1
        carouselPage = 0
        renderCollagePage()
    }

}
