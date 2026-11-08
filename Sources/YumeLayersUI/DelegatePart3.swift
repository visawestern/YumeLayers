import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

extension YumeAppDelegate {
    @objc func showMainWindow() {
        guard let window = window else { return }
        if window.isMiniaturized { window.deminiaturize(nil) }
        window.makeKeyAndOrderFront(nil)
        // NSApp.activate(ignoringOtherApps:) на macOS 14+ ничего не делает
        // (deprecated, «will have no effect»). NSRunningApplication.activate
        // реально выводит приложение на передний план и переносит окно на
        // активный рабочий стол (Space), если сейчас открыт полноэкранный
        // другой приложение.
        NSRunningApplication.current.activate(options: [.activateAllWindows])
    }

    func maximizeOnLaunch() {
        guard let window = window else { return }
        guard let screen = window.screen ?? NSScreen.main else { return }
        window.setFrame(screen.visibleFrame, display: false)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showMainWindow()
        return true
    }

    func buildMenu() -> NSMenu {
        let mainMenu = NSMenu()

        // Приложение
        let appItem = NSMenuItem()
        mainMenu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "О YumeLayers",
                        action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Скрыть YumeLayers",
                        action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(withTitle: "Скрыть остальные",
                        action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        appMenu.addItem(withTitle: "Показать все",
                        action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Выйти из YumeLayers",
                        action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu

        // Правка (работает с полями ввода)
        let editItem = NSMenuItem()
        mainMenu.addItem(editItem)
        let editMenu = NSMenu(title: "Правка")
        editMenu.addItem(withTitle: "Отменить", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(withTitle: "Повторить", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Вырезать", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Копировать", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Вставить", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Выбрать всё", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu

        // Окно
        let windowItem = NSMenuItem()
        mainMenu.addItem(windowItem)
        let windowMenu = NSMenu(title: "Окно")
        windowMenu.addItem(withTitle: "Свернуть", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: "Увеличить", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        windowMenu.addItem(.separator())
        windowMenu.addItem(withTitle: "На передний план", action: #selector(NSApplication.arrangeInFront(_:)), keyEquivalent: "")
        windowMenu.addItem(withTitle: "Показать YumeLayers", action: #selector(showMainWindow), keyEquivalent: "0")
        NSApp.windowsMenu = windowMenu
        windowItem.submenu = windowMenu

        return mainMenu
    }

    func sectionHeader(_ text: String) -> NSTextField {
        let l = NSTextField(labelWithString: text)
        l.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        l.textColor = .labelColor
        return l
    }

    func fieldRow(_ label: String, _ field: NSTextField, tooltip: String? = nil, accessory: NSView? = nil, info: String? = nil) -> StretchRow {
        let l = NSTextField(labelWithString: label)
        l.font = NSFont.systemFont(ofSize: 12)
        l.textColor = .secondaryLabelColor
        l.alignment = .left
        l.lineBreakMode = .byTruncatingTail
        field.font = NSFont.monospacedDigitSystemFont(ofSize: 13, weight: .regular)
        field.alignment = .natural
        field.bezelStyle = .roundedBezel
        field.heightAnchor.constraint(equalToConstant: 26).isActive = true
        // Поле занимает всё расстояние от кнопки info до правого края строки.
        field.setContentHuggingPriority(.defaultLow, for: .horizontal)
        field.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        field.widthAnchor.constraint(greaterThanOrEqualToConstant: 120).isActive = true
        if let tip = tooltip { field.toolTip = tip }

        var views: [NSView] = [l]
        if let infoText = info {
            views.append(makeInfoButton(infoText))
        }
        views.append(field)
        if let acc = accessory {
            acc.widthAnchor.constraint(equalToConstant: accW).isActive = true
            acc.heightAnchor.constraint(equalToConstant: 26).isActive = true
            views.append(acc)
        }
        return makeStretchRow(views)
    }

    func valueRow(_ label: String, _ ctl: EditableValueControl, info: String? = nil) -> StretchRow {
        let l = NSTextField(labelWithString: label)
        l.font = NSFont.systemFont(ofSize: 12)
        l.textColor = .secondaryLabelColor
        l.alignment = .left
        l.lineBreakMode = .byTruncatingTail
        var views: [NSView] = [l]
        if let infoText = info {
            views.append(makeInfoButton(infoText))
        }
        // Разделитель с низким приоритетом: всё свободное расстояние
        // скапливается между кнопкой info и полем (justify-space-between),
        // а поле прижато к правому краю строки.
        let spacer = NSView()
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)
        spacer.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        views.append(spacer)
        views.append(ctl)
        return makeStretchRow(views)
    }

    func makeStretchRow(_ views: [NSView]) -> StretchRow {
        let row = StretchRow(views: views)
        row.orientation = .horizontal
        row.distribution = .fill
        row.alignment = .centerY
        row.spacing = rowGap
        row.heightAnchor.constraint(equalToConstant: 28).isActive = true
        // Надпись не ужимается: вся лишняя ширина уходит разделителю.
        if let l = views.first as? NSTextField {
            l.setContentHuggingPriority(.required, for: .horizontal)
            l.setContentCompressionResistancePriority(.required, for: .horizontal)
        }
        debugGrids.append(row)
        return row
    }

    func switchRow(_ label: String, _ control: NSSwitch,
                           tooltip: String? = nil) -> NSGridView {
        let l = NSTextField(labelWithString: label)
        l.font = NSFont.systemFont(ofSize: 12)
        l.textColor = .secondaryLabelColor
        l.alignment = .left
        l.lineBreakMode = .byTruncatingTail
        l.toolTip = tooltip
        control.toolTip = tooltip

        // Переключатель прижат к правому краю колонки значений — как в
        // системных настройках macOS.
        let holder = NSView()
        holder.translatesAutoresizingMaskIntoConstraints = false
        holder.addSubview(control)
        control.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            control.trailingAnchor.constraint(equalTo: holder.trailingAnchor),
            control.centerYAnchor.constraint(equalTo: holder.centerYAnchor),
            holder.widthAnchor.constraint(equalToConstant: 120),
        ])

        // Одна строка, две колонки: лейбл | переключатель. Индексы колонок
        // обязаны существовать — иначе NSGridView бросает исключение и
        // buildUI() обрывается, оставляя пустое окно.
        let g = NSGridView(views: [[l, holder]])
        g.column(at: 0).width = labelW
        g.column(at: 1).width = 120
        g.rowSpacing = 8
        g.columnSpacing = rowGap
        g.widthAnchor.constraint(equalToConstant: labelW + rowGap + 120).isActive = true
        return g
    }

    func pathBlock(_ label: String, _ ctl: EditableValueControl, info: String) -> NSStackView {
        let l = NSTextField(labelWithString: label)
        l.font = NSFont.systemFont(ofSize: 12)
        l.textColor = .secondaryLabelColor
        l.alignment = .left
        l.lineBreakMode = .byTruncatingTail

        let head = NSStackView(views: [l, makeInfoButton(info)])
        head.orientation = .horizontal
        head.spacing = rowGap
        head.alignment = .centerY

        let block = NSStackView(views: [head, ctl])
        block.orientation = .vertical
        block.alignment = .leading
        block.spacing = 6
        block.widthAnchor.constraint(equalToConstant: 360).isActive = true
        ctl.widthAnchor.constraint(equalToConstant: 360).isActive = true
        return block
    }

    func makeInfoButton(_ text: String) -> NSButton {
        let idx = infoTexts.count
        infoTexts.append(text)
        let b = NSButton(title: "", target: self, action: #selector(infoTapped(_:)))
        b.tag = idx
        b.isBordered = false
        b.image = NSImage(systemSymbolName: "info.circle", accessibilityDescription: "Подробнее")
        b.imagePosition = .imageOnly
        b.contentTintColor = .secondaryLabelColor
        b.toolTip = "Подробнее о параметре"
        b.widthAnchor.constraint(equalToConstant: infoW).isActive = true
        b.heightAnchor.constraint(equalToConstant: infoW).isActive = true
        return b
    }

    @objc func infoTapped(_ sender: NSButton) {
        let idx = sender.tag
        guard idx < infoTexts.count else { return }
        if let pop = openPopovers[idx], pop.isShown {
            pop.performClose(nil)
            openPopovers[idx] = nil
            return
        }
        let text = NSTextField(wrappingLabelWithString: infoTexts[idx])
        text.font = NSFont.systemFont(ofSize: 12)
        text.textColor = .labelColor
        text.widthAnchor.constraint(equalToConstant: 280).isActive = true

        let card = NSVisualEffectView()
        card.material = .popover
        card.blendingMode = .withinWindow
        card.state = .active
        card.wantsLayer = true
        card.layer?.cornerRadius = 8
        card.addSubview(text)
        text.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            text.topAnchor.constraint(equalTo: card.topAnchor, constant: 12),
            text.bottomAnchor.constraint(equalTo: card.bottomAnchor, constant: -12),
            text.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 12),
            text.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -12),
        ])

        let vc = NSViewController()
        vc.view = card
        let pop = NSPopover()
        pop.contentViewController = vc
        pop.behavior = .transient
        pop.show(relativeTo: sender.bounds, of: sender, preferredEdge: .maxY)
        openPopovers[idx] = pop
    }

    func setStatus(_ text: String, ok: Bool) {
        statusLabel.stringValue = text
        statusLabel.textColor = ok ? .labelColor : .systemRed
    }

    @objc func chooseVideo() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = true
        panel.prompt = "Выбрать"
        panel.allowedContentTypes = [.mpeg4Movie, .movie, .quickTimeMovie]
        guard panel.runModal() == .OK else { return }
        let urls = panel.urls
        guard !urls.isEmpty else { return }
        setSelectedVideos(urls)
        saveSelection()
    }

    func setSelectedVideos(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        videoURLs = urls   // порядок выбора сохраняется
        renderVideoList()
        if urls.count == 1 {
            folderField.stringValue = urls[0].path
        } else {
            folderField.stringValue = "Выбрано MP4: \(urls.count) (порядок сохранится)"
        }
        extractButton.isEnabled = true
        // Показ первого видео по центру, зациклено.
        selectVideo(urls[0])
    }

    func makeProgressBlock(bar: NSProgressIndicator,
                                   percent: NSTextField,
                                   eta: NSTextField?) -> NSStackView {
        bar.style = .bar
        bar.isIndeterminate = false
        bar.minValue = 0
        bar.maxValue = 1
        bar.doubleValue = 0
        bar.controlSize = .small
        bar.setContentHuggingPriority(.defaultLow, for: .horizontal)
        bar.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        percent.font = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .semibold)
        percent.textColor = .secondaryLabelColor
        percent.alignment = .right
        percent.widthAnchor.constraint(equalToConstant: 40).isActive = true

        var views: [NSView] = [bar, percent]
        if let eta = eta {
            eta.font = NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .regular)
            eta.textColor = .tertiaryLabelColor
            eta.alignment = .right
            eta.widthAnchor.constraint(equalToConstant: 64).isActive = true
            views.append(eta)
        }
        let row = NSStackView(views: views)
        row.orientation = .horizontal
        row.spacing = 8
        row.alignment = .centerY
        row.isHidden = true
        return row
    }

    func setExtractProgress(_ p: Double, done: Int, total: Int) {
        let clamped = min(max(0, p), 1)
        extractProgressBar.doubleValue = clamped
        extractPercentLabel.stringValue = String(format: "%d%%", Int(clamped * 100))
        updateETALabel(extractEtaLabel,
                       elapsed: Date().timeIntervalSince(extractStartTime),
                       done: done, total: total)
    }

    func setTrainProgress(_ p: Double, done: Int, total: Int) {
        let clamped = min(max(0, p), 1)
        trainProgressBar.doubleValue = clamped
        trainPercentLabel.stringValue = String(format: "%d%%", Int(clamped * 100))
        updateETALabel(trainEtaLabel,
                       elapsed: Date().timeIntervalSince(trainStartTime),
                       done: done, total: total)
    }

    func startTrainHeartbeat() {
        trainHeartbeatTimer?.invalidate()
        trainHeartbeatTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.refreshTrainLiveStatus()
        }
    }

    func stopTrainHeartbeat() {
        trainHeartbeatTimer?.invalidate()
        trainHeartbeatTimer = nil
    }

    func refreshTrainLiveStatus() {
        guard isTraining, let last = lastTrainStepAt else { return }
        let since = Date().timeIntervalSince(last)
        let phase = trainLivePhase(secondsSinceStep: since,
                                   stepSeconds: lastTrainReport?.secondsPerStep)
        let stage = lastTrainReport?.stage ?? .training
        trainStateLabel.stringValue = trainLiveStateText(phase: phase, stage: stage)
        switch phase {
        case .running:
            trainStateLabel.textColor = .labelColor
            trainSpinner.startAnimation(nil)
        case .waiting:
            trainStateLabel.textColor = .secondaryLabelColor
            trainSpinner.startAnimation(nil)
        case .stalled:
            trainStateLabel.textColor = .systemOrange
            trainSpinner.stopAnimation(nil)
        }
    }

}
