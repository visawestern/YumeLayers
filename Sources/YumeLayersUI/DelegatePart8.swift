import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

extension YumeAppDelegate {
    func buildRemaining(left: NSStackView) -> NSView {
        // ——— Центральная колонка: метрики, текущее видео, обрабатываемый кадр ———
        statusLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)

        let metricsCol = NSStackView(views: [epochCard, lossCard, psnrCard])
        metricsCol.orientation = .vertical
        metricsCol.spacing = 6
        metricsCol.alignment = .leading
        // Компактные метрики вертикально, справа от списка видео.
        for c in [epochCard, lossCard, psnrCard] {
            c.heightAnchor.constraint(equalToConstant: 50).isActive = true
            c.widthAnchor.constraint(equalToConstant: 136).isActive = true
        }

        // Карточка-медиа: фон, рамка, изображение + пустое состояние.
        // Высота задаётся снаружи (в процентах от центральной колонки).


        // Плеер для выбранного MP4: воспроизведение по центру, зациклено.
        playerView.player = queuePlayer
        playerView.controlsStyle = .minimal
        playerView.videoGravity = .resizeAspect
        playerView.wantsLayer = true
        playerView.layer?.cornerRadius = 10
        playerView.layer?.masksToBounds = true
        // Плеер НЕ имеет права диктовать размеры: он всегда вписывается в карточку.
        playerView.setContentHuggingPriority(.defaultLow, for: .horizontal)
        playerView.setContentHuggingPriority(.defaultLow, for: .vertical)
        playerView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        playerView.setContentCompressionResistancePriority(.defaultLow, for: .vertical)
        let videoEmpty = simpleEmpty("film", "Выберите видеофайл MP4 слева")
        videoEmptyState = videoEmpty
        let videoCard = mediaCard(playerView, videoEmpty)
        videoMediaCard = videoCard

        // Плеер для ГОТОВОГО видео (результат интерполяции).
        resultPlayerView.player = resultQueuePlayer
        resultPlayerView.controlsStyle = .minimal
        resultPlayerView.videoGravity = .resizeAspect
        resultPlayerView.wantsLayer = true
        resultPlayerView.layer?.cornerRadius = 10
        resultPlayerView.layer?.masksToBounds = true
        resultPlayerView.setContentHuggingPriority(.defaultLow, for: .horizontal)
        resultPlayerView.setContentHuggingPriority(.defaultLow, for: .vertical)
        resultPlayerView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        resultPlayerView.setContentCompressionResistancePriority(.defaultLow, for: .vertical)
        let resultEmpty = simpleEmpty("checkmark.video", "Готовое видео появится здесь после интерполяции")
        resultEmptyState = resultEmpty
        let resultCard = mediaCard(resultPlayerView, resultEmpty)
        resultMediaCard = resultCard
        resultCard.isHidden = true

        // ——— Коллаж кадров: большая + 9 вокруг (клик ставит кадр в центр) ———

        collageBig.imageScaling = .scaleProportionallyDown
        collageBig.wantsLayer = true
        collageBig.layer?.cornerRadius = 10
        collageBig.layer?.masksToBounds = true
        collageBig.layer?.borderWidth = 0.5
        collageBig.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.25).cgColor
        collageBig.setContentHuggingPriority(.defaultLow, for: .horizontal)
        collageBig.setContentHuggingPriority(.defaultLow, for: .vertical)
        collageBig.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        collageBig.setContentCompressionResistancePriority(.defaultLow, for: .vertical)
        collageBig.translatesAutoresizingMaskIntoConstraints = false

        collageBigBadge.image = NSImage(systemSymbolName: "checkmark.circle.fill", accessibilityDescription: nil)
        collageBigBadge.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 22, weight: .bold)
        collageBigBadge.contentTintColor = .systemGreen
        collageBigBadge.translatesAutoresizingMaskIntoConstraints = false
        collageBig.addSubview(collageBigBadge)
        NSLayoutConstraint.activate([
            collageBigBadge.topAnchor.constraint(equalTo: collageBig.topAnchor, constant: 4),
            collageBigBadge.trailingAnchor.constraint(equalTo: collageBig.trailingAnchor, constant: -4),
            collageBigBadge.widthAnchor.constraint(equalToConstant: 24),
            collageBigBadge.heightAnchor.constraint(equalToConstant: 24),
        ])

        // L-раскладка: нижний ряд (5) + правый столбец (5) = 10 кадров вокруг большой.
        // Порядок чтения: нижний ряд слева → справа, затем столбец снизу → вверх.
        collageThumbs = []
        var allBtns: [NSButton] = []
        for slot in 0..<10 {
            let btn = HoverThumbButton(title: "", target: self, action: #selector(thumbClicked(_:)))
            btn.tag = slot
            btn.isBordered = false
            btn.imagePosition = .imageOnly
            btn.imageScaling = .scaleProportionallyDown
            // Миниатюра не должна «держать» ширину картинки (854 px): иначе
            // ряд из 5 миниатюр растягивает всю центральную колонку и окно.
            btn.setContentHuggingPriority(.defaultLow, for: .horizontal)
            btn.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
            btn.setContentHuggingPriority(.defaultLow, for: .vertical)
            btn.setContentCompressionResistancePriority(.defaultLow, for: .vertical)
            btn.wantsLayer = true
            btn.layer?.cornerRadius = 6
            btn.layer?.masksToBounds = true
            btn.layer?.borderWidth = 0.5
            btn.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor
            btn.layer?.backgroundColor = NSColor.quaternaryLabelColor.withAlphaComponent(0.08).cgColor
            let badge = photoBadge()
            btn.addSubview(badge)
            NSLayoutConstraint.activate([
                badge.topAnchor.constraint(equalTo: btn.topAnchor, constant: 2),
                badge.trailingAnchor.constraint(equalTo: btn.trailingAnchor, constant: -2),
                badge.widthAnchor.constraint(equalToConstant: 14),
                badge.heightAnchor.constraint(equalToConstant: 14),
            ])
            collageThumbs.append(btn)
            thumbBadges.append(badge)
            allBtns.append(btn)
        }
        // Правый столбец: слоты 0-4 (5 шт.), нижний ряд: слоты 5-9 (5 шт.).
        let vStack = NSStackView(views: (0..<5).map { allBtns[$0] })
        vStack.orientation = .vertical
        vStack.spacing = 2
        vStack.alignment = .trailing
        // Пять миниатюр делят высоту поровну — столбец совпадает с большим
        // кадром по высоте, без пустот сверху/снизу.
        vStack.distribution = .fillEqually
        let hStack = NSStackView(views: (5..<10).map { allBtns[$0] })
        hStack.orientation = .horizontal
        hStack.spacing = 2
        hStack.alignment = .centerY
        hStack.distribution = .fillEqually
        for i in 0..<5 {
            allBtns[i].widthAnchor.constraint(equalToConstant: 60).isActive = true
        }

        let collageCard = NSView()
        collageCard.wantsLayer = true
        collageCard.layer?.cornerRadius = 10
        collageCard.layer?.borderWidth = 0.5
        collageCard.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.45).cgColor
        collageCard.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        collageCard.addSubview(collageBig)
        collageCard.addSubview(vStack)
        collageCard.addSubview(hStack)

        collageEmpty.font = NSFont.systemFont(ofSize: 13)
        collageEmpty.textColor = .secondaryLabelColor
        collageEmpty.alignment = .center
        collageEmpty.translatesAutoresizingMaskIntoConstraints = false
        collageCard.addSubview(collageEmpty)
        NSLayoutConstraint.activate([
            collageEmpty.centerXAnchor.constraint(equalTo: collageCard.centerXAnchor),
            collageEmpty.centerYAnchor.constraint(equalTo: collageCard.centerYAnchor),
        ])

        // Пагинатор: ◀ [1] [2] ... ▶ — число кружочков динамическое,
        // считается по фактической ширине средней колонки.
        var dots: [NSButton] = []
        for i in 0..<dotPoolSize {
            let b = NSButton(title: "", target: self, action: #selector(pageDotClicked(_:)))
            b.tag = i
            b.isBordered = false
            b.wantsLayer = true
            b.layer?.cornerRadius = 14
            b.font = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .semibold)
            let wc = b.widthAnchor.constraint(equalToConstant: 28)
            wc.isActive = true
            b.heightAnchor.constraint(equalToConstant: 28).isActive = true
            // Кружок не должен распирать строку: видимое число считается по
            // ширине колонки, лишние кружки скрываются (см. fittedDotCount).
            b.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
            dotWidthConstraints.append(wc)
            dots.append(b)
        }
        pageDots = dots
        for b in [carouselFirst, carouselPrev, carouselNext, carouselLast] {
            b.target = self
            b.action = (b === carouselFirst) ? #selector(carouselPageFirst)
                      : (b === carouselLast) ? #selector(carouselPageLast)
                      : (b === carouselPrev) ? #selector(carouselPagePrev)
                      : #selector(carouselPageNext)
            b.bezelStyle = .rounded
            b.controlSize = .small
            b.contentTintColor = .secondaryLabelColor
            b.widthAnchor.constraint(equalToConstant: 28).isActive = true
            b.heightAnchor.constraint(equalToConstant: 32).isActive = true
            b.isEnabled = false
        }
        paginatorRow.orientation = .horizontal
        paginatorRow.spacing = 4
        paginatorRow.alignment = .centerY
        // Растягиваем пагинацию на всю ширину средней колонки:
        // стрелки по краям (в начало/в конец + ◀/▶), кружочки равномерно между ними.
        paginatorRow.distribution = .equalSpacing
        for v in [carouselFirst, carouselPrev] + pageDots + [carouselNext, carouselLast] {
            paginatorRow.addArrangedSubview(v)
        }

        NSLayoutConstraint.activate([
            // Большая — слева, упирается в нижний ряд; справа — столбец из 5.
            collageBig.leadingAnchor.constraint(equalTo: collageCard.leadingAnchor, constant: 8),
            collageBig.topAnchor.constraint(equalTo: collageCard.topAnchor, constant: 8),
            collageBig.trailingAnchor.constraint(equalTo: vStack.leadingAnchor, constant: -6),
            collageBig.bottomAnchor.constraint(equalTo: hStack.topAnchor, constant: -8),
            // Правый столбец из 5 мелких — ровно по высоте большого кадра.
            vStack.trailingAnchor.constraint(equalTo: collageCard.trailingAnchor, constant: -8),
            vStack.topAnchor.constraint(equalTo: collageBig.topAnchor),
            vStack.bottomAnchor.constraint(equalTo: collageBig.bottomAnchor),
            vStack.widthAnchor.constraint(equalToConstant: 60),
            // Нижний ряд из 5 кадров: высота следует из пропорций кадра (16:9),
            // ширины делятся поровну — ряд ровно по ширине карточки.
            // Ширина одной миниатюры = (ширина ряда − 4 зазора) / 5,
            // её высота = ширина × (480/854).
            hStack.leadingAnchor.constraint(equalTo: collageCard.leadingAnchor, constant: 10),
            hStack.bottomAnchor.constraint(equalTo: collageCard.bottomAnchor, constant: -10),
            hStack.heightAnchor.constraint(equalTo: hStack.widthAnchor,
                                           multiplier: (480.0 / 854.0) / 5.0,
                                           constant: -(4 * 2.0) * (480.0 / 854.0) / 5.0),
            hStack.trailingAnchor.constraint(equalTo: vStack.leadingAnchor, constant: -6),
        ])
        // Нижние миниатюры по высоте равны нижнему ряду.
        for i in 5..<10 {
            allBtns[i].heightAnchor.constraint(equalTo: hStack.heightAnchor).isActive = true
        }

        let videoCaption = sectionHeader("Текущее видео")

        // Шапка коллажа: заголовок + переключатель Оригинал/Реконструкция.
        // По умолчанию — ОРИГИНАЛ (реконструкция только по явному выбору).
        // Чип в конце строки показывает, сколько кадров сейчас в очереди.
        let headerSpacer = NSView()
        headerSpacer.setContentHuggingPriority(NSLayoutConstraint.Priority(1), for: .horizontal)
        headerSpacer.setContentCompressionResistancePriority(NSLayoutConstraint.Priority(1), for: .horizontal)
        sourceSwitch.selectedSegment = 0
        sourceSwitch.target = self
        sourceSwitch.action = #selector(sourceSwitchChanged(_:))
        sourceSwitch.font = NSFont.systemFont(ofSize: 11)
        sourceSwitch.controlSize = .small
        let collageHeader = NSStackView(views: [sectionHeader("Кадры"), sourceSwitch, headerSpacer, framesChip])
        collageHeader.orientation = .horizontal
        collageHeader.spacing = 12
        collageHeader.alignment = .centerY


        // Медиа-зона: переключатель вкладок + две карточки поверх друг друга.
        mediaTabsControl.selectedSegment = 0
        mediaTabsControl.target = self
        mediaTabsControl.action = #selector(mediaTabChanged(_:))
        mediaTabsControl.font = NSFont.systemFont(ofSize: 12)
        mediaTabsControl.controlSize = .small

        let mediaZone = NSView()
        mediaZone.translatesAutoresizingMaskIntoConstraints = false
        mediaZone.addSubview(videoCard)
        mediaZone.addSubview(resultCard)
        videoCard.translatesAutoresizingMaskIntoConstraints = false
        resultCard.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            videoCard.topAnchor.constraint(equalTo: mediaZone.topAnchor),
            videoCard.bottomAnchor.constraint(equalTo: mediaZone.bottomAnchor),
            videoCard.leadingAnchor.constraint(equalTo: mediaZone.leadingAnchor),
            videoCard.trailingAnchor.constraint(equalTo: mediaZone.trailingAnchor),
            resultCard.topAnchor.constraint(equalTo: mediaZone.topAnchor),
            resultCard.bottomAnchor.constraint(equalTo: mediaZone.bottomAnchor),
            resultCard.leadingAnchor.constraint(equalTo: mediaZone.leadingAnchor),
            resultCard.trailingAnchor.constraint(equalTo: mediaZone.trailingAnchor),
        ])

        let center = NSStackView(views: [mediaTabsControl, mediaZone, paginatorRow, collageHeader, collageCard])
        center.orientation = .vertical
        center.spacing = 10
        center.alignment = .leading
        center.edgeInsets = NSEdgeInsets(top: 24, left: 24, bottom: 24, right: 24)
        // Колонка должна сжиматься под ширину окна — иначе содержимое
        // (картинки кадров) распирает окно за пределы экрана.
        center.setContentHuggingPriority(.defaultLow, for: .horizontal)
        center.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        // Ширины медиа-блоков — строго в пределах центральной колонки.
        let centerInset = -2 * CGFloat(paginatorSideInset)
        collageHeader.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        mediaZone.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        mediaTabsControl.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        collageCard.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        paginatorRow.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        paginatorRow.heightAnchor.constraint(equalToConstant: 28).isActive = true

        // Вертикальные пропорции: медиа 50%, коллаж 45% зоны.
        mediaZone.heightAnchor.constraint(equalTo: center.heightAnchor, multiplier: 0.43, constant: -38).isActive = true
        collageCard.heightAnchor.constraint(equalTo: center.heightAnchor, multiplier: 0.57, constant: -118).isActive = true

        // ——— Правая колонка: журнал происходящего ———
        chart.wantsLayer = true
        // График тянется на всю свободную высоту блока — без пустой полосы
        // между прогрессом и графиком.
        chart.heightAnchor.constraint(greaterThanOrEqualToConstant: 120).isActive = true
        chart.setContentHuggingPriority(.defaultLow, for: .vertical)
        chart.setContentCompressionResistancePriority(.defaultLow, for: .vertical)

        let chartCaption = NSTextField(labelWithString: "Качество · PSNR    Ошибка · loss")
        chartCaption.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        chartCaption.textColor = .secondaryLabelColor

        logView.isEditable = false
        logView.isSelectable = true
        logView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.textColor = .secondaryLabelColor
        logView.drawsBackground = false
        logView.isVerticallyResizable = true
        logView.isHorizontallyResizable = false
        logView.textContainer?.widthTracksTextView = true
        logView.textContainerInset = NSSize(width: 10, height: 10)
        logView.autoresizingMask = [.width, .height]
        let scroll = NSScrollView()
        scroll.documentView = logView
        scroll.hasVerticalScroller = true
        scroll.borderType = .noBorder
        scroll.drawsBackground = false
        scroll.wantsLayer = true
        scroll.layer?.cornerRadius = 10
        scroll.layer?.borderWidth = 0.5
        scroll.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor

        let logCaption = NSTextField(labelWithString: "Журнал")
        logCaption.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        logCaption.textColor = .secondaryLabelColor

        let systemInfo = NSTextField(labelWithString: "")
        systemInfo.font = NSFont.systemFont(ofSize: 12)
        systemInfo.textColor = .tertiaryLabelColor
        systemInfo.stringValue = "GPU · \(MTLCreateSystemDefaultDevice()?.name ?? "недоступен")"

        // ——— Список выбранных видео (с удалением) ———
        let videosTitle = NSTextField(labelWithString: "Выбранные видео")
        videosTitle.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        videosTitle.textColor = .secondaryLabelColor

        videoListStack.orientation = .vertical
        videoListStack.spacing = 4
        videoListStack.alignment = .leading
        videoListStack.translatesAutoresizingMaskIntoConstraints = false
        videoListScroll.documentView = videoListStack
        videoListScroll.hasVerticalScroller = true
        videoListScroll.autohidesScrollers = true
        videoListScroll.borderType = .noBorder
        videoListScroll.drawsBackground = false
        videoListScroll.heightAnchor.constraint(equalToConstant: 88).isActive = true
        NSLayoutConstraint.activate([
            videoListStack.leadingAnchor.constraint(equalTo: videoListScroll.contentView.leadingAnchor),
            videoListStack.trailingAnchor.constraint(equalTo: videoListScroll.contentView.trailingAnchor),
            videoListStack.topAnchor.constraint(equalTo: videoListScroll.contentView.topAnchor),
        ])

        let leftTop = NSStackView(views: [statusLabel, systemInfo, videosTitle, videoListScroll, chartCaption, chart, logCaption])
        leftTop.orientation = .vertical
        leftTop.spacing = 10
        leftTop.alignment = .leading

        // Метрики — справа от левого блока, начинаются от самого верха колонки
        // (на одной линии со статусом «Извлечено кадров: …»).
        let topSplit = NSStackView(views: [leftTop, metricsCol])
        topSplit.orientation = .horizontal
        topSplit.spacing = 12
        topSplit.alignment = .top   // метрики не растягиваются — начинаются от топа
        topSplit.distribution = .fill

        // Журнал — на всю ширину колонки (вправо до конца) и растягивается
        // по вертикали до нижней границы, соблюдая отступы панели.
        scroll.setContentHuggingPriority(NSLayoutConstraint.Priority(1), for: .vertical)
        scroll.setContentCompressionResistancePriority(NSLayoutConstraint.Priority(250), for: .vertical)
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 120).isActive = true

        let rightPanel = NSStackView(views: [topSplit, scroll])
        rightPanel.orientation = .vertical
        rightPanel.spacing = 10
        rightPanel.alignment = .leading
        rightPanel.edgeInsets = NSEdgeInsets(top: 24, left: 24, bottom: 24, right: 24)
        // Правая колонка заполняет отведённую ей ширину (см. rightHost) и
        // сжимается вместе с окном.
        rightPanel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        rightPanel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        topSplit.setContentHuggingPriority(.defaultLow, for: .horizontal)
        topSplit.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        leftTop.setContentHuggingPriority(.defaultLow, for: .horizontal)
        leftTop.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        // ——— Разбиение на 3 вертикальные части с разделителями ———
        func fillHost(_ host: NSView, _ child: NSView) {
            host.addSubview(child)
            child.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                child.topAnchor.constraint(equalTo: host.topAnchor),
                child.bottomAnchor.constraint(equalTo: host.bottomAnchor),
                child.leadingAnchor.constraint(equalTo: host.leadingAnchor),
                child.trailingAnchor.constraint(equalTo: host.trailingAnchor),
            ])
        }

        // Левая колонка — только сверху, НЕ растягивается на всю высоту окна.
        let leftHost = NSScrollView()
        leftHost.hasVerticalScroller = true
        leftHost.autohidesScrollers = true
        leftHost.drawsBackground = false
        let leftDocument = FlippedView()
        leftHost.documentView = leftDocument
        leftDocument.translatesAutoresizingMaskIntoConstraints = false
        leftDocument.addSubview(left)
        left.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            left.topAnchor.constraint(equalTo: leftDocument.topAnchor),
            left.leadingAnchor.constraint(equalTo: leftDocument.leadingAnchor),
            left.trailingAnchor.constraint(equalTo: leftDocument.trailingAnchor),
            left.bottomAnchor.constraint(equalTo: leftDocument.bottomAnchor),
            leftDocument.leadingAnchor.constraint(equalTo: leftHost.contentView.leadingAnchor),
            leftDocument.trailingAnchor.constraint(equalTo: leftHost.contentView.trailingAnchor),
            leftDocument.topAnchor.constraint(equalTo: leftHost.contentView.topAnchor),
        ])
        let centerHost = NSView(); fillHost(centerHost, center)
        let rightHost = NSView(); fillHost(rightHost, rightPanel)

        let sepL = NSBox()
        sepL.boxType = .separator
        let sepR = NSBox()
        sepR.boxType = .separator

        // Точная ручная раскладка (вместо NSStackView, который не растягивал
        // правую колонку и оставлял дыру справа). Колонки прижаты друг к другу
        // и к краям окна: левая 33%, центр 33%, правая добирает ровно до
        // правого края — пустых мест не остаётся.
        let main = NSView()
        main.translatesAutoresizingMaskIntoConstraints = false
        leftHost.translatesAutoresizingMaskIntoConstraints = false
        centerHost.translatesAutoresizingMaskIntoConstraints = false
        rightHost.translatesAutoresizingMaskIntoConstraints = false
        sepL.translatesAutoresizingMaskIntoConstraints = false
        sepR.translatesAutoresizingMaskIntoConstraints = false
        for child in [leftHost, centerHost, rightHost, sepL, sepR] { main.addSubview(child) }
        NSLayoutConstraint.activate([
            // Левая колонка: слева сверху, ширина 33% окна.
            leftHost.leadingAnchor.constraint(equalTo: main.leadingAnchor),
            leftHost.topAnchor.constraint(equalTo: main.topAnchor),
            leftHost.bottomAnchor.constraint(equalTo: main.bottomAnchor),
            leftHost.widthAnchor.constraint(equalToConstant: 440),
            // Разделитель 1.
            sepL.leadingAnchor.constraint(equalTo: leftHost.trailingAnchor),
            sepL.widthAnchor.constraint(equalToConstant: 1),
            sepL.topAnchor.constraint(equalTo: main.topAnchor),
            sepL.bottomAnchor.constraint(equalTo: main.bottomAnchor),
            // Центр: ширина 33% окна.
            centerHost.leadingAnchor.constraint(equalTo: sepL.trailingAnchor),
            centerHost.widthAnchor.constraint(equalTo: rightHost.widthAnchor),
            centerHost.topAnchor.constraint(equalTo: main.topAnchor),
            centerHost.bottomAnchor.constraint(equalTo: main.bottomAnchor),
            // Разделитель 2.
            sepR.leadingAnchor.constraint(equalTo: centerHost.trailingAnchor),
            sepR.widthAnchor.constraint(equalToConstant: 1),
            sepR.topAnchor.constraint(equalTo: main.topAnchor),
            sepR.bottomAnchor.constraint(equalTo: main.bottomAnchor),
            // Правая колонка: от разделителя до правого края окна.
            rightHost.leadingAnchor.constraint(equalTo: sepR.trailingAnchor),
            rightHost.trailingAnchor.constraint(equalTo: main.trailingAnchor),
            rightHost.topAnchor.constraint(equalTo: main.topAnchor),
            rightHost.bottomAnchor.constraint(equalTo: main.bottomAnchor),
        ])

        let root = BackgroundView()
        root.addSubview(main)
        // Корневой view и раскладка не должны сопротивляться сжатию:
        // иначе содержимое распирает contentView шириной в тысячи пикселей.
        root.setContentHuggingPriority(.defaultLow, for: .horizontal)
        root.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        main.setContentHuggingPriority(.defaultLow, for: .horizontal)
        main.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        NSLayoutConstraint.activate([
            main.topAnchor.constraint(equalTo: root.topAnchor),
            main.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            main.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            main.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            topSplit.widthAnchor.constraint(equalTo: rightPanel.widthAnchor, constant: -48),
            leftTop.widthAnchor.constraint(equalTo: topSplit.widthAnchor, constant: -148),
            videoListScroll.widthAnchor.constraint(equalTo: leftTop.widthAnchor),
            chart.widthAnchor.constraint(equalTo: leftTop.widthAnchor),
            scroll.widthAnchor.constraint(equalTo: rightPanel.widthAnchor, constant: -48),
        ])
        layoutViews["center"] = center
        layoutViews["videoCard"] = videoCard
        layoutViews["collageCard"] = collageCard
        layoutViews["collageHeader"] = collageHeader
        layoutViews["sourceSwitch"] = sourceSwitch
        layoutViews["framesChip"] = framesChip
        layoutViews["collageVStack"] = vStack
        layoutViews["collageHStack"] = hStack
        layoutViews["paginatorRow"] = paginatorRow
        layoutViews["rightPanel"] = rightPanel
        layoutViews["videoListScroll"] = videoListScroll
        layoutViews["playerView"] = playerView
        layoutViews["logScroll"] = scroll
        layoutViews["logView"] = logView
        layoutViews["metricsCol"] = metricsCol
        layoutViews["epochCard"] = epochCard
        layoutViews["lossCard"] = lossCard
        layoutViews["psnrCard"] = psnrCard
        layoutViews["extractProgress"] = extractProgressBlock
        layoutViews["trainProgress"] = trainProgressBlock
        layoutViews["trainLive"] = trainLiveBlock
        layoutViews["reconsProgress"] = reconsProgressBlock
        layoutViews["chart"] = chart
        layoutViews["extractLimitCtl"] = extractLimitCtl
        layoutViews["epochsCtl"] = epochsCtl
        layoutViews["stepsCtl"] = stepsCtl
        layoutViews["batchCtl"] = batchCtl
        layoutViews["widthCtl"] = widthCtl
        layoutViews["heightCtl"] = heightCtl
        layoutViews["lrCtl"] = lrCtl
        layoutViews["pauseCtl"] = pauseCtl
        layoutViews["powerCtl"] = powerCtl
        layoutViews["root"] = root

        // Скрыть галочки и заполнить состояние коллажа при старте.
        renderCollagePage()
        for (i, g) in debugGrids.enumerated() {
            layoutViews["grid\(i)"] = g
        }

        setStatus("Готово к обучению", ok: true)
        return root
    }

}
