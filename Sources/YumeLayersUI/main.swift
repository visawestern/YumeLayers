// YumeLayers UI — профессиональная панель обучения на Swift/AppKit.
// Параметры задаются в окне; обучение на Metal GPU; защита от перегрева.
import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers

import YumeLayersCore

final class YumeAppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    // Поля параметров
    let folderField = NSTextField(string: "")
    let widthField = NSTextField(string: "854")
    let heightField = NSTextField(string: "480")
    let extractLimitCtl = EditableValueControl(value: "0")
    let epochsCtl = EditableValueControl(value: "20")
    let stepsCtl = EditableValueControl(value: "20")
    let batchCtl = EditableValueControl(value: "1")
    let lrCtl = EditableValueControl(value: "0.0003")
    let pauseCtl = EditableValueControl(value: "250")
    let powerCtl = EditableValueControl(value: "90")
    /// Размер кадра, на котором обучается VAE. Исходные JPEG остаются
    /// полноразмерными — декодер уменьшает их на лету (как старый стенд
    /// 80×56, на котором сеть сходилась заметно лучше 480p).
    let widthCtl = EditableValueControl(value: "80")
    let heightCtl = EditableValueControl(value: "56")
    let convPathCtl = EditableValueControl(value: "~/YumeLayers/checkpoints/conv.safetensors", width: 0)
    let resumeSwitch = NSSwitch()
    let adaptiveSwitch = NSSwitch()

    let trainButton = NSButton(title: "Обучить VAE", target: nil, action: nil)
    let stopButton = NSButton(title: "Стоп", target: nil, action: nil)
    let browseButton = NSButton(title: "Обзор", target: nil, action: nil)

    // Статус, прогресс, превью, лог
    let statusLabel = NSTextField(labelWithString: "Готово к обучению")
    // Прогресс каждого действия живёт рядом с его кнопкой (см. карточки 1 и 3).
    let extractProgressBar = NSProgressIndicator()
    let extractPercentLabel = NSTextField(labelWithString: "0%")
    let extractEtaLabel = NSTextField(labelWithString: "—")
    var extractProgressBlock: NSStackView!
    let trainProgressBar = NSProgressIndicator()
    let trainPercentLabel = NSTextField(labelWithString: "0%")
    let trainEtaLabel = NSTextField(labelWithString: "—")
    var trainProgressBlock: NSStackView!
    // Живой индикатор: спиннер + состояние + текущий кадр/шаг/скорость.
    let trainSpinner = NSProgressIndicator()
    let trainStateLabel = NSTextField(labelWithString: "Обучение идёт")
    let trainFrameLabel = NSTextField(labelWithString: "Ожидание первого шага…")
    var trainLiveBlock: NSStackView!
    var trainHeartbeatTimer: Timer?
    var lastTrainStepAt: Date?
    var lastTrainReport: TrainStepReport?
    let reconsProgressBar = NSProgressIndicator()
    let reconsPercentLabel = NSTextField(labelWithString: "0%")
    var reconsProgressBlock: NSStackView!
    let epochCard = MetricCard(title: "Эпоха", symbol: "clock")
    let lossCard = MetricCard(title: "Loss", symbol: "chart.line.downtrend.xyaxis")
    let psnrCard = MetricCard(title: "PSNR, дБ", symbol: "speedometer")
    let chart = HistoryChart()
    let previewView = NSImageView()
    let playerView = AVPlayerView()
    let queuePlayer = AVQueuePlayer()
    var looper: AVPlayerLooper?
    var videoEmptyState: NSStackView!
    var videoURLs: [URL] = []
    var isExtracting = false
    var currentExtractor: FrameExtractor?
    var extractCancelled = false
    var extractedCount = 0
    var skippedBlackTotal = 0
    let extractButton = NSButton(title: "Вытащить кадры", target: nil, action: nil)
    let stopExtractButton = NSButton(title: "Стоп", target: nil, action: nil)
    let resetParamsButton = NSButton(title: "Сброс к умолчаниям", target: nil, action: nil)
    let autoTuneButton = NSButton(title: "Авто-расчёт параметров", target: nil, action: nil)
    var extractStartTime = Date()
    var trainStartTime = Date()
    var lastCollageRender = Date.distantPast

    // Асинхронная выдача извлечённых кадров: фон складывает кадры в буфер,
    // main thread забирает их пачками по таймеру. Так на 44 000 кадров
    // не приходится 44 000 обновлений интерфейса.
    struct PendingExtractFrame {
        let name: String
        let jpeg: Data
        let source: String
        let isControl: Bool
    }
    let extractBatch = ExtractBatchPolicy()
    let pendingExtractLock = NSLock()
    var pendingExtractFrames: [PendingExtractFrame] = []
    var pendingExtractProgress: Double = 0
    var pendingExtractDone = 0
    var extractTotalAll = 0
    var extractFlushTimer: Timer?
    var lastExtractFlush = Date.distantPast
    var flushedFirstExtractBatch = false
    let carouselPrev = NSButton(image: NSImage(systemSymbolName: "chevron.left", accessibilityDescription: "Назад")!, target: nil, action: nil)
    let carouselNext = NSButton(image: NSImage(systemSymbolName: "chevron.right", accessibilityDescription: "Вперёд")!, target: nil, action: nil)
    let carouselFirst = NSButton(image: NSImage(systemSymbolName: "chevron.left.to.line", accessibilityDescription: "В начало")!, target: nil, action: nil)
    let carouselLast = NSButton(image: NSImage(systemSymbolName: "chevron.right.to.line", accessibilityDescription: "В конец")!, target: nil, action: nil)
    var pageDots: [NSButton] = []
    /// Ограничения ширины кружочков-страниц (обновляются под число цифр).
    var dotWidthConstraints: [NSLayoutConstraint] = []
    let dotPoolSize = 30   // запас кнопок; видимое число считается по ширине
    let paginatorRow = NSStackView()
    var framesStore: [MemoryFrame] = []
    /// Контрольная выборка (1 кадр/сек по каждому видео) — для визуальной
    /// оценки качества реконструкций на итоговой модели. Живёт отдельно и не
    /// освобождается по ходу обучения.
    var controlFramesStore: [MemoryFrame] = []
    var processedFrames: [Bool] = []
    /// Реконструкции в памяти: имя кадра → PNG-данные (без записи на диск).
    var reconMemory: [String: Data] = [:]
    var carouselPage = 0
    let pageSize = 10   // 5 внизу + 5 справа вокруг большой
    let collageBig = NSImageView()
    let collageBigBadge = NSImageView()
    var collageThumbs: [NSButton] = []
    var thumbBadges: [NSImageView] = []
    var selectedFrameIndex = -1
    let collageEmpty = NSTextField(labelWithString: "Кадры появятся после извлечения видео")
    let sourceSwitch = NSSegmentedControl(labels: ["Оригинал", "Реконструкция"], trackingMode: .selectOne, target: nil, action: nil)
    let framesChip = CountChip()
    var showRecon = false
    var epochsEditedByUser = false
    var stepsEditedByUser = false
    var lrEditedByUser = false
    let videoListScroll = NSScrollView()
    let videoListStack = NSStackView()

    /// Порядок кадров в L: нижний ряд 0-4 (слева → справа), затем правый
    /// столбец 5-9 (снизу → вверх, конец — правый верхний).
    let carouselStack = NSStackView()
    let carouselScroll = NSScrollView()
    var carouselPlaceholder: NSStackView!
    let logView = NSTextView()

    var isTraining = false
    /// Идут ли реконструкции в фоне (управляются кнопкой «Стоп»).
    var isReconstructing = false
    var lastEpoch = 0
    let api = ApiServer()
    var layoutViews: [String: NSView] = [:]
    var debugGrids: [NSView] = []
    var lastParams = Params()

    // ——— Генерация A→B: интерполяция между двумя опорными картинками ———
    let interpAButton = NSButton(title: "Выбрать…", target: nil, action: nil)
    let interpBButton = NSButton(title: "Выбрать…", target: nil, action: nil)
    let interpALabel = NSTextField(labelWithString: "не выбран")
    let interpBLabel = NSTextField(labelWithString: "не выбран")
    let interpSecondsCtl = EditableValueControl(value: "5")
    let interpFpsCtl = EditableValueControl(value: "24")
    let interpRunButton = NSButton(title: "Создать видео", target: nil, action: nil)
    let interpStopButton = NSButton(title: "Стоп", target: nil, action: nil)
    let interpProgressBar = NSProgressIndicator()
    let interpPercentLabel = NSTextField(labelWithString: "0%")
    var interpProgressBlock: NSStackView!
    let interpResultLabel = NSTextField(wrappingLabelWithString: "")
    var interpImageAPath = ""
    var interpImageBPath = ""
    var isInterpolating = false
    /// Чекпоинт, которым сделана последняя интерполяция.
    var lastInterpolationModel = ""

    // Плеер для ГОТОВОГО видео (результат интерполяции) + вкладки.
    let resultPlayerView = AVPlayerView()
    let resultQueuePlayer = AVQueuePlayer()
    var resultLooper: AVPlayerLooper?
    var resultEmptyState: NSStackView!
    let mediaTabsControl = NSSegmentedControl(
        labels: ["Входящее видео", "Готовое видео"],
        trackingMode: .selectOne, target: nil, action: nil)
    var videoMediaCard: NSView!
    var resultMediaCard: NSView!














    /// Разворачивает окно на весь рабочий стол при старте.
    ///
    /// Это НЕ нативный полноэкранный режим: меню-бар сверху и Dock остаются на
    /// месте, окно просто занимает `visibleFrame` — ровно то же, что даёт
    /// зелёная кнопка (или «Увеличить» в меню «Окно»). `visibleFrame` уже
    /// исключает полосу меню и Dock, поэтому окно ничего не перекрывает.


    // MARK: - Стандартное главное меню (HIG)


    // MARK: - Вспомогательное


    /// Строка формы, которая растягивается на всю ширину карточки:
    /// [надпись | ⓘ | …поле…] — поле занимает всё расстояние от кнопки
    /// info до правого края строки/колонки.
    final class StretchRow: NSStackView {}

    /// Строка формы: [надпись | ⓘ | …поле…] — кнопка info сразу после
    /// надписи, поле растягивается на всё расстояние до правого края.
    /// `info` — полное описание для выпадающей карточки.
    let labelW: CGFloat = 141
    let rowGap: CGFloat = 6
    let accW: CGFloat = 76
    let infoW: CGFloat = 16


    /// Строка с редактируемым значением: [надпись | ⓘ | …поле…].
    /// Кнопка info — сразу после надписи, значение/поле занимают всё
    /// расстояние от неё до правого края строки.


    /// Компактная строка переключателя (как в web-admin-panels): switch + лейбл.
    /// Лейбл — в той же колонке, что и у строк с числами, переключатель —
    /// в колонке значений (по правому краю). Единая сетка карточки.

    /// Блок пути: подпись с info на первой строке, поле — на отдельной
    /// строке во всю ширину карточки, чтобы длинный путь был виден целиком.

    // MARK: - Кнопка info с выпадающей карточкой-описанием
    var infoTexts: [String] = []
    var openPopovers: [Int: NSPopover] = [:]



    // MARK: - UI

    func buildUI() -> NSView {
        // ——— Левая колонка: параметры ———
        folderField.placeholderString = "Выберите видеофайл (MP4)"
        folderField.font = NSFont.systemFont(ofSize: 12)
        folderField.isEditable = false
        folderField.textColor = .secondaryLabelColor
        folderField.bezelStyle = .roundedBezel
        browseButton.target = self
        browseButton.action = #selector(chooseVideo)
        browseButton.bezelStyle = .rounded
        browseButton.image = NSImage(systemSymbolName: "film", accessibilityDescription: "Выбрать видеофайл")
        browseButton.imagePosition = .imageLeading
        browseButton.controlSize = .small

        // Карточка-группа в стиле Apple (группировка как в Vuexy/AdminLTE).
        // Раскладка — вручную через ограничения, без промежуточного стека:
        // так строки формы гарантированно растягиваются на всю ширину
        // карточки (никаких круговых зависимостей NSStackView).

        // Шаг 1 — Видео: выбор MP4 + извлечение кадров
        let left = buildLeftPanel()

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


    // MARK: - Действия

    /// Выбор одного или нескольких видеофайлов (MP4/MOV).

    /// Единая регистрация выбранных видео: список справа, поле «Файл:»,
    /// активность кнопки извлечения, показ первого видео. Используется и UI,
    /// и API — чтобы оба пути вели себя одинаково.

    /// Блок прогресса действия: [полоса | проценты | ETA]. Один вид для
    /// извлечения кадров, обучения и реконструкций. Ширина задаётся снаружи
    /// под карточку, полоса тянется на свободное место.

    /// Прогресс извлечения — под кнопкой «Вытащить кадры».

    /// Прогресс обучения — под кнопкой «Обучить VAE».

    /// Раз в секунду проверяем, живой ли тренировочный цикл: если отчётов
    /// о шагах давно нет, спиннер останавливается и строка честно говорит,
    /// сколько секунд нет прогресса.



    /// Прогресс реконструкций — отдельная строка под прогрессом обучения.

    /// Обновление прогноза времени (ETA) рядом с прогрессбаром действия.
    /// Считается линейно: (затраченное / сделано) × осталось.


    /// Текущий тепловой уровень macOS (для /api/state и контроля нагрева).

    /// Рендер списка выбранных видео в правой колонке (имя + удалить).

    /// Удаление видео из выбранного списка (кнопка × в правой колонке).








    /// Переключение страницы пагинатора (по 9 кадров: большая + 9 малых).

    /// Клик по кружочку-странице.

    /// Сколько кружочков реально влезает в строку пагинатора.
    /// Считаем от фактической ширины строки (колонка минус отступы 24+24),
    /// а не от ширины самой строки: иначе кружочки распирают строку, строка
    /// «видит» большую ширину и показывает ещё больше кружков — колонка растёт.

    /// Ширина центральной колонки из фактической ширины окна:
    /// (контент − левая колонка 440 − два разделителя по 1) / 2.

    /// Рендер кружочков-страниц: столько, сколько влезает по ширине,
    /// активная выделена. Ширина кружочка адаптивная: под число цифр
    /// (например «1000» не влезает в 28 pt — расширяем и делаем овал).

    /// Извлечение кадров из одного или нескольких MP4.
    /// Кадры нумеруются строго по порядку файлов (глобально), чтобы движение
    /// можно было привязать к seed'у.
    /// - `append`: false — очистить старые кадры; true — продолжить нумерацию с текущих.
    /// - `maxFrames`: лимит кадров (0 — авто от доступной памяти).

    /// Финализация извлечения: общий статус, кнопки, лог.

    /// Кладём кадр в буфер, который main thread забирает пачками.
    /// Вызывается из фонового потока извлечения — без обновления UI.



    /// Таймер на main thread забирает накопленные кадры пачками.
    /// 200 мс достаточно, чтобы первые кадры появились почти сразу,
    /// и при этом на 44 000 кадров не приходится 44 000 обновлений UI.


    /// Забирает пачку из буфера и добавляет кадры в интерфейс.
    /// - Parameter force: забрать всё, даже неполную пачку (финализация/остановка).

    /// Добавление пачки кадров в память. `processed` = true только когда кадр
    /// обработан моделью (тогда ставится галочка ✓).


    /// Путь к реконструкции того же кадра (recons/frame_XXXX.jpg).
    /// PNG-данные реконструкции кадра из памяти (nil, если ещё нет).

    /// Рендер коллажа: большая (выбранная) + 9 малых текущей страницы.

    /// Обновляет счётчик кадров и пагинацию, не трогая картинки.
    /// Используется во время извлечения: новые кадры сразу доступны
    /// по страницам, но коллаж не перерисовывается на каждый кадр.

    /// Клик по малому кадру — ставит его в центр коллажа.

    /// Рендер собственного окна в PNG (для диагностики интерфейса).


    /// Запуск обучения. `maxFrames` > 0 — учиться только на первых N кадрах
    /// (для тестов вместо полного датасета). 0 — все кадры.
    /// `resume` = true — продолжить с существующего чекпоинта без диалога
    /// (используется API; в UI решение принимает человек).

    /// Единая точка входа обучения: и кнопка «Обучить», и API вызывают её.
    /// Кадры для обучения всегда берутся из памяти (единый источник).
    /// `maxFrames` > 0 — обучаться только на первых N кадрах.


    /// Авто-настройка «Эпохи»/«Шагов за эпоху» под фактическое число кадров.
    /// Поле подстраивается, пока пользователь не подтвердил своё значение
    /// вручную: шаги = число кадров (полный проход за эпоху),
    /// эпохи = 3…10 полных проходов (около 3000 сэмплов суммарно).
    /// Повторный запуск извлечения пересчитывает авто-поля под новый размер очереди.

    /// Кнопка «Авто-расчёт параметров»: пересчёт под текущие кадры.




    /// Строка из core-кода (yumeLog): уже с меткой времени и уже записана в
    /// train.log — здесь только показываем её в журнале и отдаём через API.
}

let app = NSApplication.shared
app.appearance = NSAppearance(named: .aqua)
setbuf(stdout, nil)   // немедленный вывод логов API в терминал/файл
let delegate = YumeAppDelegate()
app.delegate = delegate
app.run()
