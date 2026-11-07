// YumeLayers UI — профессиональная панель обучения на Swift/AppKit.
// Параметры задаются в окне; обучение на Metal GPU; защита от перегрева.
import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers

final class FlippedView: NSView { override var isFlipped: Bool { true } }
final class BackgroundView: NSView {
    override func draw(_ dirtyRect: NSRect) {
        NSColor.windowBackgroundColor.setFill()
        dirtyRect.fill()
    }
}
import YumeLayersCore

// MARK: - Градиент-подложка окна (чтобы матовые панели «светились»)

final class GradientBackgroundView: NSView {
    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let light = NSColor.controlBackgroundColor
        let accent = NSColor.systemBlue.withAlphaComponent(0.18)
        let dark = NSColor.windowBackgroundColor
        let top = effectiveAppearance.name == .darkAqua ? dark : light
        let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                                  colors: [top.cgColor, accent.cgColor] as CFArray,
                                  locations: [0, 1])!
        ctx.drawLinearGradient(gradient,
                               start: CGPoint(x: 0, y: bounds.maxY),
                               end: CGPoint(x: bounds.maxX, y: 0),
                               options: [])
    }
}

// MARK: - Карточка метрики (паттерн «дашборд» из AppDisign-свидетельств)

final class MetricCard: NSVisualEffectView {
    private let iconView = NSImageView()
    private let titleLabel = NSTextField(labelWithString: "")
    private let valueLabel = NSTextField(labelWithString: "0")

    init(title: String, symbol: String) {
        super.init(frame: .zero)
        material = .popover
        blendingMode = .withinWindow
        state = .active
        wantsLayer = true
        layer?.cornerRadius = 8
        layer?.borderWidth = 0.5
        layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor

        iconView.image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)
        iconView.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 13, weight: .medium)
        iconView.contentTintColor = .secondaryLabelColor
        iconView.widthAnchor.constraint(equalToConstant: 18).isActive = true
        iconView.heightAnchor.constraint(equalToConstant: 18).isActive = true

        titleLabel.stringValue = title
        titleLabel.font = NSFont.systemFont(ofSize: 11)
        titleLabel.textColor = .secondaryLabelColor
        titleLabel.lineBreakMode = .byTruncatingTail

        valueLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 17, weight: .medium)
        valueLabel.textColor = .labelColor
        valueLabel.lineBreakMode = .byTruncatingTail

        let textStack = NSStackView(views: [titleLabel, valueLabel])
        textStack.orientation = .vertical
        textStack.alignment = .leading
        textStack.spacing = 1

        let row = NSStackView(views: [iconView, textStack])
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 6
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        NSLayoutConstraint.activate([
            row.topAnchor.constraint(equalTo: topAnchor, constant: 8),
            row.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -8),
            row.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            row.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            heightAnchor.constraint(greaterThanOrEqualToConstant: 48),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func setValue(_ s: String) {
        valueLabel.stringValue = s
    }
}

// MARK: - Чип со счётчиком кадров

final class CountChip: NSView {
    private let label = NSTextField(labelWithString: "")

    init() {
        super.init(frame: .zero)
        wantsLayer = true
        layer?.cornerRadius = 9
        layer?.backgroundColor = NSColor.systemBlue.withAlphaComponent(0.16).cgColor

        label.font = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .medium)
        label.textColor = .systemBlue
        label.translatesAutoresizingMaskIntoConstraints = false
        addSubview(label)
        NSLayoutConstraint.activate([
            label.topAnchor.constraint(equalTo: topAnchor, constant: 3),
            label.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -3),
            label.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            label.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func setCount(_ n: Int) {
        label.stringValue = "\(n) \(Self.framesWord(n))"
        isHidden = n == 0
    }

    private static func framesWord(_ n: Int) -> String {
        let n100 = n % 100
        let n10 = n % 10
        if n100 >= 11 && n100 <= 14 { return "кадров" }
        switch n10 {
        case 1: return "кадр"
        case 2...4: return "кадра"
        default: return "кадров"
        }
    }
}

// MARK: - Редактируемое значение: «лейбл-значение ⇄ текстбокс + ✓/✕»

final class EditableValueControl: NSView {
    private let valueLabel = NSTextField(labelWithString: "")
    private let editButton = NSButton(title: "", target: nil, action: nil)
    private let field = NSTextField(string: "")
    private let okButton = NSButton(title: "", target: nil, action: nil)
    private let cancelButton = NSButton(title: "", target: nil, action: nil)
    /// Контейнер, который выглядит как поле ввода и держит поле + кнопки
    /// ✓/✕ ВНУТРИ (append-inner, как в мессенджере).
    private let fieldContainer = NSView()
    private var committedValue = ""
    let controlWidth: CGFloat
    var onCommit: ((String) -> Void)?

    init(value: String, width: CGFloat = 150) {
        controlWidth = width
        super.init(frame: .zero)
        committedValue = value

        valueLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 13, weight: .regular)
        valueLabel.textColor = .labelColor
        valueLabel.lineBreakMode = .byTruncatingTail

        // Чуть крупнее и жирнее системного карандаша: на 13 pt тонкая
        // диагональ символа читается как «/».
        let pencilCfg = NSImage.SymbolConfiguration(pointSize: 14, weight: .medium)
        let pencil = NSImage(systemSymbolName: "pencil",
                             accessibilityDescription: "Редактировать")?
            .withSymbolConfiguration(pencilCfg)
        editButton.image = pencil
        editButton.imagePosition = .imageOnly
        editButton.isBordered = false
        editButton.contentTintColor = .secondaryLabelColor
        editButton.toolTip = "Редактировать"
        editButton.target = self
        editButton.action = #selector(startEditing)

        field.font = NSFont.monospacedDigitSystemFont(ofSize: 13, weight: .regular)
        field.bezelStyle = .roundedBezel
        field.alignment = .natural
        field.isHidden = true

        // Объёмные эмоджи-кнопки (как в Telegram): ✅ применить, ✕ отменить.
        let emojiFont = NSFont.systemFont(ofSize: 17)
        okButton.attributedTitle = NSAttributedString(string: "✅",
                                                      attributes: [.font: emojiFont])
        okButton.setButtonType(.momentaryPushIn)
        okButton.isBordered = false
        okButton.target = self
        okButton.action = #selector(commit)
        okButton.toolTip = "Применить"
        okButton.isHidden = true

        cancelButton.attributedTitle = NSAttributedString(string: "❌",
                                                          attributes: [.font: emojiFont])
        cancelButton.setButtonType(.momentaryPushIn)
        cancelButton.isBordered = false
        cancelButton.target = self
        cancelButton.action = #selector(cancelEdit)
        cancelButton.toolTip = "Отменить"
        cancelButton.isHidden = true

        // Внутренний контейнер: выглядит как поле (фон, скругление, рамка),
        // внутри — текстовое поле + кнопки справа.
        fieldContainer.wantsLayer = true
        fieldContainer.layer?.cornerRadius = 6
        fieldContainer.layer?.borderWidth = 1
        fieldContainer.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor
        fieldContainer.layer?.backgroundColor = NSColor.textBackgroundColor.cgColor

        let inner = NSStackView(views: [field, okButton, cancelButton])
        inner.orientation = .horizontal
        inner.spacing = 2
        inner.alignment = .centerY
        inner.translatesAutoresizingMaskIntoConstraints = false
        fieldContainer.addSubview(inner)
        fieldContainer.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            inner.leadingAnchor.constraint(equalTo: fieldContainer.leadingAnchor, constant: 6),
            inner.trailingAnchor.constraint(equalTo: fieldContainer.trailingAnchor, constant: -4),
            inner.topAnchor.constraint(equalTo: fieldContainer.topAnchor, constant: 2),
            inner.bottomAnchor.constraint(equalTo: fieldContainer.bottomAnchor, constant: -2),
        ])
        fieldContainer.isHidden = true

        let row = NSStackView(views: [valueLabel, fieldContainer, editButton])
        row.orientation = .horizontal
        row.spacing = 6
        row.alignment = .centerY
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        var constraints: [NSLayoutConstraint] = [
            row.topAnchor.constraint(equalTo: topAnchor),
            row.bottomAnchor.constraint(equalTo: bottomAnchor),
            row.leadingAnchor.constraint(equalTo: leadingAnchor),
            row.trailingAnchor.constraint(equalTo: trailingAnchor),
            heightAnchor.constraint(equalToConstant: 28),
            okButton.widthAnchor.constraint(equalToConstant: 26),
            cancelButton.widthAnchor.constraint(equalToConstant: 26),
        ]
        if controlWidth > 0 {
            constraints.append(widthAnchor.constraint(equalToConstant: controlWidth))
            constraints.append(fieldContainer.widthAnchor.constraint(equalToConstant: max(60, controlWidth - 34)))
            // Поле внутри контейнера растягивается на всю доступную ширину
            // (между кнопками ✓/✕), а не зажато в 30px — иначе даже
            // 2 цифры не видны при редактировании.
            field.setContentHuggingPriority(.defaultLow, for: .horizontal)
            field.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
            constraints.append(field.widthAnchor.constraint(greaterThanOrEqualToConstant: 44))
        } else {
            // Гибкая ширина (для длинных путей): значение и поле тянутся на всю строку.
            valueLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
            valueLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
            field.setContentHuggingPriority(.defaultLow, for: .horizontal)
            field.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
            fieldContainer.setContentHuggingPriority(.defaultLow, for: .horizontal)
        }
        NSLayoutConstraint.activate(constraints)
        sync()
        applyVisibility()
    }

    required init?(coder: NSCoder) { fatalError() }

    var stringValue: String { committedValue }

    func setValue(_ s: String) {
        committedValue = s
        endEditingState()
    }

    func intValue(def: Int) -> Int {
        Int(committedValue.trimmingCharacters(in: .whitespaces)) ?? def
    }

    func doubleValue(def: Double) -> Double {
        Double(committedValue.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: ",", with: ".")) ?? def
    }

    private func sync() {
        valueLabel.stringValue = committedValue
        valueLabel.toolTip = committedValue
        field.stringValue = committedValue
    }

    /// Применяет видимость по чистой протестированной логике editFieldVisibility.
    private func applyVisibility(_ isEditing: Bool = false) {
        let v = editFieldVisibility(isEditing: isEditing)
        valueLabel.isHidden = !v.showValue
        editButton.isHidden = !v.showEditButton
        fieldContainer.isHidden = !v.showInputWithActions
        field.isHidden = !v.showInputWithActions
        okButton.isHidden = !v.showInputWithActions
        cancelButton.isHidden = !v.showInputWithActions
    }

    @objc private func startEditing() {
        field.stringValue = committedValue
        applyVisibility(true)
        window?.makeFirstResponder(field)
    }

    @objc private func commit() {
        committedValue = field.stringValue.trimmingCharacters(in: .whitespaces)
        onCommit?(committedValue)
        endEditingState()
    }

    @objc private func cancelEdit() {
        endEditingState()
    }

    private func endEditingState() {
        sync()
        applyVisibility(false)
    }
}

// MARK: - Кнопка миниатюры с откликом на наведение (hover)

final class HoverThumbButton: NSButton {
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for ta in trackingAreas { removeTrackingArea(ta) }
        let ta = NSTrackingArea(rect: bounds,
                                options: [.mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
                                owner: self, userInfo: nil)
        addTrackingArea(ta)
    }

    override func mouseEntered(with event: NSEvent) {
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.12
            layer?.setValue(1.06, forKey: "transform.scale")
        }
        layer?.borderColor = NSColor.systemBlue.cgColor
        layer?.borderWidth = 1.5
        NSCursor.pointingHand.set()
    }

    override func mouseExited(with event: NSEvent) {
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.12
            layer?.setValue(1.0, forKey: "transform.scale")
        }
        layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor
        layer?.borderWidth = 0.5
        NSCursor.arrow.set()
    }
}

// MARK: - Живой график метрик (PSNR / loss) по эпохам
// Паттерн: линейные графики метрик — YouTube Studio / Statsig / PostHog.

final class HistoryChart: NSView {
    private var losses: [Double] = []
    private var psnrs: [Double] = []

    func add(loss: Double, psnr: Double) {
        losses.append(loss)
        psnrs.append(psnr)
        needsDisplay = true
    }

    func reset() {
        losses = []
        psnrs = []
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        let w = bounds.width
        let h = bounds.height
        let padL: CGFloat = 8
        let padR: CGFloat = 8
        let padT: CGFloat = 8
        let padB: CGFloat = 16
        let cw = w - padL - padR
        let ch = h - padT - padB

        // Сетка из 4 линий
        let gridColor = NSColor.separatorColor
        for i in 0...4 {
            let y = padT + ch * CGFloat(i) / 4
            gridColor.setStroke()
            let path = NSBezierPath()
            path.move(to: NSPoint(x: padL, y: y))
            path.line(to: NSPoint(x: w - padR, y: y))
            path.lineWidth = 0.5
            path.stroke()
        }

        guard !losses.isEmpty else {
            let placeholder = "График появится после первой эпохи" as NSString
            let attrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 12),
                .foregroundColor: NSColor.tertiaryLabelColor,
            ]
            let size = placeholder.size(withAttributes: attrs)
            placeholder.draw(at: NSPoint(x: (w - size.width) / 2, y: (h - size.height) / 2),
                             withAttributes: attrs)
            return
        }

        func polyline(_ values: [Double], color: NSColor) {
            let count = max(values.count, 2)
            let lo = values.min() ?? 0
            let hi = values.max() ?? 1
            let range = max(hi - lo, 1e-6)
            color.setStroke()
            let p = NSBezierPath()
            for (i, v) in values.enumerated() {
                let x = padL + cw * CGFloat(i) / CGFloat(count - 1)
                let y = padT + ch * CGFloat((v - lo) / range)
                if i == 0 { p.move(to: NSPoint(x: x, y: y)) }
                else { p.line(to: NSPoint(x: x, y: y)) }
            }
            p.lineWidth = 2
            p.lineJoinStyle = .round
            p.stroke()
        }

        polyline(psnrs, color: .systemBlue)
        polyline(losses, color: .systemGray)
    }
}

final class YumeAppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!

    // Поля параметров
    private let folderField = NSTextField(string: "")
    private let widthField = NSTextField(string: "854")
    private let heightField = NSTextField(string: "480")
    private let extractLimitCtl = EditableValueControl(value: "0")
    private let epochsCtl = EditableValueControl(value: "20")
    private let stepsCtl = EditableValueControl(value: "20")
    private let batchCtl = EditableValueControl(value: "1")
    private let lrCtl = EditableValueControl(value: "0.0003")
    private let pauseCtl = EditableValueControl(value: "250")
    private let powerCtl = EditableValueControl(value: "90")
    /// Размер кадра, на котором обучается VAE. Исходные JPEG остаются
    /// полноразмерными — декодер уменьшает их на лету (как старый стенд
    /// 80×56, на котором сеть сходилась заметно лучше 480p).
    private let widthCtl = EditableValueControl(value: "80")
    private let heightCtl = EditableValueControl(value: "56")
    private let convPathCtl = EditableValueControl(value: "~/YumeLayers/checkpoints/conv.safetensors", width: 0)
    private let resumeSwitch = NSSwitch()
    private let adaptiveSwitch = NSSwitch()

    private let trainButton = NSButton(title: "Обучить VAE", target: nil, action: nil)
    private let stopButton = NSButton(title: "Стоп", target: nil, action: nil)
    private let browseButton = NSButton(title: "Обзор", target: nil, action: nil)

    // Статус, прогресс, превью, лог
    private let statusLabel = NSTextField(labelWithString: "Готово к обучению")
    // Прогресс каждого действия живёт рядом с его кнопкой (см. карточки 1 и 3).
    private let extractProgressBar = NSProgressIndicator()
    private let extractPercentLabel = NSTextField(labelWithString: "0%")
    private let extractEtaLabel = NSTextField(labelWithString: "—")
    private var extractProgressBlock: NSStackView!
    private let trainProgressBar = NSProgressIndicator()
    private let trainPercentLabel = NSTextField(labelWithString: "0%")
    private let trainEtaLabel = NSTextField(labelWithString: "—")
    private var trainProgressBlock: NSStackView!
    // Живой индикатор: спиннер + состояние + текущий кадр/шаг/скорость.
    private let trainSpinner = NSProgressIndicator()
    private let trainStateLabel = NSTextField(labelWithString: "Обучение идёт")
    private let trainFrameLabel = NSTextField(labelWithString: "Ожидание первого шага…")
    private var trainLiveBlock: NSStackView!
    private var trainHeartbeatTimer: Timer?
    private var lastTrainStepAt: Date?
    private var lastTrainReport: TrainStepReport?
    private let reconsProgressBar = NSProgressIndicator()
    private let reconsPercentLabel = NSTextField(labelWithString: "0%")
    private var reconsProgressBlock: NSStackView!
    private let epochCard = MetricCard(title: "Эпоха", symbol: "clock")
    private let lossCard = MetricCard(title: "Loss", symbol: "chart.line.downtrend.xyaxis")
    private let psnrCard = MetricCard(title: "PSNR, дБ", symbol: "speedometer")
    private let chart = HistoryChart()
    private let previewView = NSImageView()
    private let playerView = AVPlayerView()
    private let queuePlayer = AVQueuePlayer()
    private var looper: AVPlayerLooper?
    private var videoEmptyState: NSStackView!
    private var videoURLs: [URL] = []
    private var isExtracting = false
    private var currentExtractor: FrameExtractor?
    private var extractCancelled = false
    private var extractedCount = 0
    private var skippedBlackTotal = 0
    private let extractButton = NSButton(title: "Вытащить кадры", target: nil, action: nil)
    private let stopExtractButton = NSButton(title: "Стоп", target: nil, action: nil)
    private let resetParamsButton = NSButton(title: "Сброс к умолчаниям", target: nil, action: nil)
    private let autoTuneButton = NSButton(title: "Авто-расчёт параметров", target: nil, action: nil)
    private var extractStartTime = Date()
    private var trainStartTime = Date()
    private var lastCollageRender = Date.distantPast

    // Асинхронная выдача извлечённых кадров: фон складывает кадры в буфер,
    // main thread забирает их пачками по таймеру. Так на 44 000 кадров
    // не приходится 44 000 обновлений интерфейса.
    private struct PendingExtractFrame {
        let name: String
        let jpeg: Data
        let source: String
        let isControl: Bool
    }
    private let extractBatch = ExtractBatchPolicy()
    private let pendingExtractLock = NSLock()
    private var pendingExtractFrames: [PendingExtractFrame] = []
    private var pendingExtractProgress: Double = 0
    private var pendingExtractDone = 0
    private var extractTotalAll = 0
    private var extractFlushTimer: Timer?
    private var lastExtractFlush = Date.distantPast
    private var flushedFirstExtractBatch = false
    private let carouselPrev = NSButton(image: NSImage(systemSymbolName: "chevron.left", accessibilityDescription: "Назад")!, target: nil, action: nil)
    private let carouselNext = NSButton(image: NSImage(systemSymbolName: "chevron.right", accessibilityDescription: "Вперёд")!, target: nil, action: nil)
    private let carouselFirst = NSButton(image: NSImage(systemSymbolName: "chevron.left.to.line", accessibilityDescription: "В начало")!, target: nil, action: nil)
    private let carouselLast = NSButton(image: NSImage(systemSymbolName: "chevron.right.to.line", accessibilityDescription: "В конец")!, target: nil, action: nil)
    private var pageDots: [NSButton] = []
    /// Ограничения ширины кружочков-страниц (обновляются под число цифр).
    private var dotWidthConstraints: [NSLayoutConstraint] = []
    private let dotPoolSize = 30   // запас кнопок; видимое число считается по ширине
    private let paginatorRow = NSStackView()
    private var framesStore: [MemoryFrame] = []
    /// Контрольная выборка (1 кадр/сек по каждому видео) — для визуальной
    /// оценки качества реконструкций на итоговой модели. Живёт отдельно и не
    /// освобождается по ходу обучения.
    private var controlFramesStore: [MemoryFrame] = []
    private var processedFrames: [Bool] = []
    /// Реконструкции в памяти: имя кадра → PNG-данные (без записи на диск).
    private var reconMemory: [String: Data] = [:]
    private var carouselPage = 0
    private let pageSize = 10   // 5 внизу + 5 справа вокруг большой
    private let collageBig = NSImageView()
    private let collageBigBadge = NSImageView()
    private var collageThumbs: [NSButton] = []
    private var thumbBadges: [NSImageView] = []
    private var selectedFrameIndex = -1
    private let collageEmpty = NSTextField(labelWithString: "Кадры появятся после извлечения видео")
    private let sourceSwitch = NSSegmentedControl(labels: ["Оригинал", "Реконструкция"], trackingMode: .selectOne, target: nil, action: nil)
    private let framesChip = CountChip()
    private var showRecon = false
    private var epochsEditedByUser = false
    private var stepsEditedByUser = false
    private var lrEditedByUser = false
    private let videoListScroll = NSScrollView()
    private let videoListStack = NSStackView()

    /// Порядок кадров в L: нижний ряд 0-4 (слева → справа), затем правый
    /// столбец 5-9 (снизу → вверх, конец — правый верхний).
    private func frameOffset(forSlot slot: Int) -> Int {
        frameOffsetForSlot(slot)
    }
    private let carouselStack = NSStackView()
    private let carouselScroll = NSScrollView()
    private var carouselPlaceholder: NSStackView!
    private let logView = NSTextView()

    private var isTraining = false
    /// Идут ли реконструкции в фоне (управляются кнопкой «Стоп»).
    private var isReconstructing = false
    private var lastEpoch = 0
    private let api = ApiServer()
    private var layoutViews: [String: NSView] = [:]
    private var debugGrids: [NSView] = []
    private var lastParams = Params()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.mainMenu = buildMenu()

        _ = api.start()
        // Вся отладка core-кода (yumeLog) — в журнал приложения, а не только в консоль.
        YumeLogBus.shared.subscribe { [weak self] line in
            DispatchQueue.main.async { self?.appendCoreLog(line) }
        }
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
                self.saveSelection()
                // Тот же запуск извлечения, что и у кнопки «Вытащить кадры».
                self.startExtraction(urls: self.videoURLs, append: false, maxFrames: maxFrames ?? 0)
            }
        }
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
                            self.saveSelection()
                            applied.append(key)
                        } else { unknown.append(key) }
                    default:
                        unknown.append(key)
                    }
                }
            }
            return (unknown.isEmpty, applied, unknown)
        }
        api.onResetParams = { [weak self] in
            DispatchQueue.main.async { self?.resetParams() }
        }
        api.onAutoTune = { [weak self] in
            DispatchQueue.main.async { self?.autoTuneNow() }
        }
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
        restoreSelection()
        log("Готово. Добавьте видео → извлеките кадры → начните обучение.")
    }

    @objc private func showMainWindow() {
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

    /// Разворачивает окно на весь рабочий стол при старте.
    ///
    /// Это НЕ нативный полноэкранный режим: меню-бар сверху и Dock остаются на
    /// месте, окно просто занимает `visibleFrame` — ровно то же, что даёт
    /// зелёная кнопка (или «Увеличить» в меню «Окно»). `visibleFrame` уже
    /// исключает полосу меню и Dock, поэтому окно ничего не перекрывает.
    private func maximizeOnLaunch() {
        guard let window = window else { return }
        guard let screen = window.screen ?? NSScreen.main else { return }
        window.setFrame(screen.visibleFrame, display: false)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showMainWindow()
        return true
    }

    // MARK: - Стандартное главное меню (HIG)

    private func buildMenu() -> NSMenu {
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

    // MARK: - Вспомогательное

    private func sectionHeader(_ text: String) -> NSTextField {
        let l = NSTextField(labelWithString: text)
        l.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        l.textColor = .labelColor
        return l
    }

    /// Строка формы, которая растягивается на всю ширину карточки:
    /// [надпись | ⓘ | …поле…] — поле занимает всё расстояние от кнопки
    /// info до правого края строки/колонки.
    private final class StretchRow: NSStackView {}

    /// Строка формы: [надпись | ⓘ | …поле…] — кнопка info сразу после
    /// надписи, поле растягивается на всё расстояние до правого края.
    /// `info` — полное описание для выпадающей карточки.
    private let labelW: CGFloat = 141
    private let rowGap: CGFloat = 6
    private let accW: CGFloat = 76
    private let infoW: CGFloat = 16

    private func fieldRow(_ label: String, _ field: NSTextField, tooltip: String? = nil, accessory: NSView? = nil, info: String? = nil) -> StretchRow {
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

    /// Строка с редактируемым значением: [надпись | ⓘ | …поле…].
    /// Кнопка info — сразу после надписи, значение/поле занимают всё
    /// расстояние от неё до правого края строки.
    private func valueRow(_ label: String, _ ctl: EditableValueControl, info: String? = nil) -> StretchRow {
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

    private func makeStretchRow(_ views: [NSView]) -> StretchRow {
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

    /// Компактная строка переключателя (как в web-admin-panels): switch + лейбл.
    /// Лейбл — в той же колонке, что и у строк с числами, переключатель —
    /// в колонке значений (по правому краю). Единая сетка карточки.
    private func switchRow(_ label: String, _ control: NSSwitch,
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

    /// Блок пути: подпись с info на первой строке, поле — на отдельной
    /// строке во всю ширину карточки, чтобы длинный путь был виден целиком.
    private func pathBlock(_ label: String, _ ctl: EditableValueControl, info: String) -> NSStackView {
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

    // MARK: - Кнопка info с выпадающей карточкой-описанием
    private var infoTexts: [String] = []
    private var openPopovers: [Int: NSPopover] = [:]

    private func makeInfoButton(_ text: String) -> NSButton {
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

    @objc private func infoTapped(_ sender: NSButton) {
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

    // MARK: - UI

    private func buildUI() -> NSView {
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
        func groupCard(title: String, contentViews: [NSView], spacing: CGFloat) -> NSView {
            let card = NSVisualEffectView()
            card.material = .contentBackground
            card.blendingMode = .withinWindow
            card.state = .active
            card.wantsLayer = true
            card.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
            card.layer?.cornerRadius = 12
            card.layer?.borderWidth = 0.5
            card.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.45).cgColor

            let titleLabel = sectionHeader(title)
            titleLabel.translatesAutoresizingMaskIntoConstraints = false
            card.addSubview(titleLabel)
            NSLayoutConstraint.activate([
                titleLabel.topAnchor.constraint(equalTo: card.topAnchor, constant: 16),
                titleLabel.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 16),
                titleLabel.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -16),
            ])

            var previous: NSView = titleLabel
            for v in contentViews {
                card.addSubview(v)
                v.translatesAutoresizingMaskIntoConstraints = false
                NSLayoutConstraint.activate([
                    v.topAnchor.constraint(equalTo: previous.bottomAnchor, constant: spacing),
                    v.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 16),
                ])
                // Строки формы растягиваются на всю ширину карточки; всё
                // остальное (кнопки, блоки прогресса, подсказки) прижато
                // к левому краю без растяжения.
                if v is StretchRow {
                    v.trailingAnchor.constraint(equalTo: card.trailingAnchor,
                                                constant: -16).isActive = true
                }
                previous = v
            }
            if let last = contentViews.last {
                last.bottomAnchor.constraint(equalTo: card.bottomAnchor,
                                            constant: -16).isActive = true
            }
            return card
        }

        // Шаг 1 — Видео: выбор MP4 + извлечение кадров
        let folderRow = fieldRow("Файл:", folderField, accessory: browseButton,
                                 info: "Выберите один или несколько видеофайлов (MP4/MOV). Кадры будут извлечены строго в порядке выбора — это важно для привязки движения к seed.")

        extractButton.target = self
        extractButton.action = #selector(extractFrames)
        extractButton.bezelStyle = .rounded
        extractButton.controlSize = .large
        extractButton.image = NSImage(systemSymbolName: "camera.fill", accessibilityDescription: "Вытащить кадры")
        extractButton.imagePosition = .imageLeading
        extractButton.isEnabled = false

        stopExtractButton.target = self
        stopExtractButton.action = #selector(stopExtraction)
        stopExtractButton.bezelStyle = .rounded
        stopExtractButton.controlSize = .large
        stopExtractButton.image = NSImage(systemSymbolName: "stop.fill", accessibilityDescription: "Остановить извлечение")
        stopExtractButton.imagePosition = .imageLeading
        stopExtractButton.isEnabled = false

        let extractRow = NSStackView(views: [extractButton, stopExtractButton])
        extractRow.orientation = .horizontal
        extractRow.spacing = 10
        extractRow.alignment = .centerY

        let extractHint = NSTextField(wrappingLabelWithString: "Выберите видеофайл, затем нажмите «Вытащить кадры». Каждый кадр сразу появится ниже.")
        extractHint.font = NSFont.systemFont(ofSize: 11)
        extractHint.textColor = .tertiaryLabelColor
        extractHint.alignment = .left
        extractHint.widthAnchor.constraint(equalToConstant: 340).isActive = true

        // Прогресс извлечения — сразу под кнопками «Вытащить кадры» / «Стоп».
        extractProgressBlock = makeProgressBlock(bar: extractProgressBar,
                                                 percent: extractPercentLabel,
                                                 eta: extractEtaLabel)
        extractProgressBlock.widthAnchor.constraint(equalToConstant: 340).isActive = true

        let extractLimitRow = valueRow(
            "Кадров для теста:",
            extractLimitCtl,
            info: "Сколько первых кадров извлечь. 0 — все кадры. Лимит не меняет частоту: берутся первые N кадров подряд при fps исходника, как при полном извлечении, а не прореженная выборка по всему ролику. Нужен для быстрой проверки обучения на 2000 кадрах.")
        let dataPage = groupCard(title: "1. Видео",
                                 contentViews: [folderRow, extractLimitRow, extractRow,
                                                extractProgressBlock, extractHint],
                                 spacing: 12)

        // Шаг 2 — Параметры обучения
        resetParamsButton.target = self
        resetParamsButton.action = #selector(resetParams)
        resetParamsButton.bezelStyle = .rounded
        resetParamsButton.controlSize = .small
        resetParamsButton.widthAnchor.constraint(equalToConstant: 190).isActive = true
        autoTuneButton.target = self
        autoTuneButton.action = #selector(autoTuneNow)
        autoTuneButton.bezelStyle = .rounded
        autoTuneButton.controlSize = .small
        autoTuneButton.widthAnchor.constraint(equalToConstant: 190).isActive = true
        adaptiveSwitch.state = .on   // авто-пауза при перегреве включена по умолчанию
        // Ручное подтверждение значения отключает авто-подбор только этого поля:
        // второе поле продолжает подстраиваться под число кадров.
        epochsCtl.onCommit = { [weak self] _ in self?.epochsEditedByUser = true }
        stepsCtl.onCommit = { [weak self] _ in self?.stepsEditedByUser = true }
        lrCtl.onCommit = { [weak self] _ in self?.lrEditedByUser = true }
        let paramsPage = groupCard(title: "2. Параметры обучения",
                                   contentViews: [
            switchRow("Продолжить обучение", resumeSwitch,
                      tooltip: "Продолжить с чекпоинта: обучение начнётся с последнего сохранения"),
            valueRow("Эпохи:", epochsCtl,
                     info: "Сколько раз модель полностью пройдёт по всем кадрам набора. Больше эпох — лучше качество, но дольше обучение и выше нагрев GPU."),
            valueRow("Шагов за эпоху:", stepsCtl,
                     info: "Сколько обновлений весов выполняется за одну эпоху. Один шаг = обработка одного кадра + обновление весов."),
            valueRow("Размер батча:", batchCtl,
                     info: "Сколько кадров обрабатывается за один шаг обучения. Больший батч быстрее проходит эпоху при той же тепловой нагрузке: регулятор держит долю работы GPU, а не число шагов. Верхняя граница — по свободной памяти GPU."),
            valueRow("Ширина обучения:", widthCtl,
                     info: "Ширина кадра, до которой уменьшаются кадры при обучении. Исходные JPEG в памяти остаются полноразмерными. По умолчанию 80 — быстрый тест сходимости; 854 — полное разрешение (медленнее и хуже сходится на этой сети)."),
            valueRow("Высота обучения:", heightCtl,
                     info: "Высота кадра, до которой уменьшаются кадры при обучении. По умолчанию 56. Вместе с шириной должна быть кратна 8 (лишнее добивается краем без растяжения)."),
            valueRow("Скорость обучения:", lrCtl,
                     info: "Размер шага градиентного спуска (learning rate). Слишком большое значение — расходимость/NaN, слишком малое — очень медленное обучение. По умолчанию 0.0003."),
            valueRow("Пауза, мс:", pauseCtl,
                     info: "Фиксированная пауза после каждого шага, в миллисекундах. Работает только когда «Авто-пауза (нагрев)» выключена: тогда нагрузка постоянная и равна этой паузе. При включённом авто-режиме паузу считает регулятор по температуре, и это поле не используется."),
            valueRow("Нагрузка GPU, % (≤98):", powerCtl,
                     info: "Потолок нагрузки: сколько процентов времени GPU занята работой. Значение читается как есть (например, 98 — GPU занята 98% времени). Верхняя граница 98% — физический предел, при котором WindowServer получает слоты между нашими command buffer'ами; полностью непрерывная очередь приводила к перезагрузке 4 ноября 2026. Фактическую нагрузку регулятор подбирает по температуре: рабочая точка 82 °C, потолок 85 °C, при 86 °C — принудительное остывание."),
            switchRow("Авто-пауза (нагрев)", adaptiveSwitch,
                      tooltip: "Авто-пауза при перегреве: регулятор сам подбирает нагрузку по температуре GPU — держит максимум, который не ведёт к перезагрузке"),
            autoTuneButton,
            resetParamsButton,
        ], spacing: 10)

        trainButton.target = self
        trainButton.action = #selector(startTraining as () -> Void)
        trainButton.bezelStyle = .rounded
        trainButton.keyEquivalent = "\r"
        trainButton.controlSize = .large
        trainButton.bezelColor = .systemBlue
        // Кнопки «Старт/Стоп»: иконки встроены в текст (▶/■) — у системной
        // кнопки AppKit внутренний зазор между image и title не убирается,
        // а текст с символом даёт ровную типографику без отступов.
        trainButton.image = nil
        trainButton.title = "▶ Обучить VAE"
        trainButton.imagePosition = .imageLeading

        stopButton.target = self
        stopButton.action = #selector(stopTraining)
        stopButton.bezelStyle = .rounded
        stopButton.keyEquivalent = "\u{1b}"   // Esc
        stopButton.controlSize = .regular
        stopButton.image = nil
        stopButton.title = "■ Стоп"
        stopButton.isEnabled = false

        let runButtons = NSStackView(views: [trainButton, stopButton])
        runButtons.orientation = .horizontal
        runButtons.spacing = 10
        runButtons.alignment = .centerY

        // Прогресс обучения и реконструкций — прямо под кнопками.
        trainProgressBlock = makeProgressBlock(bar: trainProgressBar,
                                               percent: trainPercentLabel,
                                               eta: trainEtaLabel)
        trainProgressBlock.widthAnchor.constraint(equalToConstant: 360).isActive = true

        // Живой индикатор под прогрессом обучения: спиннер + две строки
        // (состояние и реальный кадр/шаг). Скрыт вне обучения.
        trainSpinner.style = .spinning
        trainSpinner.controlSize = .small
        trainSpinner.isDisplayedWhenStopped = false
        trainSpinner.widthAnchor.constraint(equalToConstant: 16).isActive = true
        trainSpinner.heightAnchor.constraint(equalToConstant: 16).isActive = true

        trainStateLabel.font = NSFont.systemFont(ofSize: 11, weight: .semibold)
        trainStateLabel.textColor = .labelColor
        trainStateLabel.lineBreakMode = .byTruncatingTail

        trainFrameLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .regular)
        trainFrameLabel.textColor = .tertiaryLabelColor
        trainFrameLabel.lineBreakMode = .byTruncatingTail
        trainFrameLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        let trainLiveColumn = NSStackView(views: [trainStateLabel, trainFrameLabel])
        trainLiveColumn.orientation = .vertical
        trainLiveColumn.alignment = .leading
        trainLiveColumn.spacing = 2
        trainLiveColumn.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        trainLiveBlock = NSStackView(views: [trainSpinner, trainLiveColumn])
        trainLiveBlock.orientation = .horizontal
        trainLiveBlock.alignment = .centerY
        trainLiveBlock.spacing = 8
        trainLiveBlock.widthAnchor.constraint(equalToConstant: 360).isActive = true
        trainLiveBlock.isHidden = true
        reconsProgressBlock = makeProgressBlock(bar: reconsProgressBar,
                                                percent: reconsPercentLabel,
                                                eta: nil)
        reconsProgressBlock.widthAnchor.constraint(equalToConstant: 360).isActive = true

        let hint = NSTextField(wrappingLabelWithString: "Не запускайте несколько обучений одновременно. При перегреве система сама увеличит паузы.")
        hint.font = NSFont.systemFont(ofSize: 11)
        hint.textColor = .tertiaryLabelColor
        hint.widthAnchor.constraint(equalToConstant: 360).isActive = true

        // Шаг 3 — Запуск
        let runPage = groupCard(title: "3. Запуск",
                                contentViews: [
            runButtons,
            trainProgressBlock,
            trainLiveBlock,
            reconsProgressBlock,
            pathBlock("Файл VAE:", convPathCtl,
                      info: "Путь к файлу, в который сохраняется обученная модель (чекпоинт, safetensors). Можно указать любой файл, например ~/YumeLayers/checkpoints/my_vae.safetensors."),
            hint,
        ], spacing: 10)

        for p in [dataPage, paramsPage, runPage] {
            p.widthAnchor.constraint(equalToConstant: 392).isActive = true
            // Карточки не растягиваются по вертикали — держатся нативной высоты.
            p.setContentHuggingPriority(.required, for: .vertical)
        }

        let left = NSStackView(views: [dataPage, paramsPage, runPage])
        left.orientation = .vertical
        left.spacing = 16
        left.alignment = .leading
        left.edgeInsets = NSEdgeInsets(top: 24, left: 24, bottom: 24, right: 24)
        left.widthAnchor.constraint(equalToConstant: 440).isActive = true

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
        func mediaCard(_ content: NSView, _ empty: NSStackView) -> NSView {
            let card = NSView()
            card.wantsLayer = true
            card.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
            card.layer?.cornerRadius = 12
            card.layer?.borderWidth = 0.5
            card.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.45).cgColor
            card.addSubview(content)
            card.addSubview(empty)
            content.translatesAutoresizingMaskIntoConstraints = false
            empty.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                content.topAnchor.constraint(equalTo: card.topAnchor),
                content.bottomAnchor.constraint(equalTo: card.bottomAnchor),
                content.leadingAnchor.constraint(equalTo: card.leadingAnchor),
                content.trailingAnchor.constraint(equalTo: card.trailingAnchor),
                empty.centerXAnchor.constraint(equalTo: card.centerXAnchor),
                empty.centerYAnchor.constraint(equalTo: card.centerYAnchor),
            ])
            return card
        }

        func simpleEmpty(_ symbol: String, _ text: String) -> NSStackView {
            let icon = NSImageView()
            icon.image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)
            icon.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 36, weight: .light)
            icon.contentTintColor = .secondaryLabelColor
            let label = NSTextField(labelWithString: text)
            label.font = NSFont.systemFont(ofSize: 13)
            label.textColor = .secondaryLabelColor
            let s = NSStackView(views: [icon, label])
            s.orientation = .vertical
            s.spacing = 8
            s.alignment = .centerX
            return s
        }

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

        // ——— Коллаж кадров: большая + 9 вокруг (клик ставит кадр в центр) ———
        func photoBadge() -> NSImageView {
            let b = NSImageView()
            b.image = NSImage(systemSymbolName: "checkmark.circle.fill", accessibilityDescription: nil)
            b.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 14, weight: .bold)
            b.contentTintColor = .systemGreen
            b.translatesAutoresizingMaskIntoConstraints = false
            b.isHidden = true   // показывается только после обработки моделью
            return b
        }

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


        // Пагинатор — на месте бывшей надписи, между видео и коллажем.
        let center = NSStackView(views: [videoCaption, videoCard, paginatorRow, collageHeader, collageCard])
        center.orientation = .vertical
        center.spacing = 10
        center.alignment = .leading
        center.edgeInsets = NSEdgeInsets(top: 24, left: 24, bottom: 24, right: 24)
        // Колонка должна сжиматься под ширину окна — иначе содержимое
        // (картинки кадров) распирает окно за пределы экрана.
        center.setContentHuggingPriority(.defaultLow, for: .horizontal)
        center.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        // Ширины медиа-блоков — строго в пределах центральной колонки.
        // Отступы те же, что в paginatorSideInset (по 24 с каждой стороны).
        let centerInset = -2 * CGFloat(paginatorSideInset)
        collageHeader.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        videoCard.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        collageCard.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        paginatorRow.widthAnchor.constraint(equalTo: center.widthAnchor, constant: centerInset).isActive = true
        paginatorRow.heightAnchor.constraint(equalToConstant: 28).isActive = true

        // Вертикальные пропорции: видео 50%, коллаж 45% зоны.
        // Высоты — литералы от высоты экрана (без самоссылочного роста).
        videoCard.heightAnchor.constraint(equalTo: center.heightAnchor, multiplier: 0.43, constant: -38).isActive = true
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
        layoutViews["dataPage"] = dataPage
        layoutViews["paramsPage"] = paramsPage
        layoutViews["runPage"] = runPage
        layoutViews["left"] = left
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
        layoutViews["folderRow"] = folderRow
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

    private func setStatus(_ text: String, ok: Bool) {
        statusLabel.stringValue = text
        statusLabel.textColor = ok ? .labelColor : .systemRed
    }

    // MARK: - Действия

    /// Выбор одного или нескольких видеофайлов (MP4/MOV).
    @objc private func chooseVideo() {
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

    /// Единая регистрация выбранных видео: список справа, поле «Файл:»,
    /// активность кнопки извлечения, показ первого видео. Используется и UI,
    /// и API — чтобы оба пути вели себя одинаково.
    private func setSelectedVideos(_ urls: [URL]) {
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

    /// Блок прогресса действия: [полоса | проценты | ETA]. Один вид для
    /// извлечения кадров, обучения и реконструкций. Ширина задаётся снаружи
    /// под карточку, полоса тянется на свободное место.
    private func makeProgressBlock(bar: NSProgressIndicator,
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

    /// Прогресс извлечения — под кнопкой «Вытащить кадры».
    private func setExtractProgress(_ p: Double, done: Int, total: Int) {
        let clamped = min(max(0, p), 1)
        extractProgressBar.doubleValue = clamped
        extractPercentLabel.stringValue = String(format: "%d%%", Int(clamped * 100))
        updateETALabel(extractEtaLabel,
                       elapsed: Date().timeIntervalSince(extractStartTime),
                       done: done, total: total)
    }

    /// Прогресс обучения — под кнопкой «Обучить VAE».
    private func setTrainProgress(_ p: Double, done: Int, total: Int) {
        let clamped = min(max(0, p), 1)
        trainProgressBar.doubleValue = clamped
        trainPercentLabel.stringValue = String(format: "%d%%", Int(clamped * 100))
        updateETALabel(trainEtaLabel,
                       elapsed: Date().timeIntervalSince(trainStartTime),
                       done: done, total: total)
    }

    /// Раз в секунду проверяем, живой ли тренировочный цикл: если отчётов
    /// о шагах давно нет, спиннер останавливается и строка честно говорит,
    /// сколько секунд нет прогресса.
    private func startTrainHeartbeat() {
        trainHeartbeatTimer?.invalidate()
        trainHeartbeatTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.refreshTrainLiveStatus()
        }
    }

    private func stopTrainHeartbeat() {
        trainHeartbeatTimer?.invalidate()
        trainHeartbeatTimer = nil
    }

    private func refreshTrainLiveStatus() {
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

    /// Прогресс реконструкций — отдельная строка под прогрессом обучения.
    private func setReconsProgress(_ p: Double) {
        let clamped = min(max(0, p), 1)
        reconsProgressBar.doubleValue = clamped
        reconsPercentLabel.stringValue = String(format: "%d%%", Int(clamped * 100))
    }

    /// Обновление прогноза времени (ETA) рядом с прогрессбаром действия.
    /// Считается линейно: (затраченное / сделано) × осталось.
    private func updateETALabel(_ label: NSTextField, elapsed: TimeInterval, done: Int, total: Int) {
        guard done > 0, total > 0 else {
            label.stringValue = "—"
            return
        }
        let remaining = elapsed / Double(done) * Double(max(0, total - done))
        label.stringValue = "≈ " + formatDuration(remaining)
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let s = Int(seconds.rounded())
        if s < 60 { return "\(s) с" }
        let m = s / 60
        if m < 60 { return "\(m) мин" }
        return "\(m / 60) ч \(m % 60) мин"
    }

    /// Текущий тепловой уровень macOS (для /api/state и контроля нагрева).
    private func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// Рендер списка выбранных видео в правой колонке (имя + удалить).
    private func renderVideoList() {
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

    /// Удаление видео из выбранного списка (кнопка × в правой колонке).
    @objc private func removeVideo(_ sender: NSButton) {
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

    private func selectVideo(_ url: URL) {
        videoEmptyState.isHidden = true
        looper?.disableLooping()
        queuePlayer.pause()
        queuePlayer.removeAllItems()
        let item = AVPlayerItem(url: url)
        looper = AVPlayerLooper(player: queuePlayer, templateItem: item)
        // Без автозапуска: воспроизведение — по кнопке плеера.
        setStatus("Видео выбрано. Нажмите «Вытащить кадры»", ok: true)
    }

    private func saveSelection() {
        let paths = videoURLs.map { $0.path }
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("YumeLayers", isDirectory: true)
            .appendingPathComponent("last_video.json")
        if let data = try? JSONSerialization.data(withJSONObject: ["videos": paths]) {
            try? data.write(to: url)
        }
    }

    private func restoreSelection() {
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("YumeLayers", isDirectory: true)
            .appendingPathComponent("last_video.json")
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let arr = obj["videos"] as? [String], !arr.isEmpty else { return }
        let urls = arr.map { URL(fileURLWithPath: $0) }
        setSelectedVideos(urls)
    }

    @objc private func extractFrames() {
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

    @objc private func stopExtraction() {
        guard isExtracting else { return }
        extractCancelled = true
        currentExtractor?.cancel()
        setStatus("Остановка извлечения…", ok: false)
        log("=== Остановка извлечения ===")
    }

    @objc private func carouselPagePrev() { changeCarouselPage(-1) }
    @objc private func carouselPageNext() { changeCarouselPage(1) }
    @objc private func carouselPageFirst() {
        let pages = max(1, (framesStore.count + pageSize - 1) / pageSize)
        carouselPage = edgePage(current: carouselPage, pages: pages, toStart: true)
        selectedFrameIndex = carouselPage * pageSize
        renderCollagePage()
    }
    @objc private func carouselPageLast() {
        let pages = max(1, (framesStore.count + pageSize - 1) / pageSize)
        carouselPage = edgePage(current: carouselPage, pages: pages, toStart: false)
        selectedFrameIndex = carouselPage * pageSize
        renderCollagePage()
    }

    @objc private func sourceSwitchChanged(_ sender: NSSegmentedControl) {
        showRecon = sender.selectedSegment == 1
        renderCollagePage()
    }

    /// Переключение страницы пагинатора (по 9 кадров: большая + 9 малых).
    private func changeCarouselPage(_ d: Int) {
        let total = framesStore.count
        let pages = max(1, (total + pageSize - 1) / pageSize)
        carouselPage = min(max(0, carouselPage + d), pages - 1)
        selectedFrameIndex = carouselPage * pageSize
        renderCollagePage()
    }

    /// Клик по кружочку-странице.
    @objc private func pageDotClicked(_ sender: NSButton) {
        let pages = max(1, (framesStore.count + pageSize - 1) / pageSize)
        let visible = paginatorVisiblePages(current: carouselPage, pages: pages, limit: fittedDotCount())
        guard sender.tag >= 0, sender.tag < visible.count else { return }
        carouselPage = visible[sender.tag]
        selectedFrameIndex = carouselPage * pageSize
        renderCollagePage()
    }

    /// Сколько кружочков реально влезает в строку пагинатора.
    /// Считаем от фактической ширины строки (колонка минус отступы 24+24),
    /// а не от ширины самой строки: иначе кружочки распирают строку, строка
    /// «видит» большую ширину и показывает ещё больше кружков — колонка растёт.
    private func fittedDotCount() -> Int {
        let dot = paginatorDotWidth(pageNumber: carouselPage + 1)
        let rowWidth = paginatorRowWidth(columnWidth: Double(centerColumnWidth()))
        let n = dotCapacity(columnWidth: rowWidth, dotWidth: dot)
        return max(1, min(pageDots.count, n))
    }

    /// Ширина центральной колонки из фактической ширины окна:
    /// (контент − левая колонка 440 − два разделителя по 1) / 2.
    private func centerColumnWidth() -> CGFloat {
        let content = window?.contentView?.bounds.width ?? 1360
        return max(320, (content - 442) / 2)
    }

    /// Рендер кружочков-страниц: столько, сколько влезает по ширине,
    /// активная выделена. Ширина кружочка адаптивная: под число цифр
    /// (например «1000» не влезает в 28 pt — расширяем и делаем овал).
    private func renderPageDots(pages: Int) {
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

    /// Извлечение кадров из одного или нескольких MP4.
    /// Кадры нумеруются строго по порядку файлов (глобально), чтобы движение
    /// можно было привязать к seed'у.
    /// - `append`: false — очистить старые кадры; true — продолжить нумерацию с текущих.
    /// - `maxFrames`: лимит кадров (0 — авто от доступной памяти).
    private func startExtraction(urls: [URL], append: Bool, maxFrames: Int = 0) {
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

    /// Финализация извлечения: общий статус, кнопки, лог.
    private func finishExtraction(_ result: Result<Void, Error>) {
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

    /// Кладём кадр в буфер, который main thread забирает пачками.
    /// Вызывается из фонового потока извлечения — без обновления UI.
    private func enqueueExtractedFrame(name: String, jpeg: Data, source: String,
                                       isControl: Bool) {
        pendingExtractLock.lock()
        pendingExtractFrames.append(PendingExtractFrame(name: name, jpeg: jpeg,
                                                        source: source,
                                                        isControl: isControl))
        pendingExtractLock.unlock()
    }

    private func storeExtractProgress(_ p: Double, done: Int) {
        pendingExtractLock.lock()
        pendingExtractProgress = p
        pendingExtractDone = done
        pendingExtractLock.unlock()
    }

    private func resetExtractBatch() {
        pendingExtractLock.lock()
        pendingExtractFrames.removeAll()
        pendingExtractProgress = 0
        pendingExtractDone = 0
        pendingExtractLock.unlock()
        extractTotalAll = 0
        lastExtractFlush = Date.distantPast
        flushedFirstExtractBatch = false
    }

    /// Таймер на main thread забирает накопленные кадры пачками.
    /// 200 мс достаточно, чтобы первые кадры появились почти сразу,
    /// и при этом на 44 000 кадров не приходится 44 000 обновлений UI.
    private func startExtractFlushTimer() {
        stopExtractFlushTimer()
        lastExtractFlush = Date()
        let timer = Timer(timeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.flushExtractBatch()
        }
        RunLoop.main.add(timer, forMode: .common)
        extractFlushTimer = timer
    }

    private func stopExtractFlushTimer() {
        extractFlushTimer?.invalidate()
        extractFlushTimer = nil
    }

    /// Забирает пачку из буфера и добавляет кадры в интерфейс.
    /// - Parameter force: забрать всё, даже неполную пачку (финализация/остановка).
    private func flushExtractBatch(force: Bool = false) {
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

    /// Добавление пачки кадров в память. `processed` = true только когда кадр
    /// обработан моделью (тогда ставится галочка ✓).
    private func appendExtractedFrames(_ batch: [PendingExtractFrame]) {
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

    private func clearCarousel() {
        reconMemory.removeAll()
        framesStore.removeAll()
        controlFramesStore.removeAll()
        processedFrames.removeAll()
        selectedFrameIndex = -1
        carouselPage = 0
        renderCollagePage()
    }

    /// Путь к реконструкции того же кадра (recons/frame_XXXX.jpg).
    /// PNG-данные реконструкции кадра из памяти (nil, если ещё нет).
    private func reconData(for name: String) -> Data? {
        reconMemory[name]
    }

    /// Рендер коллажа: большая (выбранная) + 9 малых текущей страницы.
    private func renderCollagePage() {
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
    }

    /// Обновляет счётчик кадров и пагинацию, не трогая картинки.
    /// Используется во время извлечения: новые кадры сразу доступны
    /// по страницам, но коллаж не перерисовывается на каждый кадр.
    private func updatePagination() {
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

    /// Клик по малому кадру — ставит его в центр коллажа.
    @objc private func thumbClicked(_ sender: NSButton) {
        let off = frameOffset(forSlot: sender.tag)
        let idx = carouselPage * pageSize + off
        guard idx >= 0, idx < framesStore.count else { return }
        selectedFrameIndex = idx
        renderCollagePage()
    }

    /// Рендер собственного окна в PNG (для диагностики интерфейса).
    private func makeScreenshot() -> String? {        guard let content = window.contentView else { return nil }
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

    @objc private func startTraining() {
        startTraining(maxFrames: 0, resume: false)
    }

    /// Запуск обучения. `maxFrames` > 0 — учиться только на первых N кадрах
    /// (для тестов вместо полного датасета). 0 — все кадры.
    /// `resume` = true — продолжить с существующего чекпоинта без диалога
    /// (используется API; в UI решение принимает человек).
    private func startTraining(maxFrames: Int, resume: Bool) {
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

    /// Единая точка входа обучения: и кнопка «Обучить», и API вызывают её.
    /// Кадры для обучения всегда берутся из памяти (единый источник).
    /// `maxFrames` > 0 — обучаться только на первых N кадрах.
    private func startTrainingInternal(_ pIn: Params, maxFrames: Int = 0) {
        guard !isTraining, !isReconstructing else { return }
        // Нельзя запускать обучение, пока идёт раскадровка: оба процесса
        // будут мешать друг другу (кнопка «Стоп» извлечения оставалась бы
        // активной, а GPU был бы перегружен).
        guard !isExtracting else {
            log("Обучение нельзя запустить во время извлечения кадров — дождитесь завершения")
            return
        }
        var p = pIn
        if maxFrames > 0 {
            let limit = min(maxFrames, framesStore.count)
            guard limit > 0 else {
                log("Обучение: нет кадров для лимита \(maxFrames)")
                return
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
        lastParams = p
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
                DispatchQueue.main.async {
                    self.finishTraining(ok: true)
                }
            } catch {
                DispatchQueue.main.async {
                    self.log("Ошибка: \(error.localizedDescription)")
                    self.finishTraining(ok: false)
                }
            }
        }
    }

    @objc private func stopTraining() {
        TrainingControl.shared.cancel()
        setStatus("Остановка…", ok: false)
        log("=== Остановка по запросу ===")
    }

    /// Авто-настройка «Эпохи»/«Шагов за эпоху» под фактическое число кадров.
    /// Поле подстраивается, пока пользователь не подтвердил своё значение
    /// вручную: шаги = число кадров (полный проход за эпоху),
    /// эпохи = 3…10 полных проходов (около 3000 сэмплов суммарно).
    /// Повторный запуск извлечения пересчитывает авто-поля под новый размер очереди.
    private func autoTuneParams(frames: Int) {
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

    /// Кнопка «Авто-расчёт параметров»: пересчёт под текущие кадры.
    @objc private func autoTuneNow() {
        guard !framesStore.isEmpty else {
            log("Авто-расчёт: нет кадров — сначала извлеките видео")
            return
        }
        autoTuneParams(frames: framesStore.count)
    }

    @objc private func resetParams() {
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

    private func finishTraining(ok: Bool) {
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

    private func log(_ s: String) {
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

    /// Строка из core-кода (yumeLog): уже с меткой времени и уже записана в
    /// train.log — здесь только показываем её в журнале и отдаём через API.
    private func appendCoreLog(_ s: String) {
        api.appendLog(s)
        logView.string += s + "\n"
        let lines = logView.string.components(separatedBy: "\n")
        if lines.count > 1500 {
            logView.string = lines.suffix(1500).joined(separator: "\n") + "\n"
        }
        logView.scrollToEndOfDocument(nil)
    }
}

let app = NSApplication.shared
app.appearance = NSAppearance(named: .aqua)
setbuf(stdout, nil)   // немедленный вывод логов API в терминал/файл
let delegate = YumeAppDelegate()
app.delegate = delegate
app.run()
