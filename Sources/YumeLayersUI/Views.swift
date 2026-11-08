// YumeLayers UI — переиспользуемые view-компоненты.
// Выделено из main.swift для соблюдения лимита 600 строк/файл.
import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

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

