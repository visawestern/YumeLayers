// YumeLayers UI — дизайн-токены (Aave-система).
//
// Источник: aave.com style reference.
// Двойные поверхности: светлая (#ffffff / лаванда #ece6ff) ↔ тёмная
// (#221d1d / #0f0f10). Единый акцент — Aave Violet #998eff.
// Таблетки-кнопки: radius 50px (вторичные) / 1584px (полная таблетка).
// Карточки: 20px. Типографика 400–500, плотный трекинг.
import AppKit

enum Design {
    // Colors
    static let violet = NSColor(calibratedRed: 0x99 / 255.0, green: 0x8E / 255.0, blue: 0xFF / 255.0, alpha: 1)
    static let obsidian = NSColor(calibratedRed: 0x22 / 255.0, green: 0x1D / 255.0, blue: 0x1D / 255.0, alpha: 1)
    static let inkwell = NSColor(calibratedRed: 0x0F / 255.0, green: 0x0F / 255.0, blue: 0x10 / 255.0, alpha: 1)
    static let paper = NSColor.white
    static let lavender = NSColor(calibratedRed: 0xEC / 255.0, green: 0xE6 / 255.0, blue: 0xFF / 255.0, alpha: 1)
    static let bone = NSColor(calibratedRed: 0xF6 / 255.0, green: 0xF7 / 255.0, blue: 0xF4 / 255.0, alpha: 1)
    static let iron = NSColor(calibratedRed: 0x63 / 255.0, green: 0x61 / 255.0, blue: 0x61 / 255.0, alpha: 1)
    static let ash = NSColor(calibratedRed: 0xBC / 255.0, green: 0xBB / 255.0, blue: 0xBB / 255.0, alpha: 1)
    static let graphite = NSColor(calibratedRed: 0x85 / 255.0, green: 0x83 / 255.0, blue: 0x87 / 255.0, alpha: 1)

    // Radii
    static let cardRadius: CGFloat = 20
    static let inputRadius: CGFloat = 20
    static let pillSecondary: CGFloat = 50
    static let pillFull: CGFloat = 1584

    /// Первичная таблеточная кнопка (violet fill / белый текст).
    static func primaryPill(title: String, target: AnyObject?, action: Selector) -> NSButton {
        let b = NSButton(title: title, target: target, action: action)
        applyPill(b, fill: violet, text: .white, radius: pillFull)
        return b
    }

    /// Тёмная инвертированная таблетка (inkwell fill / белый текст).
    static func darkPill(title: String, target: AnyObject?, action: Selector) -> NSButton {
        let b = NSButton(title: title, target: target, action: action)
        applyPill(b, fill: inkwell, text: .white, radius: pillSecondary)
        return b
    }

    /// Светлая таблетка (paper fill / obsidian текст) — на тёмной поверхности.
    static func lightPill(title: String, target: AnyObject?, action: Selector) -> NSButton {
        let b = NSButton(title: title, target: target, action: action)
        applyPill(b, fill: paper, text: obsidian, radius: pillSecondary)
        return b
    }

    /// Обводная (ghost) таблетка.
    static func ghostPill(title: String, target: AnyObject?, action: Selector, dark: Bool = true) -> NSButton {
        let b = NSButton(title: title, target: target, action: action)
        let stroke = dark ? NSColor.white : obsidian
        applyOutlinePill(b, stroke: stroke, text: stroke)
        return b
    }

    static func applyPill(_ b: NSButton, fill: NSColor, text: NSColor, radius: CGFloat) {
        b.bezelStyle = .rounded
        b.isBordered = false
        b.wantsLayer = true
        b.layer?.backgroundColor = fill.cgColor
        b.layer?.cornerRadius = radius
        b.contentTintColor = text
        b.attributedTitle = NSAttributedString(
            string: b.title,
            attributes: [.foregroundColor: text,
                         .font: NSFont.systemFont(ofSize: 15, weight: .medium)])
        b.setButtonType(.momentaryPushIn)
    }

    static func applyOutlinePill(_ b: NSButton, stroke: NSColor, text: NSColor) {
        b.bezelStyle = .rounded
        b.isBordered = false
        b.wantsLayer = true
        b.layer?.borderWidth = 1
        b.layer?.borderColor = stroke.cgColor
        b.layer?.cornerRadius = pillSecondary
        b.contentTintColor = text
        b.attributedTitle = NSAttributedString(
            string: b.title,
            attributes: [.foregroundColor: text,
                         .font: NSFont.systemFont(ofSize: 15, weight: .medium)])
        b.setButtonType(.momentaryPushIn)
    }
}
