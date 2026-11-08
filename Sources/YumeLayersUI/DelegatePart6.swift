import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

extension YumeAppDelegate {
        func groupCard(title: String, contentViews: [NSView], spacing: CGFloat) -> NSView {
            let card = NSVisualEffectView()
            card.material = .contentBackground
            card.blendingMode = .withinWindow
            card.state = .active
            card.wantsLayer = true
            card.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
            card.layer?.cornerRadius = Design.cardRadius
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

        func mediaCard(_ content: NSView, _ empty: NSStackView) -> NSView {
            let card = NSView()
            card.wantsLayer = true
            card.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
            card.layer?.cornerRadius = Design.cardRadius
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

        func photoBadge() -> NSImageView {
            let b = NSImageView()
            b.image = NSImage(systemSymbolName: "checkmark.circle.fill", accessibilityDescription: nil)
            b.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 14, weight: .bold)
            b.contentTintColor = .systemGreen
            b.translatesAutoresizingMaskIntoConstraints = false
            b.isHidden = true   // показывается только после обработки моделью
            return b
        }

}
