// YumeLayers UI — скриншот текущего кадра → опорная A/B интерполяции.
// Вынесено из DelegatePart4.swift для соблюдения лимита 600 строк на файл.
import AppKit
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

extension YumeAppDelegate {
    // MARK: - Скриншот кадра → опорная A/B

    /// Активность кнопок «Кадр → A/B»: нужен выбранный кадр в коллаже.
    func updateShotButtons() {
        let hasFrame = selectedFrameIndex >= 0 && selectedFrameIndex < framesStore.count
        shotToAButton.isEnabled = hasFrame
        shotToBButton.isEnabled = hasFrame
    }

    @objc func shotToReferenceA() { saveSelectedFrameAsReference(isA: true) }
    @objc func shotToReferenceB() { saveSelectedFrameAsReference(isA: false) }

    /// Сохраняет текущий выбранный кадр как PNG в папку скриншотов и сразу
    /// ставит его путь в опорную A (isA=true) или B (isA=false).
    private func saveSelectedFrameAsReference(isA: Bool) {
        guard selectedFrameIndex >= 0, selectedFrameIndex < framesStore.count else {
            setStatus("Сначала выберите кадр в коллаже", ok: false)
            return
        }
        let frame = framesStore[selectedFrameIndex]
        let name = String(format: "frame_%04d_%@.png", selectedFrameIndex, isA ? "A" : "B")
        let path = (screenshotsDir as NSString).appendingPathComponent(name)
        guard let png = jpegToPNG(frame.jpeg) else {
            setStatus("Не удалось сохранить скриншот кадра", ok: false)
            log("⚠️ Скриншот: не удалось конвертировать JPEG→PNG кадра \(frame.name)")
            return
        }
        do {
            try png.write(to: URL(fileURLWithPath: path))
        } catch {
            setStatus("Не удалось записать скриншот: \(error.localizedDescription)", ok: false)
            return
        }
        if isA {
            interpImageAPath = path
            interpALabel.stringValue = "кадр \(selectedFrameIndex + 1)"
            interpALabel.toolTip = path
        } else {
            interpImageBPath = path
            interpBLabel.stringValue = "кадр \(selectedFrameIndex + 1)"
            interpBLabel.toolTip = path
        }
        updateInterpolateButtons()
        setStatus("Скриншот кадра \(selectedFrameIndex + 1) → \(isA ? "A" : "B")", ok: true)
        log("📸 Скриншот кадра \(selectedFrameIndex + 1) → \(isA ? "A" : "B"): \(path)")
    }

    /// Конвертация JPEG-данных в PNG-данные (ImageIO, без декодирования в NSImage).
    private func jpegToPNG(_ jpeg: Data) -> Data? {
        guard let src = CGImageSourceCreateWithData(jpeg as CFData, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let out = NSMutableData()
        guard let dst = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dst, img, nil)
        guard CGImageDestinationFinalize(dst) else { return nil }
        return out as Data
    }
}
