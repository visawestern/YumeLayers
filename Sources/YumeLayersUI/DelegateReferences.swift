// YumeLayers UI — /api/references: чтение/установка опорных кадров A/B.
// Вынесено из DelegatePart4.swift для соблюдения лимита 600 строк на файл.
import AppKit
import YumeLayersCore

extension YumeAppDelegate {
    /// Путь к чекпоинту сети движения рядом с чекпоинтом VAE:
    /// conv.safetensors → conv.motion.safetensors (тот же каталог).
    func motionCheckpointPath(for convPath: String) -> String {
        let path = (convPath as NSString).expandingTildeInPath
        let dir = (path as NSString).deletingLastPathComponent
        let file = (path as NSString).lastPathComponent
        let base = (file as NSString).deletingPathExtension
        return (dir as NSString).appendingPathComponent("\(base).motion.safetensors")
    }

    /// Отдельный запуск обучения сети движения (кнопка/API) на текущих кадрах.
    func trainMotionNow(steps: Int, window: Int? = nil) {
        guard !framesStore.isEmpty else {
            setStatus("Движение: нет кадров — сначала извлеките видео", ok: false)
            return
        }
        let checkpoint = bestCheckpointPath(for: (convPathCtl.stringValue as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: checkpoint) else {
            setStatus("Движение: сначала обучите VAE", ok: false)
            return
        }
        var p = Params()
        p.convPath = checkpoint
        p.width = widthCtl.intValue(def: 80)
        p.height = heightCtl.intValue(def: 56)
        p.memoryFrames = framesStore
        // Окно вычисляем на main (чтение UI из фона небезопасно).
        let w = max(8, min(128, window ?? currentInterpolationWindow()))
        setStatus("Обучение движения…", ok: true)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.trainMotionAfterVae(p: p, steps: max(1, steps), window: w)
        }
    }

    /// Регистрирует referencesProvider/onSetReferences (GET/POST /api/references).
    func configureReferencesCallback() {
        api.referencesProvider = { [weak self] in
            guard let self = self else { return [:] }
            var refs: [String: Any] = [:]
            DispatchQueue.main.sync {
                refs["imageA"] = self.interpImageAPath
                refs["imageB"] = self.interpImageBPath
            }
            return refs
        }
        api.onSetReferences = { [weak self] params in
            guard let self = self else { return (false, [], params.keys.map { "\($0)" }) }
            var applied: [String] = []
            var unknown: [String] = []
            DispatchQueue.main.sync {
                if let a = params["imageA"] as? String, !a.isEmpty {
                    self.interpImageAPath = a
                    self.interpALabel.stringValue = (a as NSString).lastPathComponent
                    self.interpALabel.toolTip = a
                    applied.append("imageA")
                } else if params["imageA"] != nil {
                    unknown.append("imageA")
                }
                if let b = params["imageB"] as? String, !b.isEmpty {
                    self.interpImageBPath = b
                    self.interpBLabel.stringValue = (b as NSString).lastPathComponent
                    self.interpBLabel.toolTip = b
                    applied.append("imageB")
                } else if params["imageB"] != nil {
                    unknown.append("imageB")
                }
                self.updateInterpolateButtons()
            }
            return (applied.isEmpty ? false : unknown.isEmpty, applied, unknown)
        }
    }
}
