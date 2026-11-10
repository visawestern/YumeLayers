// YumeLayers UI — дополнительные маршруты API (вынесены из ApiServer.swift
// для соблюдения лимита 600 строк на файл).
import Foundation
import Network
import YumeLayersCore

extension ApiServer {
    /// GET /api/references — текущие опорные кадры A/B.
    /// POST /api/references — установить: {"imageA": путь, "imageB": путь}.
    func route__api_references(conn: NWConnection, body: String, pathWithQuery: String) {
        guard let refs = referencesProvider?() else {
            send(status: 500, body: "{\"error\":\"no references provider\"}", to: conn)
            return
        }
        guard !body.isEmpty else {
            send(status: 200, body: json(refs), to: conn)
            return
        }
        guard let data = body.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let setter = onSetReferences else {
            send(status: 400, body: "{\"error\":\"bad params\"}", to: conn)
            return
        }
        let r = setter(obj)
        send(status: r.ok ? 200 : 400,
             body: json(["ok": r.ok, "applied": r.applied, "unknown": r.unknown]),
             to: conn)
    }

    /// POST /api/train-motion — обучение сети движения на текущих кадрах.
    /// Тело: {"steps": N, "window": W} (необязательны; window — длина окна
    /// движения = число кадров целевого интервала A→B).
    func route__api_train_motion(conn: NWConnection, body: String, pathWithQuery: String) {
        var steps = 300
        var window: Int? = nil
        if let data = body.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            steps = max(1, (obj["steps"] as? Int) ?? 300)
            if let w = obj["window"] as? Int { window = max(8, min(128, w)) }
        }
        guard let callback = onTrainMotion else {
            send(status: 500, body: "{\"error\":\"no train-motion handler\"}", to: conn)
            return
        }
        callback(steps, window)
        send(status: 200, body: "{\"ok\":true,\"steps\":\(steps)}", to: conn)
    }
}
