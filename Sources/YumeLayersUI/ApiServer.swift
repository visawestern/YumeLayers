// YumeLayers UI — локальный HTTP API (только 127.0.0.1) для управления обучением.
// Методы:
//   GET  /api/status           — состояние обучения, последние метрики
//   GET  /api/log?lines=100    — последние строки журнала
//   GET  /api/preview          — путь к последнему превью (PNG)
//   GET  /api/interpolation    — итог последней интерполяции А→Б
//   POST /api/train            — запуск обучения (JSON-параметры)
//   POST /api/stop             — остановка обучения
//   POST /api/interpolate      — интерполяция А→Б (JSON-параметры)
//   POST /api/interpolate/stop — остановка интерполяции
import Foundation
import Network
import YumeLayersCore

final class ApiServer {
    private let queue = DispatchQueue(label: "yume.api", qos: .userInitiated)
    private var listener: NWListener?
    /// Порт API. По умолчанию 17851; YUME_API_PORT позволяет поднять второй
    /// экземпляр для диагностики, не мешая уже запущенному приложению.
    private let port: UInt16 = UInt16(ProcessInfo.processInfo.environment["YUME_API_PORT"] ?? "") ?? 17851
    private let lock = NSLock()

    private var _isTraining = false
    private var _epoch = 0
    private var _totalEpochs = 0
    private var _loss: Double = 0
    private var _psnr: Double = 0
    private var _progress: Double = 0
    private var _log: [String] = []
    private var _lastPreview: String?
    private var _isExtracting = false
    private var _extractProgress: Double = 0
    private var _isInterpolating = false
    private var _interpolationProgress: Double = 0
    private var _lastInterpolation: [String: Any]?
    private var _lastLayered: [String: Any]?

    var onTrain: ((Int, Bool) -> Void)?
    var onStop: (() -> Void)?
    var onExtract: (([String], Int?) -> Void)?
    /// Запуск интерполяции А→Б: (путь A, путь B, длительность, fps, путь результата).
    /// Последний параметр может быть nil — тогда файл кладётся в ~/YumeLayers/outputs.
    var onInterpolate: ((String, String, Double, Int, String?) -> Void)?
    /// Остановка текущей интерполяции.
    var onInterpolateStop: (() -> Void)?
    /// Запуск слойного контура (видео → слои → обучение → превью):
    /// (видео, w, h, шаги, ёмкость латента, чекпоинт).
    var onLayeredTrain: ((String?, Int, Int, Int, Int, String?) -> Void)?
    /// Сохранение реконструкций на диск. Тело: {"dir": путь} (необязательно).
    var onSaveRecons: ((String?) -> Void)?
    /// Применение произвольного набора параметров интерфейса.
    /// Возвращает (ok, применено, нераспознано). Вызывается на главном потоке.
    var onSetParam: (([String: Any]) -> (ok: Bool, applied: [String], unknown: [String]))?
    /// Сброс параметров к умолчаниям (кнопка «Сброс»).
    var onResetParams: (() -> Void)?
    /// Авто-расчёт параметров под текущие кадры (кнопка «Авто-расчёт»).
    var onAutoTune: (() -> Void)?
    /// Прогон проверки текущих кадров: возвращает JSON-отчёт.
    var testFramesProvider: ((Int) -> String)?
    var screenshotProvider: (() -> String?)?
    var layoutProvider: (() -> String)?
    var stateProvider: (() -> String)?

    init() {
        print("YumeLayers API: http://127.0.0.1:\(port)")
    }

    func start() -> Bool {
        do {
            let parameters = NWParameters.tcp
            parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
            let listener = try NWListener(using: parameters)
            let portValue = port
            listener.newConnectionHandler = { [weak self] conn in
                self?.handle(conn)
            }
            // [weak self] + захваченный порт: сильное захватывание self здесь
            // давало цикл listener → stateUpdateHandler → ApiServer.
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    print("API listener: готов на порту \(portValue)")
                case .failed(let err):
                    print("API listener error: \(err)")
                default:
                    break
                }
            }
            listener.start(queue: queue)
            self.listener = listener
            fflush(stdout)
            return true
        } catch {
            print("Не удалось поднять API: \(error)")
            fflush(stdout)
            return false
        }
    }

    // MARK: - Запись состояния (с любых потоков)

    func setTrainingActive(_ active: Bool, totalEpochs: Int) {
        lock.lock(); defer { lock.unlock() }
        _isTraining = active
        _totalEpochs = totalEpochs
        if active { _epoch = 0; _loss = 0; _psnr = 0; _progress = 0 }
    }

    func updateProgress(epoch: Int, loss: Double, psnr: Double, progress: Double) {
        lock.lock(); defer { lock.unlock() }
        _epoch = epoch
        _loss = loss
        _psnr = psnr
        _progress = progress
    }

    func appendLog(_ s: String) {
        lock.lock(); defer { lock.unlock() }
        _log.append(s)
        if _log.count > 1500 { _log.removeFirst(_log.count - 1500) }
    }

    func setPreview(path: String?) {
        lock.lock(); defer { lock.unlock() }
        _lastPreview = path
    }

    func setExtractingActive(_ active: Bool) {
        lock.lock(); defer { lock.unlock() }
        _isExtracting = active
        if active { _extractProgress = 0 }
    }

    func updateExtractProgress(_ p: Double) {
        lock.lock(); defer { lock.unlock() }
        _extractProgress = p
    }

    /// Интерполяция идёт / не идёт.
    func setInterpolatingActive(_ active: Bool) {
        lock.lock(); defer { lock.unlock() }
        _isInterpolating = active
        if active { _interpolationProgress = 0 }
    }

    /// Прогресс интерполяции (0…1).
    func updateInterpolationProgress(_ p: Double) {
        lock.lock(); defer { lock.unlock() }
        _interpolationProgress = p
    }

    /// Итог последней интерполяции (путь, кадры, fps, размеры) или ошибка.
    func setLastInterpolation(_ info: [String: Any]?) {
        lock.lock(); defer { lock.unlock() }
        _lastInterpolation = info
    }

    /// Итог последнего слойного прогона (PSNR/SSIM, превью, чекпоинт).
    func setLastLayered(_ info: [String: Any]?) {
        lock.lock(); defer { lock.unlock() }
        _lastLayered = info
    }

    // MARK: - Обработка соединений

    private func handle(_ conn: NWConnection) {
        conn.start(queue: queue)
        queue.asyncAfter(deadline: .now() + 10) { conn.cancel() }
        receiveRequest(conn, buffered: Data())
    }

    private func receiveRequest(_ conn: NWConnection, buffered: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, complete, error in
            guard let self = self, error == nil else { conn.cancel(); return }
            var buffer = buffered
            if let data = data { buffer.append(data) }
            guard buffer.count <= 65536 else {
                self.send(status: 413, body: "{}", to: conn); return
            }
            do {
                if let request = try LocalHTTPRequest.parse(buffer, port: self.port) {
                    self.respond(to: request, connection: conn)
                } else if complete {
                    self.send(status: 400, body: "{}", to: conn)
                } else {
                    self.receiveRequest(conn, buffered: buffer)
                }
            } catch {
                self.send(status: 400, body: "{}", to: conn)
            }
        }
    }

    private func respond(to text: String, connection conn: NWConnection) {
        let lines = text.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { send(status: 400, body: "", to: conn); return }
        let parts = requestLine.components(separatedBy: " ")
        guard parts.count >= 2 else { send(status: 400, body: "", to: conn); return }
        let method = parts[0]
        let pathWithQuery = parts[1]
        let path = pathWithQuery.components(separatedBy: "?").first ?? pathWithQuery

        let bodyStart = text.range(of: "\r\n\r\n")
        let body: String
        if let bodyStart = bodyStart {
            body = String(text[bodyStart.upperBound...])
        } else {
            body = ""
        }

        switch (method, path) {
        case ("GET", "/api/status"), ("POST", "/api/status"):
            self.route__api_status(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/interpolation"), ("POST", "/api/interpolation"):
            self.route__api_interpolation(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/log"), ("POST", "/api/log"):
            self.route__api_log(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/preview"), ("POST", "/api/preview"):
            self.route__api_preview(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/screenshot"), ("POST", "/api/screenshot"):
            self.route__api_screenshot(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/layout"), ("POST", "/api/layout"):
            self.route__api_layout(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/state"), ("POST", "/api/state"):
            self.route__api_state(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/train"), ("POST", "/api/train"):
            self.route__api_train(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/stop"), ("POST", "/api/stop"):
            self.route__api_stop(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/extract"), ("POST", "/api/extract"):
            self.route__api_extract(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/set"), ("POST", "/api/set"):
            self.route__api_set(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/reset"), ("POST", "/api/reset"):
            self.route__api_reset(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/autotune"), ("POST", "/api/autotune"):
            self.route__api_autotune(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/interpolate"), ("POST", "/api/interpolate"):
            self.route__api_interpolate(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/interpolate/stop"), ("POST", "/api/interpolate/stop"):
            self.route__api_interpolate_stop(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/layered/train"), ("POST", "/api/layered/train"):
            self.route__api_layered_train(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/layered/result"), ("POST", "/api/layered/result"):
            self.route__api_layered_result(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/save-recons"), ("POST", "/api/save-recons"):
            self.route__api_save_recons(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/test-frames"), ("POST", "/api/test-frames"):
            self.route__api_test_frames(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        case ("GET", "/api/compose"), ("POST", "/api/compose"):
            self.route__api_compose(conn: conn, body: body, pathWithQuery: pathWithQuery); return
        default:
            send(status: 404, body: "{\"error\":\"not found\"}", to: conn)
        }
    }


    private func route__api_status(conn: NWConnection, body: String, pathWithQuery: String) {
            send(status: 200, body: statusJSON(), to: conn)
    }

    /// POST /api/compose — чистая композиция слоёв (Фаза 5):
    /// {"w":..,"h":..,"background":[...],"objects":[...],"mask":[...],"light":[...]}
    /// → {"composite":[...]} (3·w·h float32, [0…1]).
    private func route__api_compose(conn: NWConnection, body: String, pathWithQuery: String) {
        guard let data = body.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let w = json["w"] as? Int, w > 0,
              let h = json["h"] as? Int, h > 0,
              let bg = json["background"] as? [Float],
              let ob = json["objects"] as? [Float],
              let mask = json["mask"] as? [Float],
              let light = json["light"] as? [Float] else {
            send(status: 400, body: "{\"error\":\"bad request\"}", to: conn)
            return
        }
        let composite = composeLayers(w: w, h: h, background: bg, objects: ob,
                                      mask: mask, light: light)
        guard let out = try? JSONSerialization.data(withJSONObject: ["composite": composite]),
              let text = String(data: out, encoding: .utf8) else {
            send(status: 500, body: "{\"error\":\"serialize\"}", to: conn)
            return
        }
        send(status: 200, body: text, to: conn)
    }

    private func route__api_interpolation(conn: NWConnection, body: String, pathWithQuery: String) {
            send(status: 200, body: interpolationJSON(), to: conn)
    }

    private func route__api_log(conn: NWConnection, body: String, pathWithQuery: String) {
            let linesCount = queryInt(pathWithQuery, "lines", def: 100)
            send(status: 200, body: logJSON(lines: linesCount), to: conn)
    }

    private func route__api_preview(conn: NWConnection, body: String, pathWithQuery: String) {
            send(status: 200, body: previewJSON(), to: conn)
    }

    private func route__api_screenshot(conn: NWConnection, body: String, pathWithQuery: String) {
            if let path = screenshotProvider?() {
                send(status: 200, body: json(["path": path]), to: conn)
            } else {
                send(status: 500, body: "{\"error\":\"no window\"}", to: conn)
            }
    }

    private func route__api_layout(conn: NWConnection, body: String, pathWithQuery: String) {
            if let json = layoutProvider?() {
                send(status: 200, body: json, to: conn)
            } else {
                send(status: 500, body: "{\"error\":\"no layout\"}", to: conn)
            }
    }

    private func route__api_state(conn: NWConnection, body: String, pathWithQuery: String) {
            if let json = stateProvider?() {
                send(status: 200, body: json, to: conn)
            } else {
                send(status: 500, body: "{\"error\":\"no state\"}", to: conn)
            }
    }

    private func route__api_train(conn: NWConnection, body: String, pathWithQuery: String) {
            // Обучение запускается ТОЙ ЖЕ функцией и с ТЕМИ ЖЕ параметрами,
            // что и кнопка «Обучить VAE» в интерфейсе. Параметры из тела:
            // {"maxFrames": N} — учиться только на первых N кадрах
            // (0 — все кадры); {"resume": true} — продолжить с чекпоинта
            // без диалога (нужно для API: диалог никто не нажмёт).
            let limit: Int
            let resume: Bool
            if let data = body.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                limit = max(0, (obj["maxFrames"] as? Int) ?? 0)
                resume = (obj["resume"] as? Bool) ?? false
            } else {
                limit = 0
                resume = false
            }
            onTrain?(limit, resume)
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_stop(conn: NWConnection, body: String, pathWithQuery: String) {
            onStop?()
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_extract(conn: NWConnection, body: String, pathWithQuery: String) {
            guard let data = body.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                send(status: 400, body: "{\"error\":\"bad params\"}", to: conn); return
            }
            var paths: [String] = []
            if let arr = obj["paths"] as? [String] {
                paths = arr.filter { !$0.isEmpty }
            } else if let p = obj["path"] as? String, !p.isEmpty {
                paths = [p]
            }
            guard !paths.isEmpty else {
                send(status: 400, body: "{\"error\":\"bad params\"}", to: conn); return
            }
            let maxFrames = obj["maxFrames"] as? Int
            onExtract?(paths, maxFrames)
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_set(conn: NWConnection, body: String, pathWithQuery: String) {
            // Универсальный сеттер параметров интерфейса. Тело — словарь
            // «ключ → значение» (напр. {"lr": 0.01, "epochs": 6}).
            // Возвращает, какие ключи применены, а какие не распознаны.
            guard let data = body.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let setter = onSetParam else {
                send(status: 400, body: "{\"error\":\"bad params\"}", to: conn); return
            }
            let r = setter(obj)
            send(status: r.ok ? 200 : 400,
                 body: json(["ok": r.ok, "applied": r.applied, "unknown": r.unknown]),
                 to: conn)
    }

    private func route__api_reset(conn: NWConnection, body: String, pathWithQuery: String) {
            onResetParams?()
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_autotune(conn: NWConnection, body: String, pathWithQuery: String) {
            onAutoTune?()
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_interpolate(conn: NWConnection, body: String, pathWithQuery: String) {
            // Интерполяция между двумя опорными картинками.
            // Тело: {"imageA": путь, "imageB": путь, "seconds": N, "fps": N,
            //        "out": путь к .mp4 (необязательно)}.
            guard let data = body.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let imageA = obj["imageA"] as? String, !imageA.isEmpty,
                  let imageB = obj["imageB"] as? String, !imageB.isEmpty else {
                send(status: 400, body: "{\"error\":\"нужны imageA и imageB\"}", to: conn); return
            }
            let seconds = (obj["seconds"] as? Double) ?? 5
            let fps = (obj["fps"] as? Int) ?? 24
            let out = obj["out"] as? String
            onInterpolate?(imageA, imageB, seconds, fps, out)
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_interpolate_stop(conn: NWConnection, body: String, pathWithQuery: String) {
            onInterpolateStop?()
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_layered_train(conn: NWConnection, body: String, pathWithQuery: String) {
            // Слойный контур: видео → слои (фон/объекты/свет) → обучение → превью.
            // Тело: {"video": путь, "w": N, "h": N, "steps": N,
            //        "latentChannels": N, "checkpoint": путь}. Все кроме video
            // необязательны. Итог — через GET /api/layered/result.
            var video: String?
            var lw = 96, lh = 72, lsteps = 200, llatent = 32
            var lck: String?
            if let data = body.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                video = obj["video"] as? String
                lw = (obj["w"] as? Int) ?? 96
                lh = (obj["h"] as? Int) ?? 72
                lsteps = (obj["steps"] as? Int) ?? 200
                llatent = (obj["latentChannels"] as? Int) ?? 32
                lck = obj["checkpoint"] as? String
            }
            onLayeredTrain?(video, lw, lh, lsteps, llatent, lck)
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_layered_result(conn: NWConnection, body: String, pathWithQuery: String) {
            lock.lock(); defer { lock.unlock() }
            if let info = _lastLayered {
                send(status: 200, body: json(["result": info]), to: conn)
            } else {
                send(status: 200, body: "{\"result\":null}", to: conn)
            }
    }

    private func route__api_save_recons(conn: NWConnection, body: String, pathWithQuery: String) {
            // Сохранение реконструкций в памяти на диск.
            // Тело: {"dir": путь к папке} — папка необязательна.
            var dir: String?
            if let data = body.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                dir = obj["dir"] as? String
            }
            onSaveRecons?(dir)
            send(status: 200, body: "{\"ok\":true}", to: conn)
    }

    private func route__api_test_frames(conn: NWConnection, body: String, pathWithQuery: String) {
            // Проверка текущих кадров в памяти. Тело: {"count": N}
            // (0 — все кадры). Отчёт возвращается сразу.
            let count: Int
            if let data = body.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                count = (obj["count"] as? Int) ?? 0
            } else {
                count = 0
            }
            if let provider = testFramesProvider {
                send(status: 200, body: provider(count), to: conn)
            } else {
                send(status: 500, body: "{\"error\":\"no frames\"}", to: conn)
            }
    }

    private func queryInt(_ pathWithQuery: String, _ key: String, def: Int) -> Int {
        guard let q = pathWithQuery.components(separatedBy: "?").dropFirst().first else { return def }
        for pair in q.components(separatedBy: "&") {
            let kv = pair.components(separatedBy: "=")
            if kv.count == 2, kv[0] == key { return Int(kv[1]) ?? def }
        }
        return def
    }

    private func statusJSON() -> String {
        lock.lock(); defer { lock.unlock() }
        return """
        {"training":\(_isTraining),"epoch":\(_epoch),"totalEpochs":\(_totalEpochs),\
        "loss":\(_loss),"psnr":\(_psnr),"progress":\(_progress),\
        "extracting":\(_isExtracting),"extractProgress":\(_extractProgress),\
        "interpolating":\(_isInterpolating),"interpolationProgress":\(_interpolationProgress)}
        """
    }

    /// Итог последней интерполяции — отдельным JSON, чтобы не тащить
    /// вложенный объект в /api/status.
    private func interpolationJSON() -> String {
        lock.lock(); defer { lock.unlock() }
        guard let info = _lastInterpolation else { return "{\"result\":null}" }
        return json(["result": info])
    }

    private func logJSON(lines: Int) -> String {
        lock.lock(); defer { lock.unlock() }
        let tail = Array(_log.suffix(max(lines, 0)))
        return json(["lines": tail])
    }

    private func previewJSON() -> String {
        lock.lock(); defer { lock.unlock() }
        if let p = _lastPreview {
            return json(["path": p])
        }
        return "{\"path\":null}"
    }

    private func json(_ object: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8) else { return "{}" }
        return text
    }

    private func send(status: Int, body: String, to conn: NWConnection) {
        var headers = "HTTP/1.1 \(status) " + (status == 200 ? "OK" : "Error") + "\r\n"
        headers += "Content-Type: application/json\r\n"
        headers += "Content-Length: \(body.utf8.count)\r\n"
        headers += "Connection: close\r\n\r\n"
        let payload = headers + body
        conn.send(content: Data(payload.utf8), completion: .contentProcessed { _ in
            conn.cancel()
        })
    }
}
