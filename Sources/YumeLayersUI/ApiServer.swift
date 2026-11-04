// YumeLayers UI — локальный HTTP API (только 127.0.0.1) для управления обучением.
// Методы:
//   GET  /api/status           — состояние обучения, последние метрики
//   GET  /api/log?lines=100    — последние строки журнала
//   GET  /api/preview          — путь к последнему превью (PNG)
//   POST /api/train            — запуск обучения (JSON-параметры)
//   POST /api/stop             — остановка обучения
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

    var onTrain: (() -> Void)?
    var onStop: (() -> Void)?
    var onExtract: (([String], Int?) -> Void)?
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
            listener.newConnectionHandler = { [weak self] conn in
                self?.handle(conn)
            }
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    print("API listener: готов на порту \(self.port)")
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
        case ("GET", "/api/status"):
            send(status: 200, body: statusJSON(), to: conn)
        case ("GET", "/api/log"):
            let linesCount = queryInt(pathWithQuery, "lines", def: 100)
            send(status: 200, body: logJSON(lines: linesCount), to: conn)
        case ("GET", "/api/preview"):
            send(status: 200, body: previewJSON(), to: conn)
        case ("GET", "/api/screenshot"):
            if let path = screenshotProvider?() {
                send(status: 200, body: json(["path": path]), to: conn)
            } else {
                send(status: 500, body: "{\"error\":\"no window\"}", to: conn)
            }
        case ("GET", "/api/layout"):
            if let json = layoutProvider?() {
                send(status: 200, body: json, to: conn)
            } else {
                send(status: 500, body: "{\"error\":\"no layout\"}", to: conn)
            }
        case ("GET", "/api/state"):
            if let json = stateProvider?() {
                send(status: 200, body: json, to: conn)
            } else {
                send(status: 500, body: "{\"error\":\"no state\"}", to: conn)
            }
        case ("POST", "/api/train"):
            // Обучение запускается ТОЙ ЖЕ функцией и с ТЕМИ ЖЕ параметрами,
            // что и кнопка «Обучить VAE» в интерфейсе. Никаких параметров из
            // тела запроса — чтобы изменить параметры, их правят в UI
            // (кнопка редактирования). Это исключает расхождение API и кнопки.
            onTrain?()
            send(status: 200, body: "{\"ok\":true}", to: conn)
        case ("POST", "/api/stop"):
            onStop?()
            send(status: 200, body: "{\"ok\":true}", to: conn)
        case ("POST", "/api/extract"):
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
        default:
            send(status: 404, body: "{\"error\":\"not found\"}", to: conn)
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
        "extracting":\(_isExtracting),"extractProgress":\(_extractProgress)}
        """
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
