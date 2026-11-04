import Foundation

/// Bounded HTTP/1.1 framing for the local diagnostic API; nil means more bytes are needed.
public enum LocalHTTPRequest {
    public enum Invalid: Error { case request }

    public static func parse(_ data: Data, port: UInt16 = 17851) throws -> String? {
        guard data.count <= 65536 else { throw Invalid.request }
        guard let boundary = data.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        guard let header = String(data: data[..<boundary.lowerBound], encoding: .utf8) else {
            throw Invalid.request
        }
        let lines = header.components(separatedBy: "\r\n")
        let requestLine = lines[0].split(separator: " ")
        guard requestLine.count == 3, requestLine[2] == "HTTP/1.1" else { throw Invalid.request }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { throw Invalid.request }
            let name = line[..<colon].lowercased()
            guard headers[name] == nil else { throw Invalid.request }
            headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        // Белый список Host привязан к фактическому порту: сервер принимает
        // только запросы к самому себе (второй экземпляр на другом порту —
        // тоже легитимен, но только для своего порта).
        let allowedHosts = ["127.0.0.1:\(port)", "localhost:\(port)"]
        guard let host = headers["host"], allowedHosts.contains(host.lowercased()),
              headers["transfer-encoding"] == nil else { throw Invalid.request }
        if let origin = headers["origin"], origin != "http://" + host { throw Invalid.request }
        let lengthText = headers["content-length"] ?? "0"
        guard !lengthText.isEmpty, lengthText.utf8.allSatisfy({ (48...57).contains($0) }),
              let length = Int(lengthText), length <= 65536 - boundary.upperBound else {
            throw Invalid.request
        }
        let expected = boundary.upperBound + length
        guard data.count >= expected else { return nil }
        guard data.count == expected, let text = String(data: data, encoding: .utf8) else {
            throw Invalid.request
        }
        return text
    }
}
