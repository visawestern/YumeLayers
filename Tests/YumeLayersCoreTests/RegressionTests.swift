import XCTest
import Metal
@testable import YumeLayersCore

final class RegressionTests: XCTestCase {
    func testHTTPWaitsForWholeBody() throws {
        let head = "POST /api/extract HTTP/1.1\r\nHost: 127.0.0.1:17851\r\nContent-Length: 2\r\n\r\n"
        XCTAssertNil(try LocalHTTPRequest.parse(Data((head + "{").utf8)))
        XCTAssertNotNil(try LocalHTTPRequest.parse(Data((head + "{}").utf8)))
    }
    func testHTTPRejectsRemoteOriginsAndRebinding() {
        for extra in ["Host: evil.example", "Host: localhost:17851\r\nOrigin: https://evil.example",
                      "Host: localhost:17851\r\nContent-Length: -1",
                      "Host: localhost:17851\r\nContent-Length: 0\r\nContent-Length: 2",
                      "Host: localhost:17851\r\nTransfer-Encoding: chunked"] {
            XCTAssertThrowsError(try LocalHTTPRequest.parse(Data("POST /api/train HTTP/1.1\r\n\(extra)\r\n\r\n".utf8)))
        }
    }
    func testHTTPRejectsOversizeAndPipelining() {
        XCTAssertThrowsError(try LocalHTTPRequest.parse(Data(repeating: 65, count: 65537)))
        XCTAssertThrowsError(try LocalHTTPRequest.parse(Data("GET /api/status HTTP/1.1\r\nHost: localhost:17851\r\n\r\nextra".utf8)))
    }
    func testCheckpointRoundTripAndMalformedLength() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try saveState([Matrix([1, 2, 3, 4], rows: 2)], to: url.path, epoch: 3)
        let loaded = try loadStateWithMeta(path: url.path)
        XCTAssertEqual(loaded.mats[0].data, [1, 2, 3, 4])
        XCTAssertEqual(loaded.epoch, 3)
        var bad = Data("<safeten".utf8)
        bad.append(Data(repeating: 255, count: 8))
        try bad.write(to: url)
        XCTAssertThrowsError(try loadState(path: url.path))
    }
    func testCheckpointRejectsZeroShape() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        let header = Data("{\"p0\":{\"dtype\":\"F32\",\"shape\":[0,2],\"data_offsets\":[0,0]}}".utf8)
        var data = Data("<safeten".utf8)
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(header)
        try data.write(to: url)
        XCTAssertThrowsError(try loadState(path: url.path))
    }
    func testFrameCacheCyclesAndRandomAccess() throws {
        let png = try XCTUnwrap(encodePNG(width: 2, height: 2, rgb: Array(repeating: 0.5, count: 12)))
        let frames = (0..<5).map { MemoryFrame(name: "\($0)", jpeg: png) }
        let cache = try XCTUnwrap(FrameCache(frames: frames, w: 2, h: 2, budgetBytes: 48))
        // FrameCache pads to multiple of 8: 2x2 -> 8x8, 8*8*3 = 192
        let paddedSize = ((2 + 7) / 8 * 8) * ((2 + 7) / 8 * 8) * 3
        for index in [0, 1, 2, 3, 4, 0, 4, 1, 0] {
            XCTAssertEqual(cache.floatRow(at: index)?.count, paddedSize)
        }
    }
    func testBufferPoolNeverLoansSameBufferTwice() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let pool = BufferPool(MetalEngine.shared)
        let initial = pool.get(16)
        pool.put(initial)
        let a = pool.get(16), b = pool.get(16)
        XCTAssertTrue(a === initial)
        XCTAssertFalse(a === b)
    }
    func testMultipleUploadsRemainDistinctAfterReuse() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let engine = MetalEngine.shared, session = GpuSession(MetalEngine.shared)
        let a = GPUTensor(engine: engine, rows: 1, cols: 4)
        let b = GPUTensor(engine: engine, rows: 1, cols: 4)
        for _ in 0..<3 {
            session.begin()
            session.upload(a, from: [1, 2, 3, 4])
            session.upload(b, from: [5, 6, 7, 8])
            _ = session.end()
            XCTAssertEqual(a.toArray(), [1, 2, 3, 4])
            XCTAssertEqual(b.toArray(), [5, 6, 7, 8])
        }
    }
    func testBiasBroadcastUsesColumnCount() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let engine = MetalEngine.shared, session = GpuSession(MetalEngine.shared)
        let input = GPUTensor(engine: engine, data: [1, 2, 3, 4, 5, 6], rows: 2)
        let bias = GPUTensor(engine: engine, data: [10, 20, 30], rows: 1)
        session.begin()
        let result = session.addBias(input, bias)
        _ = session.end()
        XCTAssertEqual(result.toArray(), [11, 22, 33, 14, 25, 36])
    }
    func testGPUFramePoolReloadsOriginalFramesOnSecondPass() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let engine = MetalEngine.shared
        let frames = try [Float(0.1), 0.9, 0.4].enumerated().map { index, value in
            MemoryFrame(name: "\(index)", jpeg: try XCTUnwrap(encodePNG(width: 8, height: 8, rgb: Array(repeating: value, count: 192))))
        }
        let pool = try XCTUnwrap(GPUFramePool(engine: engine, frames: frames, w: 8, h: 8, size: 2))
        let first = (0..<3).map { _ in pool.next().toArray() }
        let second = (0..<3).map { _ in pool.next().toArray() }
        XCTAssertEqual(first, second)
        XCTAssertNotEqual(first[0], first[1])
    }
}
