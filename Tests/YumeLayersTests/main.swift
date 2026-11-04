// YumeLayers — тестовая сюита (собственный раннер).
//
// Компилируется вместе с Sources/YumeLayersCore/*.swift одним модулем,
// поэтому видит internal-API ядра без @testable. Запуск: scripts/test.sh
//
// Формат вывода намеренно машинно-читаемый: строки "ok/FAIL" + итог,
// код возврата 1 при любом падении (годится для CI и pre-commit).

import Foundation
import Metal

// MARK: - Мини-фреймворк

var gChecks = 0
var gFailures = 0
var gCurrentSection = ""
var gFilter: String? = CommandLine.arguments.dropFirst().first

func section(_ name: String) {
    gCurrentSection = name
    print("\n── \(name) ──")
}

@discardableResult
func expect(_ condition: Bool, _ name: String,
            file: StaticString = #file, line: UInt = #line) -> Bool {
    gChecks += 1
    if !condition {
        gFailures += 1
        let f = ("\(file)" as NSString).lastPathComponent
        print("FAIL  [\(gCurrentSection)] \(name)  (\(f):\(line))")
    } else {
        print("ok    \(name)")
    }
    return condition
}

func expectEqual<T: Equatable>(_ actual: T, _ expected: T, _ name: String,
                               file: StaticString = #file, line: UInt = #line) {
    if actual == expected {
        expect(true, name, file: file, line: line)
    } else {
        expect(false, "\(name) — ожидалось \(expected), получено \(actual)",
               file: file, line: line)
    }
}

func expectClose(_ actual: Double, _ expected: Double, _ tol: Double, _ name: String,
                 file: StaticString = #file, line: UInt = #line) {
    let ok = actual.isFinite && abs(actual - expected) <= tol
    expect(ok, "\(name) — ожидалось \(expected)±\(tol), получено \(actual)",
           file: file, line: line)
}

func expectThrows<T>(_ name: String, _ body: () throws -> T,
                     file: StaticString = #file, line: UInt = #line) {
    do {
        _ = try body()
        expect(false, "\(name) — ошибка не брошена", file: file, line: line)
    } catch {
        expect(true, name, file: file, line: line)
    }
}

/// Пропуск теста (нет GPU и т.п.) — не считается ни успехом, ни падением.
func skip(_ name: String) { print("skip  \(name)") }

let gHasMetal = MTLCreateSystemDefaultDevice() != nil

// MARK: - 1. HTTP-парсер локального API

func testHTTP() {
    section("HTTP: парсинг запросов")

    let head = "POST /api/extract HTTP/1.1\r\nHost: 127.0.0.1:17851\r\nContent-Length: 2\r\n\r\n"
    expect((try? LocalHTTPRequest.parse(Data((head + "{").utf8))) == nil,
           "неполное тело → nil (ждём ещё)")
    expect((try? LocalHTTPRequest.parse(Data((head + "{}").utf8))) != nil,
           "полное тело → запрос разобран")

    let bad: [String] = [
        "Host: evil.example",
        "Host: localhost:17851\r\nOrigin: https://evil.example",
        "Host: localhost:17851\r\nContent-Length: -1",
        "Host: localhost:17851\r\nContent-Length: 0\r\nContent-Length: 2",
        "Host: localhost:17851\r\nTransfer-Encoding: chunked",
        "Host: localhost:17852",
    ]
    for extra in bad {
        let raw = "POST /api/train HTTP/1.1\r\n\(extra)\r\n\r\n"
        expect((try? LocalHTTPRequest.parse(Data(raw.utf8))) == nil,
               "отклонён небезопасный заголовок: \(extra.replacingOccurrences(of: "\r\n", with: " | "))")
    }

    expect((try? LocalHTTPRequest.parse(Data(repeating: 65, count: 65537))) == nil,
           "тело больше лимита → отказ")
    expect((try? LocalHTTPRequest.parse(
        Data("GET /api/status HTTP/1.1\r\nHost: localhost:17851\r\n\r\nextra".utf8))) == nil,
           "лишние байты после тела (pipelining) → отказ")
    expect((try? LocalHTTPRequest.parse(
        Data("GET /api/state HTTP/1.1\r\nHost: localhost:17851\r\n\r\n".utf8), port: 17851)) != nil,
           "GET без Content-Length принимается")
    expect((try? LocalHTTPRequest.parse(
        Data("GET /api/state HTTP/1.1\r\nHost: localhost:17851\r\n\r\n".utf8), port: 17852)) == nil,
           "Host с чужим портом отклоняется")
}

// MARK: - 2. Чекпоинты

func testCheckpoints() {
    section("Чекпоинты: safetensors")

    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_ckpt_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let path = dir.appendingPathComponent("conv.safetensors").path
    do {
        try saveState([Matrix([1, 2, 3, 4], rows: 2)], to: path, epoch: 3, step: 7,
                      totalSteps: 10, dataFingerprint: "abc")
        let st = try loadStateWithMeta(path: path)
        expectEqual(st.mats.count, 1, "одна матрица прочитана")
        expectEqual(st.mats[0].data, [1, 2, 3, 4], "данные совпадают")
        expectEqual(st.mats[0].rows, 2, "число строк совпадает")
        expectEqual(st.epoch, 3, "эпоха восстановлена")
        expectEqual(st.step, 7, "шаг восстановлен")
        expectEqual(st.fingerprint, "abc", "отпечаток данных восстановлен")
    } catch {
        expect(false, "round-trip чекпоинта: \(error)")
    }

    // NaN в весах — чекпоинт не должен сохраняться.
    do {
        let nanPath = dir.appendingPathComponent("nan.safetensors").path
        expectThrows("NaN в весах → saveState бросает") {
            try saveState([Matrix([1, .nan], rows: 1)], to: nanPath)
        }
        expect(!FileManager.default.fileExists(atPath: nanPath),
               "файл с NaN не создан")
    }

    // Битый заголовок.
    do {
        let badPath = dir.appendingPathComponent("bad.safetensors").path
        var bad = Data("<safeten".utf8)
        bad.append(Data(repeating: 255, count: 8))
        try bad.write(to: URL(fileURLWithPath: badPath))
        expectThrows("битая длина заголовка → ошибка") { _ = try loadState(path: badPath) }
    } catch {
        expect(false, "подготовка битого файла: \(error)")
    }

    // Нулевая размерность.
    do {
        let zeroPath = dir.appendingPathComponent("zero.safetensors").path
        let header = Data("{\"p0\":{\"dtype\":\"F32\",\"shape\":[0,2],\"data_offsets\":[0,0]}}".utf8)
        var data = Data("<safeten".utf8)
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(header)
        try data.write(to: URL(fileURLWithPath: zeroPath))
        expectThrows("нулевая размерность → ошибка") { _ = try loadState(path: zeroPath) }
    } catch {
        expect(false, "подготовка нулевого шейпа: \(error)")
    }

    // Несуществующий файл.
    expectThrows("нет файла → ошибка") {
        _ = try loadState(path: dir.appendingPathComponent("nope.safetensors").path)
    }
}

// MARK: - 3. Бэкапы и файловые утилиты

func testBackups() {
    section("Бэкапы чекпоинтов")

    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_bak_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let existing = dir.appendingPathComponent("conv.safetensors").path
    let payload = Data("weights".utf8)
    try? payload.write(to: URL(fileURLWithPath: existing))
    let empty = dir.appendingPathComponent("empty.safetensors").path
    try? Data().write(to: URL(fileURLWithPath: empty))
    let missing = dir.appendingPathComponent("missing.safetensors").path

    expect(isNonEmptyFile(existing), "непустой файл распознан")
    expect(!isNonEmptyFile(empty), "пустой файл не считается непустым")
    expect(!isNonEmptyFile(missing), "отсутствующий файл не считается непустым")

    guard let b1 = createCheckpointBackup(fromPath: existing) else {
        expect(false, "бэкап непустого файла создан"); return
    }
    expect(FileManager.default.fileExists(atPath: b1), "бэкап существует на диске")
    expect((try? Data(contentsOf: URL(fileURLWithPath: b1))) == payload,
           "содержимое бэкапа совпадает")
    expect(b1 != existing, "имя бэкапа отличается от оригинала")

    let b2 = createCheckpointBackup(fromPath: existing)
    expect(b2 != nil && b2 != b1, "повторный бэкап в ту же секунду → другое имя")

    expect(createCheckpointBackup(fromPath: empty) == nil, "пустой файл не бэкапится")
    expect(createCheckpointBackup(fromPath: missing) == nil, "отсутствующий не бэкапится")
    expectEqual(createBackups(forPaths: [existing, empty, missing]).count, 1,
                "createBackups бэкапит только непустые")
}

// MARK: - 4. GPU: базовые операции

func testGPUOps() {
    section("GPU: базовые операции")

    guard gHasMetal else {
        skip("Metal недоступен — GPU-тесты пропущены")
        return
    }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // matmul: 7×5 × 5×9, сверка с CPU.
    do {
        let A = Matrix.random(rows: 7, cols: 5)
        let B = Matrix.random(rows: 5, cols: 9)
        session.begin()
        let C = session.matmul(GPUTensor(engine: engine, data: A.data, rows: 7),
                               GPUTensor(engine: engine, data: B.data, rows: 5))
        _ = session.end()
        let cpu = matmulCPU(A, B)
        let gpu = C.toArray()
        var maxDelta: Float = 0
        for i in 0..<cpu.data.count { maxDelta = max(maxDelta, abs(cpu.data[i] - gpu[i])) }
        expect(maxDelta < 1e-5, "matmul совпадает с CPU (delta \(maxDelta))")
    }

    // matmul с transA/transB.
    do {
        let M = 9, K = 6, N = 7
        let Araw = Matrix.random(rows: K, cols: M)   // транспонируем
        let Braw = Matrix.random(rows: N, cols: K)   // транспонируем
        session.begin()
        let C = session.matmul(GPUTensor(engine: engine, data: Araw.data, rows: K),
                               GPUTensor(engine: engine, data: Braw.data, rows: N),
                               transA: true, transB: true, M: M, N: N, K: K)
        _ = session.end()
        var expected = [Float](repeating: 0, count: M * N)
        for r in 0..<M {
            for c in 0..<N {
                var s: Float = 0
                for k in 0..<K { s += Araw.data[k * M + r] * Braw.data[c * K + k] }
                expected[r * N + c] = s
            }
        }
        let gpu = C.toArray()
        var maxDelta: Float = 0
        for i in 0..<expected.count { maxDelta = max(maxDelta, abs(expected[i] - gpu[i])) }
        expect(maxDelta < 1e-5, "matmul transA+transB совпадает с CPU (delta \(maxDelta))")
    }

    // Тайловый путь (m,n > 32, объём ≥ 1M) — краевые тайлы не кратны 16.
    do {
        let M = 37, K = 41, N = 53
        var a = [Float](repeating: 0, count: M * K)
        var b = [Float](repeating: 0, count: K * N)
        for i in 0..<a.count { a[i] = Float(i % 13) * 0.1 }
        for i in 0..<b.count { b[i] = Float(i % 7) * 0.2 }
        session.begin()
        let C = session.matmul(GPUTensor(engine: engine, data: a, rows: M),
                               GPUTensor(engine: engine, data: b, rows: K))
        _ = session.end()
        let gpu = C.toArray()
        var maxDelta: Float = 0
        for r in 0..<M {
            for c in 0..<N {
                var s: Float = 0
                for k in 0..<K { s += a[r * K + k] * b[k * N + c] }
                maxDelta = max(maxDelta, abs(s - gpu[r * N + c]))
            }
        }
        expect(maxDelta < 1e-4, "matmul 37×41×53 (краевые тайлы) совпадает с CPU (delta \(maxDelta))")
    }

    // addBias — broadcast по столбцам, а не по строкам.
    do {
        let input = GPUTensor(engine: engine, data: [1, 2, 3, 4, 5, 6], rows: 2)
        let bias = GPUTensor(engine: engine, data: [10, 20, 30], rows: 1)
        session.begin()
        let r = session.addBias(input, bias)
        _ = session.end()
        expectEqual(r.toArray(), [11, 22, 33, 14, 25, 36], "addBias транслирует по столбцам")
    }

    // sumCols — суммирование по строкам.
    do {
        let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  // 4×3
        session.begin()
        let r = session.sumCols(GPUTensor(engine: engine, data: data, rows: 4))
        _ = session.end()
        expectEqual(r.toArray(), [22, 26, 30], "sumCols суммирует по строкам")
    }

    // Активации.
    do {
        let x: [Float] = [-2, -0.5, 0, 0.5, 2]
        let t = GPUTensor(engine: engine, data: x, rows: 1)
        session.begin()
        let relu = session.relu(t)
        let sig = session.sigmoid(t)
        let th = session.tanh(t)
        _ = session.end()
        let r = relu.toArray(), s = sig.toArray(), thv = th.toArray()
        for i in 0..<x.count {
            expectClose(Double(r[i]), Double(max(0, x[i])), 1e-6, "relu[\(i)]")
            expectClose(Double(s[i]), 1.0 / (1.0 + exp(-Double(x[i]))), 1e-5, "sigmoid[\(i)]")
            expectClose(Double(thv[i]), tanh(Double(x[i])), 1e-5, "tanh[\(i)]")
        }
    }

    // reluGrad / sigmoidGrad / tanhGrad — против численной производной.
    do {
        let z: [Float] = [-1.3, -0.2, 0.4, 1.7]
        let g: [Float] = [1, 1, 1, 1]
        let zt = GPUTensor(engine: engine, data: z, rows: 1)
        let gt = GPUTensor(engine: engine, data: g, rows: 1)
        let reluOut = GPUTensor(engine: engine, data: z.map { max(0, $0) }, rows: 1)
        session.begin()
        let rg = session.reluGrad(reluOut, gt)
        let sg = session.sigmoidGrad(zt, gt)
        let tg = session.tanhGrad(zt)
        _ = session.end()
        let rgv = rg.toArray(), sgv = sg.toArray(), tgv = tg.toArray()
        for i in 0..<z.count {
            expectClose(Double(rgv[i]), z[i] > 0 ? 1 : 0, 1e-6, "reluGrad[\(i)]")
            let s = 1.0 / (1.0 + exp(-Double(z[i])))
            expectClose(Double(sgv[i]), s * (1 - s), 1e-5, "sigmoidGrad[\(i)]")
            let th = tanh(Double(z[i]))
            expectClose(Double(tgv[i]), 1 - th * th, 1e-5, "tanhGrad[\(i)]")
        }
    }

    // Adam — один шаг против эталона.
    do {
        let p: [Float] = [1, -2, 0.5, 0]
        let gr: [Float] = [0.1, -0.4, 0, 0.3]
        let pt = GPUTensor(engine: engine, data: p, rows: 1)
        let gt = GPUTensor(engine: engine, data: gr, rows: 1)
        let m = GPUTensor(engine: engine, rows: 1, cols: 4)
        let v = GPUTensor(engine: engine, rows: 1, cols: 4)
        session.begin()
        session.adamUpdate(pt, gt, m, v, t: 1, lr: 0.001)
        _ = session.end()
        let out = pt.toArray()
        for i in 0..<4 {
            let mm = 0.1 * gr[i]
            let vv = 0.001 * gr[i] * gr[i]
            let mHat = mm / 0.1
            let vHat = vv / 0.001
            let expected = p[i] - 0.001 * mHat / (sqrt(vHat) + 1e-8)
            expectClose(Double(out[i]), Double(expected), 1e-5, "adam[\(i)]")
        }
    }
}

// MARK: - 5. GPU: свёрточные примитивы

func testConvPrimitives() {
    section("GPU: свёрточные примитивы")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // im2col + col2im для stride=1 должны быть взаимно обратны (сумма перекрытий).
    do {
        let ic = 2, ih = 6, iw = 5, k = 3, pad = 1
        var x = [Float](repeating: 0, count: ic * ih * iw)
        for i in 0..<x.count { x[i] = Float(i) * 0.1 }
        let xt = GPUTensor(engine: engine, data: x, rows: ic)
        session.begin()
        let cols = session.im2col(xt, ih: ih, iw: iw, ic: ic, kh: k, kw: k, stride: 1, pad: pad)
        _ = session.end()
        let colsArr = cols.toArray()
        expectEqual(cols.rows, ic * k * k, "im2col: строк = ic*kh*kw")
        expectEqual(cols.cols, ih * iw, "im2col: столбцов = oh*ow")
        // Проверяем одну точку вручную: col=0 (c=0,ky=0,kx=0), sp=(0,0) → x[0,0]
        expectClose(Double(colsArr[0]), Double(x[0]), 1e-6, "im2col: левый верхний элемент")
        // (0,1) при pad=1 даёт x[-1,0] = 0
        expectClose(Double(colsArr[1]), 0, 1e-6, "im2col: паддинг даёт 0")
    }

    // upsample2x / downsample2x — размеры и суммы.
    do {
        let c = 2, h = 2, w = 3
        let x: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        let xt = GPUTensor(engine: engine, data: x, rows: c)
        session.begin()
        let up = session.upsample2x(xt, h: h, w: w)
        let down = session.downsample2x(up, h: h, w: w)
        _ = session.end()
        expectEqual(up.cols, 4 * h * w, "upsample2x: площадь ×4")
        expectEqual(down.cols, h * w, "downsample2x: площадь возвращается")
        let dv = down.toArray()
        // Каждый блок 2×2 суммируется → 4×исходное.
        for i in 0..<x.count {
            expectClose(Double(dv[i]), Double(x[i]) * 4, 1e-6, "downsample2x[\(i)] = 4×исходного")
        }
    }

    // gradW — против CPU (A·Bᵀ).
    do {
        let oc = 3, kk = 4, K = 25
        var a = [Float](repeating: 0, count: oc * K)
        var b = [Float](repeating: 0, count: kk * K)
        for i in 0..<a.count { a[i] = Float((i % 9) - 4) * 0.3 }
        for i in 0..<b.count { b[i] = Float((i % 7) - 3) * 0.2 }
        session.begin()
        let g = session.gradW(GPUTensor(engine: engine, data: a, rows: oc),
                              GPUTensor(engine: engine, data: b, rows: kk),
                              oc: oc, kk: kk)
        _ = session.end()
        let gv = g.toArray()
        var maxDelta: Float = 0
        for r in 0..<oc {
            for c in 0..<kk {
                var s: Float = 0
                for i in 0..<K { s += a[r * K + i] * b[c * K + i] }
                maxDelta = max(maxDelta, abs(s - gv[r * kk + c]))
            }
        }
        expect(maxDelta < 1e-5, "gradW совпадает с CPU (delta \(maxDelta))")
    }

    // biasRowGrad: plain-путь (cols ≤ 4096) и атомарный (cols > 4096).
    do {
        for cols in [64, 5000] {
            let rows = 3
            var data = [Float](repeating: 0, count: rows * cols)
            for i in 0..<data.count { data[i] = Float(i % 11) * 0.01 }
            session.begin()
            let g = session.biasRowGrad(GPUTensor(engine: engine, data: data, rows: rows))
            _ = session.end()
            let gv = g.toArray()
            var ok = true
            for r in 0..<rows {
                var s: Float = 0
                for c in 0..<cols { s += data[r * cols + c] }
                if abs(s - gv[r]) > 1e-3 { ok = false }
            }
            expect(ok, "biasRowGrad при cols=\(cols) совпадает с CPU")
        }
    }
}

// MARK: - 6. GPU: жизненный цикл буферов

func testBufferLifecycle() {
    section("GPU: буферы и сессия")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // Пул не выдаёт один буфер дважды.
    do {
        let pool = BufferPool(engine)
        let initial = pool.get(16)
        pool.put(initial)
        let a = pool.get(16), b = pool.get(16)
        expect(a === initial, "пул возвращает освобождённый буфер")
        expect(!(a === b), "пул не выдаёт один буфер дважды")
    }

    // Повторные upload в одну сессию не затирают друг друга.
    do {
        let a = GPUTensor(engine: engine, rows: 1, cols: 4)
        let b = GPUTensor(engine: engine, rows: 1, cols: 4)
        for _ in 0..<3 {
            session.begin()
            session.upload(a, from: [1, 2, 3, 4])
            session.upload(b, from: [5, 6, 7, 8])
            _ = session.end()
            expectEqual(a.toArray(), [1, 2, 3, 4], "upload a не затёрт")
            expectEqual(b.toArray(), [5, 6, 7, 8], "upload b не затёрт")
        }
    }

    // copy внутри сессии: копирует данные без отдельного command buffer.
    do {
        let src = GPUTensor(engine: engine, data: [1, 2, 3, 4, 5, 6], rows: 2)
        let dst = GPUTensor(engine: engine, rows: 2, cols: 3)
        session.begin()
        session.copy(dst, from: src)
        _ = session.end()
        expectEqual(dst.toArray(), [1, 2, 3, 4, 5, 6], "copy в сессии копирует данные")
    }

    // zeroFill внутри сессии: гарантированные нули поверх мусора.
    do {
        let t = GPUTensor(engine: engine, data: [7, 7, 7, 7, 7, 7], rows: 2)
        session.begin()
        session.zeroFill(t)
        _ = session.end()
        expectEqual(t.toArray(), [0, 0, 0, 0, 0, 0], "zeroFill обнуляет тензор")
    }

    // addInPlace после zeroFill даёт ровно второе слагаемое
    // (регрессия на «сумма с неинициализированной памятью»).
    do {
        let acc = GPUTensor(engine: engine, data: [123, -456, 789, 0.5], rows: 1)
        let g = GPUTensor(engine: engine, data: [1, 2, 3, 4], rows: 1)
        session.begin()
        session.zeroFill(acc)
        session.addInPlace(acc, g)
        _ = session.end()
        expectEqual(acc.toArray(), [1, 2, 3, 4], "zeroFill + addInPlace = точная копия")
    }
}

// MARK: - 7. Свёрточный VAE: форма и детерминизм

func testConvVAEShape() {
    section("ConvVAE: форма и детерминизм")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24

    let vae = ConvVAEGPU(engine: engine)
    let img = makeSyntheticScene(w: w, h: h, rng: &systemRNG)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h, deterministic: true)
    _ = session.end()

    expectEqual(cache.z.rows, ConvVAEGPU.latentC, "латент: C каналов")
    expectEqual(cache.z.cols, (w / 8) * (h / 8), "латент: (w/8)×(h/8)")
    expectEqual(cache.recon.rows, 3, "реконструкция: 3 канала")
    expectEqual(cache.recon.cols, w * h, "реконструкция: w×h")

    let recon = cache.recon.toArray()
    expect(recon.allSatisfy { $0.isFinite }, "реконструкция без NaN")
    expect(recon.allSatisfy { $0 >= 0 && $0 <= 1 }, "реконструкция в [0,1] (после сигмоиды)")
}

// MARK: - 7b. Вариационный путь: примитивы

/// Максимальное |a-b| по двум массивам (для сверки тензоров).
func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return .infinity }
    var m = 0.0
    for i in 0..<a.count { m = max(m, abs(Double(a[i] - b[i]))) }
    return m
}

func testVAEPrimitives() {
    section("VAE: примитивы")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // exp
    do {
        let a = GPUTensor(engine: engine, data: [0, 1, -2], rows: 1)
        session.begin()
        let e = session.exp(a)
        _ = session.end()
        let v = e.toArray()
        expectClose(Double(v[0]), 1.0, 1e-6, "exp(0)=1")
        expectClose(Double(v[1]), 2.718281828, 1e-5, "exp(1)=e")
        expectClose(Double(v[2]), 0.135335283, 1e-6, "exp(-2)")
    }

    // expHalf: σ = exp(0.5·logvar), logvar зажат в [-10, 10]
    do {
        let a = GPUTensor(engine: engine, data: [0, 1.386294361, 40, -40], rows: 1)
        session.begin()
        let e = session.expHalf(a)
        _ = session.end()
        let v = e.toArray()
        expectClose(Double(v[0]), 1.0, 1e-6, "σ(logvar=0)=1")
        expectClose(Double(v[1]), 2.0, 1e-5, "σ(logvar=ln4)=2")
        expectClose(Double(v[2]), exp(5.0), 1e-3, "σ: logvar зажат сверху (10)")
        expectClose(Double(v[3]), exp(-5.0), 1e-6, "σ: logvar зажат снизу (-10)")
    }

    // addConst
    do {
        let a = GPUTensor(engine: engine, data: [1, 2, 3], rows: 1)
        session.begin()
        let c = session.addConst(a, -1.5)
        _ = session.end()
        expectEqual(c.toArray(), [-0.5, 0.5, 1.5], "addConst: A + s")
    }

    // fma: C = A + B*D  (z = mu + σ·ε)
    do {
        let a = GPUTensor(engine: engine, data: [1, 2, 3], rows: 1)
        let b = GPUTensor(engine: engine, data: [2, 2, 2], rows: 1)
        let d = GPUTensor(engine: engine, data: [0.5, 1, -1], rows: 1)
        session.begin()
        let c = session.fma(a, b, d)
        _ = session.end()
        expectEqual(c.toArray(), [2, 4, 1], "fma: A + B*D")
    }

    // split2 / merge2 — взаимно обратны
    do {
        let c = 3, plane = 5
        var raw = [Float]()
        for ch in 0..<(2 * c) { for i in 0..<plane { raw.append(Float(ch * 100 + i)) } }
        let x = GPUTensor(engine: engine, data: raw, rows: 2 * c)
        session.begin()
        let (mu, lv) = session.split2(x, c: c, plane: plane)
        let back = session.merge2(mu, lv, c: c, plane: plane)
        _ = session.end()
        expectEqual(maxAbsDiff(back.toArray(), raw), 0, "merge2∘split2 = тождество")
        expectEqual(mu.toArray().first!, 0, "split2: mu[0] = X[0]")
        expectEqual(lv.toArray().first!, 300, "split2: logvar[0] = X[C·plane]")
    }

    // randn: детерминированный, среднее≈0, σ≈1
    do {
        let n = 1 << 16
        session.begin()
        let g1 = session.randn(seed: 7, rows: 1, cols: n)
        let g2 = session.randn(seed: 7, rows: 1, cols: n)
        let g3 = session.randn(seed: 8, rows: 1, cols: n)
        _ = session.end()
        let v1 = g1.toArray(), v2 = g2.toArray(), v3 = g3.toArray()
        expectEqual(maxAbsDiff(v1, v2), 0, "randn: одинаковый seed → одинаковый шум")
        expect(maxAbsDiff(v1, v3) > 0.1, "randn: другой seed → другой шум")
        let mean = v1.reduce(0) { $0 + Double($1) } / Double(n)
        let varSum = v1.reduce(0) { $0 + pow(Double($1) - mean, 2) } / Double(n)
        expectClose(mean, 0.0, 0.02, "randn: среднее ≈ 0")
        expectClose(sqrt(varSum), 1.0, 0.02, "randn: σ ≈ 1")
        expect(v1.allSatisfy { $0.isFinite }, "randn: без NaN/Inf")
    }

    // KL: 0 при (mu=0, logvar=0); 0.5·n при (mu=1, logvar=0)
    do {
        let n = 10
        let mu0 = GPUTensor(engine: engine, data: [Float](repeating: 0, count: n), rows: 1)
        let lv0 = GPUTensor(engine: engine, data: [Float](repeating: 0, count: n), rows: 1)
        session.begin()
        let k0 = session.kl(mu: mu0, logvar: lv0)
        _ = session.end()
        expectClose(Double(k0.contents().assumingMemoryBound(to: Float.self)[0]), 0, 1e-5,
                    "KL(N(0,1)‖N(0,1)) = 0")

        let mu1 = GPUTensor(engine: engine, data: [Float](repeating: 1, count: n), rows: 1)
        session.begin()
        let k1 = session.kl(mu: mu1, logvar: lv0)
        _ = session.end()
        expectClose(Double(k1.contents().assumingMemoryBound(to: Float.self)[0]),
                    0.5 * Double(n), 1e-4, "KL: mu=1 → 0.5·n")
    }
}

// MARK: - 7c. Вариационный путь: прямой проход и сэмплирование

func testVAEVariational() {
    section("VAE: перепараметризация")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let plane = (w / 8) * (h / 8)

    let vae = ConvVAEGPU(engine: engine)
    let img = makeSyntheticScene(w: w, h: h, rng: &systemRNG)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    session.begin()
    let dc = vae.forward(session, image: x, w: w, h: h, deterministic: true)
    _ = session.end()

    expectEqual(dc.mu.rows, ConvVAEGPU.latentC, "mu: C каналов латента")
    expectEqual(dc.logvar.rows, ConvVAEGPU.latentC, "logvar: C каналов латента")
    expectEqual(dc.mu.cols, plane, "mu: (w/8)×(h/8)")
    expectEqual(dc.z.n, dc.mu.n, "z той же формы, что mu")
    expectEqual(dc.recon.rows, 3, "реконструкция: 3 канала")

    let mu = dc.mu.toArray(), lv = dc.logvar.toArray(), sig = dc.sigma.toArray()
    expect(mu.allSatisfy { $0.isFinite }, "mu без NaN")
    expect(lv.allSatisfy { $0.isFinite }, "logvar без NaN")
    // σ = exp(0.5·lv) — проверяем на всех элементах
    var sigErr = 0.0
    for i in 0..<lv.count {
        let expectS = exp(0.5 * Double(min(max(lv[i], -10), 10)))
        sigErr = max(sigErr, abs(Double(sig[i]) - expectS))
    }
    expect(sigErr < 1e-5, "σ = exp(0.5·logvar) — макс. ошибка \(sigErr)")

    // Детерминированный режим: z ровно mu (ε = 0).
    expectEqual(maxAbsDiff(dc.z.toArray(), mu), 0, "deterministic: z == mu")

    // Стохастический режим: z = mu + σ·ε, ε ≠ 0.
    session.begin()
    let sc = vae.forward(session, image: x, w: w, h: h, seed: 424242)
    _ = session.end()
    let zs = sc.z.toArray(), eps = sc.eps.toArray()
    expect(eps.contains { abs($0) > 1e-6 }, "стохастический режим: ε ≠ 0")
    var zErr = 0.0
    for i in 0..<zs.count {
        zErr = max(zErr, abs(Double(zs[i]) - (Double(sc.mu.toArray()[i]) + Double(sig[i]) * Double(eps[i]))))
    }
    expect(zErr < 1e-4, "z = mu + σ·ε — макс. ошибка \(zErr)")

    // Тот же seed → тот же z (воспроизводимость обучения).
    session.begin()
    let sc2 = vae.forward(session, image: x, w: w, h: h, seed: 424242)
    _ = session.end()
    expectEqual(maxAbsDiff(sc2.z.toArray(), zs), 0, "тот же seed → тот же z")

    // Другой seed → другой z.
    session.begin()
    let sc3 = vae.forward(session, image: x, w: w, h: h, seed: 424243)
    _ = session.end()
    expect(maxAbsDiff(sc3.z.toArray(), zs) > 1e-4, "другой seed → другой z")

    // Реконструкция в [0,1] (сигмоида) и без NaN в обоих режимах.
    let r = sc.recon.toArray()
    expect(r.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "реконструкция в [0,1] без NaN")

    // Латент должен быть УЗКИМ МЕСТОМ, а не копией кадра.
    // Сжатие = (3·hw) / (C·hw/64) = 192/C.
    let ratio = ConvVAEGPU.compressionRatio
    expect(ratio >= 8,
           String(format: "латент сжимает кадр в %.0f× (≥8× — иначе это не латент)", ratio))
    expectEqual(dc.mu.n * Int(ratio.rounded()), 3 * w * h,
                "сжатие согласовано с формой латента")
    // Проверяем на реальном кадре: 3·w·h значений RGB → mu.n значений латента.
    let actualRatio = Double(3 * w * h) / Double(dc.mu.n)
    expectClose(actualRatio, ratio, 0.001, "фактическое сжатие = 192/C")
}

// MARK: - 7d. Вариационный путь: градиенты против конечных разностей

func testVAEGradients() {
    section("VAE: градиенты vs конечные разности")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let beta: Float = 0.5

    let vae = ConvVAEGPU(engine: engine)
    let img = makeSyntheticScene(w: w, h: h, rng: &systemRNG)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    /// Полная потеря на CPU: MSE + β·KL (в точности та же формула, что в backward).
    ///
    /// `deterministic` здесь всегда false: шум ε задаётся только (seed, index),
    /// поэтому между вызовами он не меняется, а путь перепараметризации
    /// `dlv += dz·ε·0.5·σ` оказывается задействован — иначе он не проверяется.
    func loss() -> Double {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
        _ = session.end()
        let r = c.recon.toArray()
        let mu = c.mu.toArray(), lv = c.logvar.toArray()
        var mse = 0.0
        for i in 0..<r.count {
            let d = Double(r[i] - img[i])
            mse += d * d
        }
        mse /= Double(r.count)
        var kl = 0.0
        for i in 0..<mu.count {
            let l = Double(min(max(lv[i], -10), 10))
            kl += 0.5 * (Double(mu[i]) * Double(mu[i]) + exp(l) - 1 - l)
        }
        kl /= Double(mu.count)
        return mse + Double(beta) * kl
    }

    // Небольшая тренировка, чтобы веса ушли от инициализации и градиенты были значимы.
    for step in 1...20 {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
        let (_, g) = vae.backward(session, c, target: x, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: 0.002)
        _ = session.end()
    }

    // Аналитические градиенты (детерминированный режим — ε фиксирован).
    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
    let (_, grads) = vae.backward(session, cache, target: x, beta: beta)
    _ = session.end()

    // Градиенты должны быть НЕнулевыми, иначе сверка 0 == 0 проходит вхолостую.
    var maxGrad = 0.0
    var perLayer: [String] = []
    for (i, g) in grads.enumerated() {
        let a = g.toArray()
        let m = a.map { abs(Double($0)) }.max() ?? 0
        maxGrad = max(maxGrad, m)
        if i % 2 == 0 { perLayer.append(String(format: "L%d=%.2e", i / 2, m)) }
    }
    print("      |grad| по слоям: \(perLayer.joined(separator: " "))")
    expect(maxGrad > 1e-6,
           String(format: "градиенты невырождены (max |g| = %.2e)", maxGrad))

    // Активации энкодера не должны схлопнуться в ноль целиком.
    expect(cache.a1.toArray().contains { $0 > 1e-6 }, "a1 не вырождена")
    expect(cache.a2.toArray().contains { $0 > 1e-6 }, "a2 не вырождена")

    // Точки для сверки выбираем по наибольшему |градиенту|: у мёртвых ReLU-каналов
    // и аналитика, и конечная разность равны нулю, и такая проверка ничего не значит.
    let eps: Float = 1e-2
    var checked = 0
    for layerIdx in [2, 3] {   // e3 — перед латентом, d1 — сразу после
        let g = grads[layerIdx * 2].toArray()
        let top = (0..<g.count).sorted { abs(g[$0]) > abs(g[$1]) }.prefix(2)
        for weightIdx in top {
            let layer = vae.layers[layerIdx]
            let analytic = Double(g[weightIdx])
            expect(abs(analytic) > 1e-7,
                   String(format: "слой %d вес %d: градиент значим (%.2e)", layerIdx, weightIdx, analytic))

            let orig = layer.W.getValue(weightIdx)
            layer.W.setValue(weightIdx, orig + eps)
            let lp = loss()
            layer.W.setValue(weightIdx, orig - eps)
            let lm = loss()
            layer.W.setValue(weightIdx, orig)

            let numeric = (lp - lm) / (2 * Double(eps))
            let denom = max(1e-9, abs(analytic) + abs(numeric))
            let rel = abs(analytic - numeric) / denom
            expect(rel < 0.10,
                   String(format: "слой %d вес %d: аналит %.4e числ %.4e (отн %.1f%%)",
                          layerIdx, weightIdx, analytic, numeric, rel * 100))
            checked += 1
        }
    }
    expectEqual(checked, 4, "проверено 4 точки")
}

// MARK: - 7e. Обучение действительно снижает потерю

func testVAEConvergence() {
    section("VAE: обучение снижает потерю")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let beta: Float = 0.01

    let vae = ConvVAEGPU(engine: engine)
    let img = makeSyntheticScene(w: w, h: h, rng: &systemRNG)
    let x = GPUTensor(engine: engine, data: img, rows: 3)

    func loss() -> Double {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        _ = session.end()
        let r = c.recon.toArray()
        var s = 0.0
        for i in 0..<r.count {
            let d = Double(r[i] - img[i])
            s += d * d
        }
        return s / Double(r.count)
    }

    let before = loss()
    for step in 1...60 {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: UInt32(step))
        let (_, g) = vae.backward(session, c, target: x, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: 0.01)
        _ = session.end()
    }
    let after = loss()
    expect(after < before, String(format: "потеря упала: %.5f → %.5f", before, after))
    expect(after.isFinite, "потеря конечна после обучения")
}

var systemRNG = SystemRandomNumberGenerator()

// MARK: - 7f. Бюджет видеопамяти на пачку

func testVRAMBudget() {
    section("Бюджет VRAM: размер пачки")

    // Один кадр 854×480 требует ~547 МБ активаций (доминирует im2col слоя d3).
    let big = convVAEBytesPerSample(w: 856, h: 480)
    let bigMB = Double(big) / 1_048_576
    expect(bigMB > 450 && bigMB < 650,
           String(format: "854×480: %.0f МБ на кадр (ожидалось 450–650)", bigMB))

    // Для сравнения: сам кадр — всего 4.9 МБ. Старая формула видела только его.
    let frameMB = Double(856 * 480 * 3 * 4) / 1_048_576
    expect(frameMB < 6, String(format: "кадр целиком: %.1f МБ — в 100 раз меньше активаций", frameMB))

    // Квадратичный рост по числу пикселей.
    let half = convVAEBytesPerSample(w: 428, h: 240)
    let ratio = Double(big) / Double(max(half, 1))
    expect(ratio > 3.5 && ratio < 4.5,
           String(format: "память ∝ площади: 4× пикселей → %.2f× памяти", ratio))

    // 8 ГБ рабочего набора, половина под пачку → не 16 кадров, а единицы.
    let budget8GB = 8 * 1024 * 1024 * 1024 / 2
    let b = effectiveBatchSize(requested: 16, w: 856, h: 480, budgetBytes: budget8GB)
    expect(b >= 5 && b <= 9,
           "8 ГБ VRAM, 854×480: пачка \(b) (ожидалось 5–9, а не 16)")

    // Маленькое разрешение — пачка ограничена только запросом.
    let small = effectiveBatchSize(requested: 4, w: 64, h: 64,
                                   budgetBytes: 8 * 1024 * 1024 * 1024 / 2)
    expectEqual(small, 4, "64×64: пачка = запрошенная")

    // Нулевой бюджет → всё равно 1 (иначе обучение невозможно).
    expectEqual(effectiveBatchSize(requested: 8, w: 856, h: 480, budgetBytes: 0), 1,
                "нулевой бюджет → пачка 1")

    // Запрос меньше бюджета → запрос не увеличивается.
    expectEqual(effectiveBatchSize(requested: 1, w: 856, h: 480, budgetBytes: budget8GB), 1,
                "запрос 1 → пачка 1")

    // Пул кадров (60%) + активации (30%) не должны превышать рабочий набор.
    let ws: UInt64 = 8 * 1024 * 1024 * 1024
    let poolBytes = gpuFramePoolCapacity(framesTotal: 43187, frameBytes: 856 * 480 * 3 * 4,
                                         workingSet: ws) * 856 * 480 * 3 * 4
    let actBytes = activationBudgetBytes(workingSet: ws)
    expect(poolBytes + actBytes <= Int(ws),
           "пул + активации ≤ рабочий набор (\(poolBytes / 1_048_576) + \(actBytes / 1_048_576) МБ)")
    expectClose(activationBudgetFraction, 0.30, 1e-9, "доля активаций = 30%")
    // Пул ограничен ещё и сверху (preferred = 1024 кадра), поэтому доля ≤ 60%.
    let poolFraction = Double(poolBytes) / Double(ws)
    expect(poolFraction <= 0.60 + 1e-6,
           String(format: "доля пула кадров ≤ 60%% (%.1f%%)", poolFraction * 100))
    expect(poolFraction > 0.5,
           String(format: "пул загружает больше половины VRAM (%.1f%%)", poolFraction * 100))

    // Сообщение при OOM должно называть объём и подсказывать, что делать.
    let msg = allocationFailureMessage(bytes: 512 * 1_048_576, availableBytes: 100 * 1_048_576,
                                       what: "тензор 16×410880")
    expect(msg.contains("512 МБ"), "OOM-сообщение: нужный объём — \(msg)")
    expect(msg.contains("100 МБ"), "OOM-сообщение: свободный объём")
    expect(msg.contains("тензор 16×410880"), "OOM-сообщение: что именно не влезло")
    expect(msg.contains("пачки"), "OOM-сообщение: подсказка про размер пачки")
}

// MARK: - 7g. Отметки «кадр обработан»

func testProcessedMarks() {
    section("Отметки обработанных кадров")

    // Пустой массив выравнивается по числу кадров.
    var p: [Bool] = []
    let added = markProcessedFrames(names: ["a", "b", "c"],
                                    reconNames: ["a", "c"], processed: &p)
    expectEqual(added, 2, "отмечено 2 из 3")
    expectEqual(p, [true, false, true], "метки по именам")

    // Повторный вызов не добавляет новых отметок.
    let again = markProcessedFrames(names: ["a", "b", "c"],
                                    reconNames: ["a", "c"], processed: &p)
    expectEqual(again, 0, "повторно — 0 новых")
    expectEqual(p, [true, false, true], "метки не сброшены")

    // Догрузка: появились новые кадры — старые метки сохраняются.
    let added2 = markProcessedFrames(names: ["a", "b", "c", "d"],
                                     reconNames: ["d"], processed: &p)
    expectEqual(added2, 1, "новый кадр отмечен")
    expectEqual(p, [true, false, true, true], "старые метки на месте")

    // Реконструкций нет — ничего не отмечаем.
    var q: [Bool] = []
    expectEqual(markProcessedFrames(names: ["x"], reconNames: [], processed: &q), 0,
                "нет реконструкций → 0")
    expectEqual(q, [false], "флаг остался false")

    // Список кадров короче — лишние метки отбрасываются (префикс сохраняется).
    var r: [Bool] = [true, true, true, true]
    markProcessedFrames(names: ["a", "b"], reconNames: ["a"], processed: &r)
    expectEqual(r, [true, true], "длина приведена к числу кадров, префикс сохранён")
}

// MARK: - 7h. Воспроизводимость

func testReproducibility() {
    section("Воспроизводимость")

    // Синтетическая сцена должна быть одинаковой между вызовами: иначе
    // convcheck/convprofile нельзя сравнивать между прогонами.
    let a = makeNormalSynthetic(w: 32, h: 24)
    let b = makeNormalSynthetic(w: 32, h: 24)
    expectEqual(maxAbsDiff(a, b), 0, "makeNormalSynthetic детерминирован")
    expectEqual(a.count, 32 * 24 * 3, "размер сцены = w×h×3")

    // Явный RNG всё ещё даёт разные сцены — вариативность не потеряна.
    var r1 = SplitMix64(seed: 1)
    var r2 = SplitMix64(seed: 2)
    let s1 = makeSyntheticScene(w: 32, h: 24, rng: &r1)
    let s2 = makeSyntheticScene(w: 32, h: 24, rng: &r2)
    expect(maxAbsDiff(s1, s2) > 1e-6, "разные seed → разные сцены")

    // SplitMix64 воспроизводим и не выдаёт константу.
    var g1 = SplitMix64(seed: 42), g2 = SplitMix64(seed: 42)
    let seq1 = (0..<8).map { _ in g1.next() }
    let seq2 = (0..<8).map { _ in g2.next() }
    expectEqual(seq1, seq2, "SplitMix64: одинаковый seed → одинаковая последовательность")
    expect(Set(seq1).count == 8, "SplitMix64 не залипает на одном значении")

    // Две модели с одним seed инициализируются одинаково.
    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let v1 = ConvVAEGPU(engine: engine, seed: 777)
    let v2 = ConvVAEGPU(engine: engine, seed: 777)
    let v3 = ConvVAEGPU(engine: engine, seed: 778)
    expectEqual(maxAbsDiff(v1.e1.W.toArray(), v2.e1.W.toArray()), 0,
                "одинаковый seed → одинаковые веса")
    expect(maxAbsDiff(v1.e1.W.toArray(), v3.e1.W.toArray()) > 1e-9,
           "разный seed → разные веса")
}

// MARK: - 7i. Совместимость чекпоинта с архитектурой

func testCheckpointCompat() {
    section("Чекпоинт: совместимость архитектуры")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let vae = ConvVAEGPU(engine: engine)
    let C = ConvVAEGPU.latentC

    // Корректный набор: W0,b0..W5,b5 в формах текущей модели.
    func correctMats() -> [Matrix] {
        vae.layers.flatMap { l in
            [Matrix(rows: l.oc, cols: l.ic * l.k * l.k, fill: 0.01),
             Matrix(rows: l.oc, cols: 1, fill: 0)]
        }
    }

    do {
        try vae.load(from: correctMats())
        expect(true, "корректные формы загружаются")
    } catch {
        expect(false, "корректные формы загружаются — \(error.localizedDescription)")
    }

    // Слишком мало матриц → код 10.
    do {
        try vae.load(from: Array(correctMats().prefix(6)))
        expect(false, "мало матриц → ошибка")
    } catch {
        expect(isIncompatibleCheckpoint(error), "мало матриц → распознано как несовместимость")
        expectEqual((error as NSError).code, 10, "мало матриц → код 10")
    }

    // Чекпоинт от прежней архитектуры: e3 был 64 канала вместо 2·C, d1 — 64 на входе.
    do {
        var old = correctMats()
        old[4] = Matrix(rows: 64, cols: 32 * 9, fill: 0.01)   // e3.W: старые 64 канала
        old[6] = Matrix(rows: 32, cols: 64 * 9, fill: 0.01)   // d1.W: старые 64 на входе
        try vae.load(from: old)
        expect(false, "старая архитектура → ошибка")
    } catch {
        expect(isIncompatibleCheckpoint(error), "старая архитектура → распознано как несовместимость")
        expectEqual((error as NSError).code, 11, "старая архитектура → код 11")
    }

    // Чужая ошибка (не от load) не должна считаться несовместимостью.
    let ioError = NSError(domain: NSCocoaErrorDomain, code: 260,
                          userInfo: [NSLocalizedDescriptionKey: "нет файла"])
    expect(!isIncompatibleCheckpoint(ioError), "ошибка ввода-вывода — не несовместимость")

    // Смещение формы: верное число матриц, но неверная ширина латента у d2.
    do {
        var broken = correctMats()
        broken[9] = Matrix(rows: 16, cols: 32 * 9 + 1, fill: 0)
        try vae.load(from: broken)
        expect(false, "битая форма d2 → ошибка")
    } catch {
        expect(isIncompatibleCheckpoint(error), "битая форма d2 → распознано")
    }
}

// MARK: - 7j. Журнал обучения не спамит нулём

func testTrainStepLogging() {
    section("Журнал обучения: без спама нулём")

    // Служебный отчёт (подготовка / догрузка VRAM) — done = 0. Не логируется,
    // иначе на каждую догрузку окна приходит 64 одинаковые строки «шаг 0».
    expect(!shouldLogTrainStep(done: 0, total: 129561), "done=0 — служебный отчёт, не пишем")
    expect(!shouldLogTrainStep(done: 0, total: 0), "done=0 при total=0 — не пишем")

    // Настоящие шаги: каждый 200-й.
    expect(shouldLogTrainStep(done: 200, total: 129561), "шаг 200 — пишем")
    expect(!shouldLogTrainStep(done: 201, total: 129561), "шаг 201 — не пишем")
    expect(shouldLogTrainStep(done: 2000, total: 129561), "шаг 2000 — пишем")
    expect(!shouldLogTrainStep(done: 1, total: 129561), "шаг 1 — не пишем")

    // Последний шаг пишется всегда, даже если не кратен 200.
    expect(shouldLogTrainStep(done: 129561, total: 129561), "последний шаг — пишем")
    expect(!shouldLogTrainStep(done: 129560, total: 129561), "предпоследний — не пишем")

    // Нулевой интервал не должен приводить к делению на ноль.
    expect(shouldLogTrainStep(done: 7, total: 100, every: 0), "every=0 → не падаем")

    // Сколько строк даст догрузка окна: 1024 кадра / 16 = 64 отчёта, все done=0.
    let spam = (0..<64).filter { _ in shouldLogTrainStep(done: 0, total: 129561) }.count
    expectEqual(spam, 0, "догрузка окна VRAM больше не пишет ни строки")

    // А за 1000 шагов должно появиться ровно 5 строк (200,400,600,800,1000).
    let lines = (1...1000).filter { shouldLogTrainStep(done: $0, total: 129561) }.count
    expectEqual(lines, 5, "за 1000 шагов — 5 строк прогресса")
}

// MARK: - 7b. Термо-регулятор GPU

func testThermalGovernor() {
    section("Термо-регулятор GPU")

    // --- Скважность: сколько спать после шага ---
    // GPU занят `gpuMs`, значит доля занятости = gpu/(gpu+sleep).
    // Чтобы уложиться в duty: sleep = gpu·(1/duty − 1).
    expectClose(dutySleepMs(gpuMs: 170, duty: 0.55, maxPauseMs: 2000), 170 * (1 / 0.55 - 1),
                0.01, "170 мс при duty 55% → пауза 139 мс")
    expectClose(dutySleepMs(gpuMs: 170, duty: 0.5, maxPauseMs: 2000), 170,
                0.01, "duty 50% → пауза равна времени работы")

    // Главная регрессия: пауза не должна быть нулевой на штатном режиме.
    // Раньше в цикле шага стоял жёсткий Thread.sleep(0.03) — 30 мс на кадр
    // 854×480 при 170 мс работы, то есть скважность ~85% на много часов.
    expect(dutySleepMs(gpuMs: 170, duty: 0.55, maxPauseMs: 2000) > 100,
           "на штатном режиме пауза больше 100 мс, а не 30")

    // Вырожденные входы не должны давать NaN/отрицательных пауз.
    expectEqual(dutySleepMs(gpuMs: 0, duty: 0.55, maxPauseMs: 2000), 0, "нулевая работа → пауза 0")
    expectEqual(dutySleepMs(gpuMs: -5, duty: 0.55, maxPauseMs: 2000), 0, "отрицательное время → 0")
    expectEqual(dutySleepMs(gpuMs: .nan, duty: 0.55, maxPauseMs: 2000), 0, "NaN → 0")
    expectEqual(dutySleepMs(gpuMs: 170, duty: 0, maxPauseMs: 2000), 0, "duty 0 → не делим на ноль")
    expectEqual(dutySleepMs(gpuMs: 170, duty: 1, maxPauseMs: 2000), 0, "duty 100% → пауза 0")
    expectEqual(dutySleepMs(gpuMs: 170, duty: 1.5, maxPauseMs: 2000), 0, "duty >1 → 0")

    // Пауза ограничена сверху, иначе один шаг уснёт навсегда.
    expectEqual(dutySleepMs(gpuMs: 10_000, duty: 0.5, maxPauseMs: 2000), 2000,
                "пауза ограничена maxPauseMs")

    // 99% — это действительно 99%, а не «почти 99%»: на шаге 170 мс пауза
    // выходит 1.7 мс, то есть ~1% времени. Проверка ловит случай, когда
    // потолок скважности урезали, а расчёт паузы остался прежним.
    let sleepAt99 = dutySleepMs(gpuMs: 170, duty: 0.99, maxPauseMs: 5000)
    expectClose(sleepAt99, 170 * (1 / 0.99 - 1), 1e-9, "99% → пауза по формуле")
    expect(sleepAt99 < 2.5, "99% → пауза меньше 2.5 мс на шаге 170 мс")

    // --- Целевая скважность по уровню нагрева ---
    let dn = targetDuty(for: .nominal)
    let dw = targetDuty(for: .warm)
    let dh = targetDuty(for: .hot)
    let dc = targetDuty(for: .critical)
    expect(dn < 1 && dn > 0.2, "штатная скважность в разумных пределах")
    expect(dn > dw && dw > dh && dh > dc, "чем жарче, тем ниже скважность")
    expect(dc > 0, "критический уровень всё же не 0 — обучение не встаёт навсегда")

    // --- Уровень нагрева ---
    // Без базовой линии дрейф измерить не из чего.
    expectEqual(heatLevel(stepMs: 500, baselineMs: 0, thermalRaw: 0), .nominal,
                "нет базовой линии → nominal")
    // Штатный разброс не должен поднимать тревогу.
    expectEqual(heatLevel(stepMs: 200, baselineMs: 170, thermalRaw: 0), .nominal,
                "дрейф 1.18× — это шум")
    expectEqual(heatLevel(stepMs: 220, baselineMs: 170, thermalRaw: 0), .warm,
                "дрейф 1.29× — warm")
    // Ровно тот случай из лога: 0.8 с против 0.2 с.
    expectEqual(heatLevel(stepMs: 800, baselineMs: 200, thermalRaw: 0), .critical,
                "дрейф 4× (как в логе 0.2→0.8 с) — critical")

    // Показания ОС перебивают дрейф.
    expectEqual(heatLevel(stepMs: 200, baselineMs: 200, thermalRaw: 2), .hot,
                "thermalState .serious → hot")
    expectEqual(heatLevel(stepMs: 200, baselineMs: 200, thermalRaw: 3), .critical,
                "thermalState .critical → critical")
    // Но и при спокойной ОС четырёхкратный дрейф уже тревога.
    expectEqual(heatLevel(stepMs: 200, baselineMs: 200, thermalRaw: 0), .nominal,
                "нет дрейфа и нет жара → nominal")

    // --- Дрейф без температуры — это не нагрев ---
    // Ровно случай из журнала 4 ноября: шаг 366 мс против базы 139 мс
    // (2.6×) при 64 °C. Раньше это давало «критично» на холодной карте.
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0, tempC: 64), .warm,
                "2.6× дрейф при 64 °C — это сброс по мощности, а не нагрев")
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0, tempC: 70), .warm,
                "при 70 °C — тоже не выше warm")
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0, tempC: 83), .hot,
                "при 83 °C дрейф уже можно считать нагревом, но не критичным")
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0, tempC: 86), .critical,
                "у потолка — критично, без скидок")
    expectEqual(heatLevel(stepMs: 366, baselineMs: 139, thermalRaw: 0), .critical,
                "без датчика остаётся прежнее поведение по дрейфу")

    // --- Гаситель дребезга ---
    // 328/366 мс на пороге 2.5× переключали уровень каждый шаг.
    var deb = HeatDebouncer(steps: 3)
    var seen: [HeatLevel] = []
    for i in 0..<10 {
        seen.append(deb.update(i % 2 == 0 ? .hot : .critical))
    }
    expect(seen.allSatisfy { $0 == .nominal },
           "чередование горячо/критично не меняет уровень (дребезг погашен)")

    // Устойчивый переход принимается ровно после `steps` замеров.
    var deb2 = HeatDebouncer(steps: 3)
    expectEqual(deb2.update(.hot), .nominal, "1-й замер — ещё не переключились")
    expectEqual(deb2.update(.hot), .nominal, "2-й замер — ещё не переключились")
    expectEqual(deb2.update(.hot), .hot, "3-й замер — переключились")
    expectEqual(deb2.update(.hot), .hot, "дальше держится")
    // Одиночный выброс вниз не сбрасывает уровень.
    expectEqual(deb2.update(.nominal), .hot, "одиночный сброс проигнорирован")
    expectEqual(deb2.update(.nominal), .hot, "два сброса подряд — ещё держим")
    expectEqual(deb2.update(.nominal), .nominal, "три подряд — снизились")

    // --- Базовая линия шага ---
    expectEqual(updateStepBaseline(0, sample: 170), 170, "первый замер задаёт базу")
    expectEqual(updateStepBaseline(170, sample: 200), 170, "база не растёт от медленного шага")
    expectEqual(updateStepBaseline(170, sample: 100), 100, "база опускается к быстрому шагу")
    expectEqual(updateStepBaseline(170, sample: Double.nan), 170, "NaN не портит базу")
    expectEqual(updateStepBaseline(170, sample: 0), 170, "ноль не портит базу")

    // --- Решение на шаг (это вызывает цикл обучения) ---
    // Первый шаг: базы нет, ОС спокойна, пользовательская пауза 250 мс.
    let d1 = throttleDecision(gpuMs: 170, wallMs: 170, baselineMs: 0, thermalRaw: 0,
                              floorMs: 250, maxPauseMs: 2000)
    expectEqual(d1.level, .nominal, "первый шаг — норма")
    expectEqual(d1.sleepMs, 250, "пауза не меньше пользовательской")
    expectClose(d1.baselineMs, 170, 1e-9, "база запомнила первый замер")

    // Без пользовательской паузы работает чистая скважность.
    let d2 = throttleDecision(gpuMs: 170, wallMs: 170, baselineMs: 0, thermalRaw: 0,
                              floorMs: 0, maxPauseMs: 2000)
    expectClose(d2.sleepMs, 139.09, 0.01, "пауза по скважности 55%")

    // Тот же кадр, но GPU разогрелась: тепловой уровень режет скважность.
    let d3 = throttleDecision(gpuMs: 170, wallMs: 170, baselineMs: 170, thermalRaw: 3,
                              floorMs: 0, maxPauseMs: 2000)
    expectEqual(d3.level, .critical, "critical по показаниям ОС")
    expectClose(d3.sleepMs, 170 * (1 / 0.12 - 1), 0.5, "на critical пауза в 7 раз больше работы")

    // Дрейф из лога: 0.8 с против базы 0.2 с — регулятор обязан среагировать
    // без всяких показаний ОС.
    let d4 = throttleDecision(gpuMs: 700, wallMs: 800, baselineMs: 200, thermalRaw: 0,
                              floorMs: 250, maxPauseMs: 2000)
    expectEqual(d4.level, .critical, "дрейф 4× → critical без показаний ОС")
    expect(d4.sleepMs > 1000, "на дрейфе пауза больше секунды")

    // Потолок соблюдается всегда, даже на критическом уровне.
    let d5 = throttleDecision(gpuMs: 5000, wallMs: 5000, baselineMs: 100, thermalRaw: 3,
                              floorMs: 250, maxPauseMs: 2000)
    expectEqual(d5.sleepMs, 2000, "пауза не превышает maxPauseMs")

    // База не портится медленным шагом: иначе дрейф перестанет быть виден.
    let d6 = throttleDecision(gpuMs: 170, wallMs: 900, baselineMs: 170, thermalRaw: 0,
                              floorMs: 0, maxPauseMs: 2000)
    expectClose(d6.baselineMs, 170, 1e-9, "медленный шаг не поднимает базу")
    expect(d6.level >= .hot, "медленный шаг распознан как нагрев")
}

// MARK: - 7c. Замкнутый контур по температуре GPU

func testPowerController() {
    section("Контур по температуре GPU")

    // Потолок из «мощности» в процентах.
    expectClose(dutyCeiling(powerPercent: 100), 0.98, 1e-9, "100% → 0.98")
    expectClose(dutyCeiling(powerPercent: 85), 0.833, 1e-9, "85% → 0.833")
    expectClose(dutyCeiling(powerPercent: 50), 0.49, 1e-9, "50% → 0.49")
    expectClose(dutyCeiling(powerPercent: 0), 0.10, 1e-9, "0% → нижний предел 0.10")
    expectClose(dutyCeiling(powerPercent: 200), 0.98, 1e-9, "больше 100% → 0.98")
    // 100% — это не «без паузы вообще»: 2% простоя оставлены WindowServer.
    expect(maxSafeDuty < 1.0, "даже 100% оставляет слоты WindowServer")
    expectClose(maxSafeDuty, 0.98, 1e-9, "100% мощности → 98% скважности")

    // Без телеметрии режим не меняем: гадать хуже, чем держать текущий.
    var none = PowerController(ceiling: 0.765, startDuty: 0.5)
    expectClose(none.update(stats: nil), 0.5, 1e-9, "нет датчика — скважность та же")

    // Перегрев: вниз, но не больше 5% за замер — иначе нагрузка «дёргается».
    var hot = PowerController(ceiling: 0.9, startDuty: 0.6)
    let h1 = hot.update(stats: GpuStats(temperatureC: 90, coreClockMHz: 900,
                                        activityPercent: 95, powerW: 45))
    expectClose(h1, 0.55, 1e-9, "90 °C → минус 5% за замер")

    // Холодно: поднимается до потолка и не выше.
    var cold = PowerController(ceiling: 0.765, startDuty: 0.5)
    var d = 0.5
    for _ in 0..<40 {
        d = cold.update(stats: GpuStats(temperatureC: 60, coreClockMHz: 1400,
                                        activityPercent: 95, powerW: 40))
    }
    expectClose(d, 0.765, 1e-9, "на холоде упирается в потолок, а не в 1.0")

    // Замкнутый контур с простой тепловой моделью: должен сойтись и не
    // раскачаться. Это главная проверка — именно раскачка и перегревала.
    var loop = PowerController(ceiling: 0.9, startDuty: 0.9)
    var temp = 70.0
    var duties: [Double] = []
    for _ in 0..<200 {
        temp += (65.0 + 25.0 * loop.duty - temp) * 0.15
        duties.append(loop.update(stats: GpuStats(temperatureC: temp, coreClockMHz: 1200,
                                                  activityPercent: 95, powerW: 40)))
    }
    let settled = duties.suffix(20).reduce(0, +) / 20
    let earlier = duties[duties.count - 60]
    expect(abs(settled - earlier) < 0.06, "контур сходится, а не раскачивается")
    expect(temp <= 85.5, "температура не выходит за потолок 85 °C")
    expect(duties.min()! > 0.05, "нагрузка не проваливается в ноль")

    // Детектор троттлинга по частоте сознательно убран: замер показал, что
    // Core Clock в PerformanceStatistics скачет 0…468 МГц на реальной
    // нагрузке (срезы попадают в промежутки между command buffer'ами).
    // Решение по такому сигналу — реакция на шум, поэтому проверяем,
    // что шум частоты скважность НЕ меняет.
    var noise = PowerController(ceiling: 0.9, startDuty: 0.8)
    _ = noise.update(stats: GpuStats(temperatureC: 65, coreClockMHz: 1400,
                                     activityPercent: 95, powerW: 40))
    let noisyDuty = noise.update(stats: GpuStats(temperatureC: 65, coreClockMHz: 40,
                                                 activityPercent: 2, powerW: 5))
    expect(noisyDuty > 0.8, "падение частоты в срезе не считается троттлингом")
    expectClose(noise.peakClockMHz, 1400, 1e-9, "пик частоты — только для журнала")

    // --- Разбор словаря телеметрии ---
    let raw: [String: Any] = ["Temperature(C)": 67, "Core Clock(MHz)": 268,
                              "GPU Activity(%)": 18, "Total Power(W)": 17]
    let st = parseGpuStats(raw)
    expect(st != nil, "словарь разобран")
    expectEqual(Int(st?.temperatureC ?? 0), 67, "температура прочитана")
    expectEqual(Int(st?.coreClockMHz ?? 0), 268, "частота прочитана")
    expectEqual(Int(st?.powerW ?? 0), 17, "мощность прочитана")
    expect(st?.hasTemperature == true, "температура осмысленна")

    // Без температуры запись бесполезна (узел без телеметрии).
    expect(parseGpuStats(["Core Clock(MHz)": 300]) == nil, "без температуры → nil")
    expect(parseGpuStats(["Temperature(C)": 0]) == nil, "нулевая температура → nil")
    // Значения могут прийти строками — разбор должен выдержать.
    expectEqual(Int(parseGpuStats(["Temperature(C)": "71"])?.temperatureC ?? 0), 71,
                "температура строкой разобрана")

    // Пороги откалиброваны замерами `yume gpuheat`: при 74 °C на краю
    // кристалла шаг уже замедлялся, значит запас кончился. 93 °C — точка, на
    // которой машина уходит в перезагрузку, поэтому 85 °C — жёсткий потолок,
    // а 82 °C — рабочая точка с тремя градусами на реакцию регулятора.
    let defaults = PowerController(ceiling: 0.765)
    expectClose(defaults.targetTempC, 82, 1e-9, "рабочая точка 82 °C")
    expectClose(defaults.maxTempC, 85, 1e-9, "потолок 85 °C — выше уходить нельзя")
    expectClose(emergencyTempC, 86, 1e-9, "аварийный порог 86 °C")
    expect(defaults.targetTempC < defaults.maxTempC, "рабочая точка ниже потолка")
    expect(emergencyTempC > defaults.maxTempC, "аварийный порог выше потолка")

    // Дрейф считается по времени GPU, поэтому собственная пауза не может
    // выглядеть как нагрев. Иначе: больше пауза → больше дрейф → ещё пауза.
    var base = 0.0
    var steady = HeatLevel.nominal
    for _ in 0..<30 {
        let d = throttleDecision(gpuMs: 140, wallMs: 140, baselineMs: base, thermalRaw: 0,
                                 floorMs: 250, maxPauseMs: 2000)
        base = d.baselineMs
        steady = d.level
    }
    expectEqual(steady, .nominal, "стабильное время GPU не даёт ложного нагрева")

    // А настоящий рост времени GPU — даёт.
    var b2 = 0.0
    var risen: HeatLevel = .nominal
    for i in 0..<10 {
        let ms: Double = i < 5 ? 140 : 260
        let d = throttleDecision(gpuMs: ms, wallMs: ms, baselineMs: b2, thermalRaw: 0,
                                 floorMs: 0, maxPauseMs: 2000)
        b2 = d.baselineMs
        risen = d.level
    }
    expect(risen >= .warm, "рост времени GPU с 140 до 260 мс распознан как нагрев")

    // --- Пауза между эпохами: только по теплу ---
    // Пауза — тепловой инструмент, а не часть расписания. Холодная карта
    // не должна «остывать»: замер на 99% показал, что она и так не греется.
    expectEqual(epochCoolPauseMs(tempC: 60, pauseMs: 250), 0,
                "60 °C → паузы между эпохами нет")
    expectEqual(epochCoolPauseMs(tempC: 81, pauseMs: 250), 0,
                "ниже рабочей точки → паузы нет")
    expectClose(epochCoolPauseMs(tempC: 82, pauseMs: 250), 500, 1e-9,
                "рабочая точка → половина паузы")
    expectClose(epochCoolPauseMs(tempC: 84, pauseMs: 250), 500, 1e-9,
                "между точкой и потолком → половина паузы")
    expectClose(epochCoolPauseMs(tempC: 85, pauseMs: 250), 1000, 1e-9,
                "у потолка → полная пауза, но не меньше 1000 мс")
    expectClose(epochCoolPauseMs(tempC: 88, pauseMs: 3000), 3000, 1e-9,
                "своя большая пауза уважается")
    // Нет датчика — не гадаем и не стоим: пошаговый регулятор всё равно
    // держит нагрузку, а эта пауза случается раз в эпоху.
    expectEqual(epochCoolPauseMs(tempC: nil, pauseMs: 250), 0, "нет датчика → 0")
    expectEqual(epochCoolPauseMs(tempC: .nan, pauseMs: 250), 0, "NaN → 0")
    expectEqual(epochCoolPauseMs(tempC: 0, pauseMs: 250), 0, "нулевая температура → 0")
}

// MARK: - 8. Чистые функции UI/логики (регрессия)

func testPureLogic() {
    section("Чистая логика")

    // Пагинатор.
    expectEqual(paginatorVisiblePages(current: 0, pages: 5, limit: 3), [0, 1, 2],
                "страницы: начало")
    expectEqual(paginatorVisiblePages(current: 4, pages: 5, limit: 3), [2, 3, 4],
                "страницы: конец")
    expectEqual(paginatorVisiblePages(current: 2, pages: 5, limit: 3), [1, 2, 3],
                "страницы: середина")
    expectEqual(paginatorVisiblePages(current: 0, pages: 0, limit: 3), [],
                "страницы: пусто")
    expectEqual(paginatorDotWidth(pageNumber: 9), 28, "ширина кружка: 1 цифра")
    expectEqual(paginatorDotWidth(pageNumber: 100), 34, "ширина кружка: 3 цифры")
    expectEqual(paginatorDotWidth(pageNumber: 1000), 40, "ширина кружка: 4 цифры")
    expectEqual(dotCapacity(columnWidth: 446, dotWidth: 28), 9, "вместимость 446/28")
    expectEqual(paginatorRowWidth(columnWidth: 494), 446, "ширина строки пагинатора")

    // Батчи.
    expectEqual(batchRanges(total: 10, batch: 4), [0..<4, 4..<8, 8..<10], "батчи 4/4/2")
    expectEqual(batchRanges(total: 0, batch: 4), [], "батчи: пусто")
    expectEqual(batchRanges(total: 3, batch: 8), [0..<3], "батчи: один неполный")

    // Контрольная выборка.
    expect(isControlFrame(index: 0, fps: 24), "контрольный: кадр 0")
    expect(isControlFrame(index: 24, fps: 24), "контрольный: кадр 24")
    expect(!isControlFrame(index: 23, fps: 24), "не контрольный: кадр 23")
    expect(isControlFrame(index: 0, fps: 0), "контрольный: fps=0 безопасен")

    // Прогресс.
    expectEqual(progressInFrames(stepDone: 10, totalSteps: 100, batch: 8).done, 80,
                "прогресс в кадрах: done")
    expectEqual(progressInFrames(stepDone: 10, totalSteps: 100, batch: 8).total, 800,
                "прогресс в кадрах: total")
    expectEqual(progressInFrames(stepDone: 10, totalSteps: 100, batch: 0).done, 10,
                "прогресс: батч 0 → 1")

    // Логирование прогресса.
    expect(shouldLogProgress(done: 0, total: 100, lastLoggedPercent: 0), "лог: старт")
    expect(shouldLogProgress(done: 100, total: 100, lastLoggedPercent: 95), "лог: финиш")
    expect(shouldLogProgress(done: 5, total: 100, lastLoggedPercent: 0), "лог: 5%")
    expect(!shouldLogProgress(done: 4, total: 100, lastLoggedPercent: 0), "лог: 4% не пишем")

    // Автотюнинг.
    let tuned = autoTuneValues(frames: 1000, batch: 4, resume: false,
                               currentSteps: 20, currentEpochs: 20, currentLr: 0.001,
                               stepsEditedByUser: false, epochsEditedByUser: false,
                               lrEditedByUser: false)
    expectEqual(tuned.steps, 250, "автотюнинг: шаги = кадры/батч")
    expectEqual(tuned.epochs, 3, "автотюнинг: эпохи = 3 при 1000 кадрах")

    // Живой статус.
    expectEqual(formatThousands(43187), "43 187", "разряды через пробел")
    expect(trainLivePhase(secondsSinceStep: 5, stepSeconds: 0.5) == .running,
           "статус: идёт")
    expect(trainLivePhase(secondsSinceStep: 95, stepSeconds: 0.5) == .stalled(seconds: 95),
           "статус: зависло")

    // Батч и фильтр.
    expectEqual(frameOffsetForSlot(0), 9, "коллаж: слот 0 → смещение 9")
    expectEqual(frameOffsetForSlot(5), 0, "коллаж: слот 5 → смещение 0")
}

// MARK: - Рендер слоёв (ground-truth разметка)

func testLayerRenderer() {
    section("Рендер слоёв: процедурные сцены с разметкой")

    // Детерминизм: один seed → побитово одна сцена.
    do {
        let a = makeLayeredScene(w: 96, h: 64, seed: 42)
        let b = makeLayeredScene(w: 96, h: 64, seed: 42)
        expectEqual(a.background, b.background, "детерминизм: фон")
        expectEqual(a.objects, b.objects, "детерминизм: объекты")
        expectEqual(a.mask, b.mask, "детерминизм: маска")
        expectEqual(a.light, b.light, "детерминизм: свет")
        expectEqual(a.composite, b.composite, "детерминизм: сборка")
    }
    // Разные seed → разные сцены.
    do {
        let a = makeLayeredScene(w: 96, h: 64, seed: 1)
        let b = makeLayeredScene(w: 96, h: 64, seed: 2)
        expect(a.background != b.background, "разные seed → разные фоны")
        expect(a.mask != b.mask, "разные seed → разные маски")
    }

    let scene = makeLayeredScene(w: 128, h: 96, seed: 2026)
    let n = scene.w * scene.h

    // Диапазоны: всё конечно и в [0,1].
    expect(scene.background.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "фон: значения конечны и в [0,1]")
    expect(scene.objects.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "объекты: значения конечны и в [0,1]")
    expect(scene.light.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "свет: значения конечны и в [0,1]")
    expect(scene.composite.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "сборка: значения конечны и в [0,1]")

    // Маска бинарна и непуста с обеих сторон.
    expect(scene.mask.allSatisfy { $0 == 0 || $0 == 1 }, "маска бинарна {0,1}")
    expect(scene.mask.contains(where: { $0 == 1 }), "маска: объекты есть")
    expect(scene.mask.contains(where: { $0 == 0 }), "маска: фон есть")
    let objPixels = scene.mask.filter { $0 == 1 }.count
    expect(objPixels > 0 && objPixels < n, "маска покрывает часть, а не весь кадр")

    // Композиция восстанавливается из слоёв ТОЧНО (формула декодера).
    let rebuilt = composeLayers(w: scene.w, h: scene.h, background: scene.background,
                                objects: scene.objects, mask: scene.mask, light: scene.light)
    expectEqual(rebuilt, scene.composite, "composeLayers(слои) == кадр (побитово)")

    // Формула сборки вручную на подмножестве пикселей.
    do {
        var ok = true
        for i in stride(from: 0, to: n, by: 97) {
            let m = scene.mask[i], l = scene.light[i]
            for c in 0..<3 {
                let v = (scene.background[i * 3 + c] * (1 - m) + scene.objects[i * 3 + c] * m) * l
                let expectV = min(1, max(0, v))
                if abs(scene.composite[i * 3 + c] - expectV) > 1e-6 { ok = false }
            }
        }
        expect(ok, "rgb = (фон·(1−m)+obj·m)·свет — формула выполняется попиксельно")
    }
    // Вне маски объекты нулевые, кадр = фон·свет.
    do {
        var okBg = true, okZero = true
        for i in 0..<n where scene.mask[i] == 0 {
            if scene.objects[i * 3] != 0 || scene.objects[i * 3 + 1] != 0 || scene.objects[i * 3 + 2] != 0 {
                okZero = false
            }
            for c in 0..<3 {
                if abs(scene.composite[i * 3 + c] - min(1, scene.background[i * 3 + c] * scene.light[i])) > 1e-6 {
                    okBg = false
                }
            }
        }
        expect(okZero, "вне маски объекты = 0")
        expect(okBg, "вне маски кадр = фон·свет")
    }

    // Свет: направленный (у солнца ярче) + виньетка → есть перепад.
    do {
        var mn = Float.greatestFiniteMagnitude, mx = -Float.greatestFiniteMagnitude
        for v in scene.light { mn = min(mn, v); mx = max(mx, v) }
        expect(mx - mn > 0.1, "свет: перепад > 0.1 (есть направление)")
        expect(mn >= 0.05, "свет: нет чёрных дыр (min ≥ 0.05)")
    }

    // CHW-раскладка: 8 каналов, порядок bgR..bgB objR..objB mask light.
    do {
        let ch = scene.channelsCHW()
        expectEqual(ch.count, 8 * n, "CHW: размер 8·H·W")
        var ok = true
        for i in 0..<n {
            if ch[6 * n + i] != scene.mask[i] { ok = false }
            if abs(ch[7 * n + i] - scene.light[i]) > 0 { ok = false }
            if abs(ch[0 * n + i] - scene.background[i * 3]) > 0 { ok = false }
            if abs(ch[3 * n + i] - scene.objects[i * 3]) > 0 { ok = false }
        }
        expect(ok, "CHW: порядок каналов фон|объекты|маска|свет")
    }

    // 480p smoke: рендерится целиком, макет цел. Порог времени — для -O
    // (штатный scripts/test.sh). Под -Onone Swift-массивы без оптимизации
    // дают ~100× замедления, это не показатель архитектуры.
    do {
        let t0 = Date()
        let big = makeLayeredScene(w: 854, h: 480, seed: 7)
        let dt = Date().timeIntervalSince(t0)
        expectEqual(big.background.count, 854 * 480 * 3, "480p: размер фона")
        expectEqual(big.channelsCHW().count, 854 * 480 * 8, "480p: 8 каналов CHW")
        expect(dt < 5, "480p: рендер < 5 с (фактически \(String(format: "%.2f", dt)) с)")
    }

    // Предпросмотр: 4 PNG создаются и не пустые.
    do {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("yume_layers_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let prefix = dir.appendingPathComponent("scene").path
        try saveLayerPreview(scene: scene, prefix: prefix)
        for name in ["scene_composite.png", "scene_background.png",
                     "scene_objects.png", "scene_light.png"] {
            let url = dir.appendingPathComponent(name)
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
            expect(size > 100, "превью \(name) записано (\(size) байт)")
        }
    } catch {
        expect(false, "превью слоёв: ошибка \(error)")
    }
}

// MARK: - Универсальная слоистая архитектура

func testUniversalLayers() {
    section("Универсальные слои: любые комбинации экспертов")

    // Фон дополняется объектными экспертами — фон при этом НЕ меняется.
    do {
        let bg = makeComposedScene(w: 96, h: 64, recipe: .backdrop(), seed: 9)
        let ext = makeComposedScene(w: 96, h: 64,
            recipe: .backdrop().adding(LayerSpec(.figure, count: 2, role: .objects)), seed: 9)
        expectEqual(bg.flattened().background, ext.flattened().background,
                    "добавление объектных экспертов не меняет фон (побитово)")
        expect(bg.composite != ext.composite, "добавление объектов меняет кадр")
        expect(ext.flattened().mask.contains(where: { $0 == 1 }), "объекты появились в маске")
        expectEqual(ext.layers.count, bg.layers.count + 2, "каждая фигура = отдельный слой-эксперт")
    }

    // Роль — КТО произвёл слой, а не ЧТО в нём: фон-эксперт может рисовать фигуры.
    do {
        let a = makeComposedScene(w: 96, h: 64, recipe: LayerRecipe([
            LayerSpec(.sky, role: .background),
            LayerSpec(.figure, role: .objects)]), seed: 3)
        let b = makeComposedScene(w: 96, h: 64, recipe: LayerRecipe([
            LayerSpec(.sky, role: .background),
            LayerSpec(.figure, role: .background)]), seed: 3)   // та же семантика
        expectEqual(a.composite, b.composite, "роль эксперта не влияет на композицию")
        expectEqual(a.flattened().mask, b.flattened().mask, "фон-эксперт тоже делает объекты")
    }

    // Любая комбинация: композиция восстанавливается из 8 каналов ТОЧНО.
    let recipes: [(String, LayerRecipe)] = [
        ("пусто", LayerRecipe([])),
        ("только небо", LayerRecipe([LayerSpec(.sky, role: .background)])),
        ("только фигуры", LayerRecipe([LayerSpec(.figure, count: 2, role: .objects)])),
        ("небо+земля+фигуры", LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.ground, role: .background),
            LayerSpec(.figure, count: 3, role: .objects)])),
        ("перекрытие: земля поверх фигур", LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.figure, role: .objects),
            LayerSpec(.ground, role: .background)])),
        ("облако поверх фигур", LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.figure, role: .objects),
            LayerSpec(.clouds, count: 2, role: .background)])),
        ("фон-эксперт делает всё", LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.ground, role: .background),
            LayerSpec(.figure, count: 2, role: .background, group: .objects)])),
        ("только свет", LayerRecipe([LayerSpec(.lightGradient, role: .lighting)])),
        ("свет+фигуры+тени", LayerRecipe([
            LayerSpec(.lightGradient, role: .lighting), LayerSpec(.figure, count: 2, role: .objects),
            LayerSpec(.shadow, count: 2, role: .lighting)])),
        ("солнце+облака", LayerRecipe([
            LayerSpec(.sun, role: .background), LayerSpec(.clouds, count: 4, role: .background)])),
        ("классика", .classic(figureCount: 3)),
    ]
    for (name, recipe) in recipes {
        let scene = makeComposedScene(w: 96, h: 64, recipe: recipe, seed: 77).flattened()
        let rebuilt = composeLayers(w: scene.w, h: scene.h, background: scene.background,
                                    objects: scene.objects, mask: scene.mask, light: scene.light)
        expectEqual(rebuilt, scene.composite, "[\(name)] 8 каналов → кадр побитово")
        expect(scene.mask.allSatisfy { $0 == 0 || $0 == 1 }, "[\(name)] маска бинарна")
        expect(scene.composite.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
               "[\(name)] кадр в [0,1]")
        expect(scene.light.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
               "[\(name)] свет в [0,1]")
    }

    // Перекрытие: земля, отрисованная ПОСЛЕ фигуры, скрывает её — маска это видит.
    do {
        let alone = makeComposedScene(w: 96, h: 64, recipe: LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.figure, role: .objects)]), seed: 5)
            .flattened()
        let occluded = makeComposedScene(w: 96, h: 64, recipe: LayerRecipe([
            LayerSpec(.sky, role: .background), LayerSpec(.figure, role: .objects),
            LayerSpec(.ground, role: .background)]), seed: 5).flattened()
        let a = alone.mask.filter { $0 == 1 }.count
        let b = occluded.mask.filter { $0 == 1 }.count
        expect(b < a, "перекрытие: скрытая часть фигур ушла из маски (\(a) → \(b))")
    }

    // Масштаб: 8 облаков + 8 фигур на 480p рендерится целиком (-O).
    do {
        let heavy = LayerRecipe([
            LayerSpec(.sky, role: .background),
            LayerSpec(.clouds, count: 8, role: .background),
            LayerSpec(.ground, role: .background),
            LayerSpec(.figure, count: 8, role: .objects),
            LayerSpec(.lightGradient, role: .lighting),
        ])
        let t0 = Date()
        let s = makeComposedScene(w: 854, h: 480, recipe: heavy, seed: 11).flattened()
        let dt = Date().timeIntervalSince(t0)
        expectEqual(s.channelsCHW().count, 854 * 480 * 8, "480p: 8 каналов при 16 слоях")
        expect(dt < 8, "480p: 16 слоёв < 8 с (\(String(format: "%.2f", dt)) с)")
    }
}

// MARK: - Качество и покадровый план

func testSceneQuality() {
    section("Качество не хуже входа · ускорение движения")

    // PSNR: идентично → ∞, известная ошибка → 20 дБ.
    let ref = [Float](repeating: 0.5, count: 300)
    expect(psnr(reference: ref, candidate: ref) == .infinity, "PSNR идентичных кадров = ∞")
    let shifted = [Float](repeating: 0.6, count: 300)
    expectClose(Double(psnr(reference: ref, candidate: shifted)), 20, 0.01,
                "PSNR при MSE 0.01 = 20 дБ")
    expect(luminanceSSIM(reference: ref, candidate: ref) > 0.999, "SSIM идентичных = 1")
    expect(luminanceSSIM(reference: ref, candidate: shifted) < 1, "SSIM differing < 1")

    // Ворот качества: не хуже входа → принят, регрессия → отклонён.
    do {
        let floor = QualityFloor(minPSNR: 30, minSSIM: 0.90)
        expect(floor.accepts(reference: ref, reconstruction: ref), "пол: идеальная реконструкция принята")
        let bad = [Float](repeating: 0.2, count: 300)
        expect(!floor.accepts(reference: ref, reconstruction: bad), "пол: регрессия отклонена")
        expect(floor.accepts(current: 33, previous: 32), "пол: улучшение принято")
        expect(!floor.accepts(current: 30.1, previous: 32), "пол: просадка > 0.5 дБ отклонена")
        expect(floor.report(reference: ref, reconstruction: ref).contains("принят"),
               "отчёт качества: принят")
    }

    // Покадровый план: статика на ключевых кадрах, объекты — каждый кадр.
    do {
        let plan = FramePlan(keyframeInterval: 8)
        expect(plan.needsRecompute(frame: 0, kind: .sky), "ключевой кадр: небо пересчитано")
        expect(!plan.needsRecompute(frame: 3, kind: .sky), "промежуточный кадр: небо из кэша")
        expect(plan.needsRecompute(frame: 3, kind: .figure), "объекты пересчитываются покадрово")
        expect(plan.needsRecompute(frame: 3, kind: .sky, motionScore: 0.9),
               "смена плана: кэш сброшен")
        let layers: [LayerSpec] = [
            LayerSpec(.sky, role: .background), LayerSpec(.ground, role: .background),
            LayerSpec(.lightGradient, role: .lighting), LayerSpec(.figure, count: 2, role: .objects),
        ]
        expect(plan.estimatedSpeedup(layers: layers) > 1.5,
               "кэш статичных экспертов даёт ускорение > 1.5×")
    }
}

// MARK: - 8-канальный layered VAE: математика потерь (CPU, без Metal)

func testLayeredLossMath() {
    section("Layered loss: композиция + градиенты (CPU)")

    // 1. Точная сборка в CHW-раскладке обязана совпадать с HWC-композитором
    //    рендерера. Это тот самый инвариант composeLayers(flattened) == composite.
    let scene = makeLayeredScene(w: 32, h: 32, seed: 2026)
    let t = LayeredTarget(scene: scene)
    expectEqual(t.channelsCHW.count, 8 * 32 * 32, "таргет: 8·w·h значений")
    expectEqual(t.compositeCHW.count, 3 * 32 * 32, "композит: 3·w·h значений")
    let exactCHW = composeCHW(w: 32, h: 32, recon: t.channelsCHW)
    expect(maxAbsDiff(exactCHW, t.compositeCHW) < 1e-6,
           "composeCHW(таргет) == composite поканально")
    let exactHWC = composeHWC(w: 32, h: 32, recon: t.channelsCHW)
    expect(maxAbsDiff(exactHWC, scene.composite) < 1e-6,
           "composeHWC(таргет) == composite кадра")

    // 2. Идеальная реконструкция → нулевая потеря, возмущённая → > 0.
    expectClose(layeredLoss(recon: t.channelsCHW, target: t), 0, 1e-9,
                "идеальная реконструкция: loss = 0")
    var bad = t.channelsCHW
    for i in 0..<bad.count { bad[i] = min(1, max(0, bad[i] + 0.05)) }
    expect(layeredLoss(recon: bad, target: t) > 0, "испорченная реконструкция: loss > 0")

    // 3. Аналитический градиент == конечная разность по ВСЕМ 8 каналам,
    //    включая маску и свет: именно они несут композиционный член.
    let n = 4 * 4
    let smallChannels: [Float] = (0..<(8 * n)).map {
        0.25 + 0.03 * Float($0 / n) + 0.01 * Float($0 % n)
    }
    let smallComposite: [Float] = [Float](repeating: 0.42, count: 3 * n)
    let small = LayeredTarget(w: 4, h: 4,
                              channelsCHW: smallChannels,
                              compositeCHW: smallComposite)
    var recon = [Float](repeating: 0, count: 8 * n)
    for i in 0..<recon.count { recon[i] = 0.30 + 0.02 * Float(i % 7) }
    let g = layeredLossGradient(recon: recon, target: small,
                                layerWeight: 1, composeWeight: 1)
    expectEqual(g.count, 8 * n, "градиент: 8·w·h значений")
    let eps: Float = 1e-3
    var worst = 0.0
    for ch in 0..<8 {
        for i in [0, 5, 11] {
            let idx = ch * n + i
            var rp = recon, rm = recon
            rp[idx] += eps
            rm[idx] -= eps
            let lp = layeredLoss(recon: rp, target: small,
                                 layerWeight: 1, composeWeight: 1)
            let lm = layeredLoss(recon: rm, target: small,
                                 layerWeight: 1, composeWeight: 1)
            let numeric: Double = (lp - lm) / (2 * Double(eps))
            let analytic = Double(g[idx])
            let denom = max(1e-6, abs(analytic) + abs(numeric))
            let rel = abs(analytic - numeric) / denom
            worst = max(worst, rel)
            expect(rel < 0.05,
                   String(format: "канал %d пикс %d: аналит %.3e / числ %.3e (%.2f%%)",
                          ch, i, analytic, numeric, rel * 100))
        }
    }
    print(String(format: "      макс. относительная ошибка градиента: %.2f%%", worst * 100))

    // 4. Композиционный член реально влияет на градиенты маски, света и фона.
    let gNoComp = layeredLossGradient(recon: recon, target: small,
                                      layerWeight: 1, composeWeight: 0)
    expect(maxAbsDiff(Array(g[6 * n..<8 * n]), Array(gNoComp[6 * n..<8 * n])) > 1e-6,
           "без composeWeight градиенты маски/света меняются")
    expect(maxAbsDiff(Array(g[0..<3 * n]), Array(gNoComp[0..<3 * n])) > 1e-6,
           "composeWeight влияет и на каналы фона")

    // 5. Пол качества считается по СОБРАННОМУ кадру, а не по слоям по отдельности.
    let q = composedQuality(recon: t.channelsCHW, target: t)
    expect(q.psnr == .infinity && q.ssim > 0.999, "идеальная сборка: PSNR ∞ / SSIM 1")
    let qBad = composedQuality(recon: bad, target: t)
    expect(qBad.psnr.isFinite && qBad.psnr < 40, "испорченная сборка: PSNR конечен")

    // 6. Бюджет памяти layered-пути учитывает дополнительные буферы.
    expect(layeredActivationBytes(w: 32, h: 32) > 0, "layered-буферы учтены в бюджете")
    expectEqual(layeredActivationBytes(w: 64, h: 64),
                4 * layeredActivationBytes(w: 32, h: 32),
                "память layered-буферов ∝ площади")
}

// MARK: - 8-канальный layered VAE: Metal-ядро градиента

func testLayeredLossGPU() {
    section("Layered loss: Metal-ядро градиента")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24, n = w * h
    let t = LayeredTarget(scene: makeLayeredScene(w: w, h: h, seed: 77))
    let tg = LayeredTargetGPU(engine: engine, target: t)

    // Возмущённая реконструкция строго внутри (0,1): кламп не зануляет градиент.
    var recon = [Float](repeating: 0, count: 8 * n)
    for i in 0..<recon.count { recon[i] = 0.30 + 0.02 * Float(i % 7) }
    let r = GPUTensor(engine: engine, data: recon, rows: 8)

    session.begin()
    let gpuGrad = session.layeredGrad(recon: r, target: tg,
                                      layerWeight: 1, composeWeight: 1)
    _ = session.end()

    let got = gpuGrad.toArray()
    let cpu = layeredLossGradient(recon: recon, target: t,
                                  layerWeight: 1, composeWeight: 1)
    expectEqual(got.count, 8 * n, "Metal: форма градиента 8·w·h")
    expect(got.allSatisfy { $0.isFinite }, "Metal: градиент без NaN")
    expect(maxAbsDiff(got, cpu) < 1e-5,
           String(format: "Metal-градиент == CPU-аналитике (max Δ %.2e)", maxAbsDiff(got, cpu)))
    expect(got.contains { abs($0) > 1e-7 }, "Metal: градиент невырожден")
}

// MARK: - 8-канальный layered VAE: форма, память, чекпоинт

func testLayeredVAEShape() {
    section("Layered VAE: 8 каналов вместо 3")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let t = LayeredTarget(scene: makeLayeredScene(w: w, h: h, seed: 11))
    let x = GPUTensor(engine: engine, data: t.channelsCHW, rows: 8)

    let vae3 = ConvVAEGPU(engine: engine)
    let vae8 = ConvVAEGPU(engine: engine, channels: 8)
    expectEqual(vae3.channels, 3, "старый VAE остался 3-канальным")
    expectEqual(vae8.channels, 8, "новый VAE: 8 каналов")
    expectEqual(vae8.e1.ic, 8, "e1 принимает 8 каналов")
    expectEqual(vae8.d3.oc, 8, "d3 выдаёт 8 каналов")
    expectClose(ConvVAEGPU.layeredCompressionRatio, 32, 1e-9,
                "сжатие 8-канального входа = 32×")
    expectClose(ConvVAEGPU.compressionRatio, 12, 1e-9,
                "сжатие 3-канального входа = 12×")

    session.begin()
    let cache = vae8.forward(session, image: x, w: w, h: h, deterministic: true)
    _ = session.end()
    expectEqual(cache.recon.rows, 8, "реконструкция: 8 каналов")
    expectEqual(cache.recon.cols, w * h, "реконструкция: w×h")
    let r = cache.recon.toArray()
    expect(r.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
           "8 каналов в [0,1] (сигмоида)")
    expectEqual(cache.z.rows, ConvVAEGPU.latentC, "латент не изменился")

    // Бюджет VRAM: 8 каналов + layered-буферы дороже, но кадр всё ещё влезает.
    let b3 = convVAEBytesPerSample(w: 856, h: 480)
    let b8 = convVAEBytesPerSample(w: 856, h: 480, channels: 8, layered: true)
    expect(b8 > b3, "8-канальный путь требует больше активаций")
    expect(Double(b8) / 1_048_576 < 650, "480p layered-кадр укладывается в бюджет")
    expect(effectiveBatchSize(requested: 16, w: 856, h: 480,
                              budgetBytes: 8 * 1024 * 1024 * 1024 / 2,
                              channels: 8, layered: true) >= 1,
           "layered-пачка ≥ 1")

    // Старый 3-канальный чекпоинт несовместим с 8-канальной моделью:
    // это не битый файл, а другая архитектура (код 11).
    do {
        try vae8.load(from: vae3.matrices)
        expect(false, "3-канальный чекпоинт отклонён 8-канальной моделью")
    } catch {
        expect(isIncompatibleCheckpoint(error), "несовместимость распознана (код 11)")
        expectEqual((error as NSError).code, 11, "код ошибки = 11")
    }
}

// MARK: - 8-канальный layered VAE: градиенты сети vs конечные разности

func testLayeredVAEGradients() {
    section("Layered VAE: градиенты vs конечные разности")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)
    let w = 32, h = 24
    let beta: Float = 0.05
    let t = LayeredTarget(scene: makeLayeredScene(w: w, h: h, seed: 33))
    let tg = LayeredTargetGPU(engine: engine, target: t)
    let x = GPUTensor(engine: engine, data: t.channelsCHW, rows: 8)
    let vae = ConvVAEGPU(engine: engine, channels: 8)

    /// Полная потеря: MSE по 8 слоям + MSE собранного кадра + β·KL.
    /// Именно её должен дифференцировать `layered_grad_kernel`.
    func loss() -> Double {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
        _ = session.end()
        return layeredVAELoss(cache: c, target: t,
                              layerWeight: 1, composeWeight: 1, beta: beta)
    }

    // Небольшая тренировка, чтобы веса ушли от инициализации.
    for step in 1...12 {
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
        let (_, g) = vae.backwardLayered(session, c, target: tg, beta: beta)
        vae.applyAdam(session, grads: g, t: step, lr: 0.002)
        _ = session.end()
    }

    session.begin()
    let cache = vae.forward(session, image: x, w: w, h: h, deterministic: false, seed: 5)
    let (_, grads) = vae.backwardLayered(session, cache, target: tg, beta: beta)
    _ = session.end()

    var maxGrad = 0.0
    for g in grads {
        maxGrad = max(maxGrad, g.toArray().map { abs(Double($0)) }.max() ?? 0)
    }
    expect(maxGrad > 1e-6,
           String(format: "layered-градиенты невырождены (max |g| = %.2e)", maxGrad))

    let eps: Float = 1e-2
    var checked = 0
    for layerIdx in [2, 3] {   // e3 — перед латентом, d1 — сразу после
        let g = grads[layerIdx * 2].toArray()
        let top = (0..<g.count).sorted { abs(g[$0]) > abs(g[$1]) }.prefix(2)
        for weightIdx in top {
            let layer = vae.layers[layerIdx]
            let analytic = Double(g[weightIdx])
            let orig = layer.W.getValue(weightIdx)
            layer.W.setValue(weightIdx, orig + eps)
            let lp = loss()
            layer.W.setValue(weightIdx, orig - eps)
            let lm = loss()
            layer.W.setValue(weightIdx, orig)

            let numeric = (lp - lm) / (2 * Double(eps))
            let rel = abs(analytic - numeric) / max(1e-9, abs(analytic) + abs(numeric))
            expect(rel < 0.12,
                   String(format: "слой %d вес %d: аналит %.4e числ %.4e (отн %.1f%%)",
                          layerIdx, weightIdx, analytic, numeric, rel * 100))
            checked += 1
        }
    }
    expectEqual(checked, 4, "проверено 4 точки")
}

// MARK: - 8-канальный layered VAE: тренер и пол качества

func testLayeredTrainer() {
    section("Layered trainer: сходимость + ворота качества")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let report = trainLayeredVAE(steps: 60, w: 32, h: 24, seed: 2026)
    expect(report.firstLoss.isFinite && report.lastLoss.isFinite,
           "потери конечны на обоих концах обучения")
    expect(report.lastLoss < report.firstLoss,
           String(format: "потеря упала: %.5f → %.5f", report.firstLoss, report.lastLoss))
    expect(report.psnr.isFinite, "PSNR собранного кадра конечен")
    expect(report.ssim.isFinite && report.ssim >= 0 && report.ssim <= 1,
           "SSIM собранного кадра в [0,1]")
    // Ворота обязаны быть честными: короткий прогон на 32×24 не может
    // выдавать 30 дБ, и «принят» должен совпадать с формальным порогом.
    expectEqual(report.acceptedByFloor,
                report.psnr >= 30 && report.ssim >= 0.90,
                "ворота качества согласованы с порогом 30 дБ / 0.90")
    expect(report.compositeChannels == 3, "выход сборки — RGB, 3 канала")
}

// MARK: - Жёсткий потолок 90% после перегрева

func testDutyCap90() {
    section("Предохранитель GPU: не выше 90%")

    expectClose(safeDutyCap, 0.90, 1e-9, "потолок скважности = 90%")
    expectEqual(defaultGpuPowerPercent, 90, "мощность по умолчанию = 90%")
    expectClose(effectiveDutyCeiling(powerPercent: 100), 0.90, 1e-9,
                "даже 100% в GUI упирается в 90%")
    expectClose(effectiveDutyCeiling(powerPercent: 200), 0.90, 1e-9,
                "некорректные 200% тоже упираются в 90%")
    expect(effectiveDutyCeiling(powerPercent: 85) < safeDutyCap,
           "85% остаётся ниже потолка")
    expect(effectiveDutyCeiling(powerPercent: 100) <= maxSafeDuty,
           "предохранитель не слабее физического предела")

    let guard90 = GpuDutyGuard()
    expectClose(guard90.duty, 0.90, 1e-9, "guard: скважность 90%")
    let pause = guard90.pauseMs(gpuMs: 100)
    expect(pause > 0, "guard: после шага 100 мс есть пауза")
    expectClose(100.0 / (100.0 + pause), 0.90, 1e-6,
                "guard: фактическая занятость GPU = 90%")
    expectEqual(guard90.pauseMs(gpuMs: 0), 0, "guard: нулевой шаг → без паузы")
    expectEqual(guard90.emergencyPauseMs(tempC: 70), 0, "guard: 70 °C без аварии")
    expect(guard90.emergencyPauseMs(tempC: emergencyTempC + 1) > 0,
           "guard: аварийная температура → принудительное остывание")
    expect(Params().gpuPowerPercent == 90, "Params по умолчанию = 90%")
}

/// `runGpuHeat` — диагностический прогрев, но он гоняет тот же 856×480
/// forward+backward, что и обучение, и поэтому не имеет права обходить потолок.
/// До исправления там стоял `ceiling: auto ? maxSafeDuty : 1.0`, то есть 98–100%.
func testHeatDutyCap() {
    section("Прогрев GPU: 90% без обходных путей")

    expectClose(clampedGpuDuty(1.0), safeDutyCap, 1e-9,
                "gpuheat: ручной режим 100% упирается в 90%")
    expectClose(clampedGpuDuty(0.99), safeDutyCap, 1e-9,
                "gpuheat: 99% упирается в 90%")
    expectClose(clampedGpuDuty(0.90), safeDutyCap, 1e-9,
                "gpuheat: ровно 90% разрешены")
    expectClose(clampedGpuDuty(0.75), 0.75, 1e-9,
                "gpuheat: 75% проходят как есть")
    let autoDuty = clampedGpuDuty(-1)
    expect(autoDuty.isFinite && autoDuty >= 0.05 && autoDuty <= safeDutyCap,
           "gpuheat: авто-режим (duty ≤ 0) остаётся в коридоре")
    let nanDuty = clampedGpuDuty(Double.nan)
    expect(nanDuty.isFinite && nanDuty <= safeDutyCap,
           "gpuheat: NaN не превращается в полную нагрузку")

    // Сам регулятор с новым потолком не должен выходить за 90% ни при
    // каких показаниях датчика, включая «холодно» (там он разгоняется).
    var power = PowerController(ceiling: safeDutyCap, startDuty: 0.5)
    expectClose(power.ceiling, safeDutyCap, 1e-9,
                "PowerController: потолок = 90%")
    for temp in stride(from: 40.0, through: 100.0, by: 5.0) {
        let stats = GpuStats(temperatureC: temp, coreClockMHz: 400,
                             activityPercent: 90, powerW: 30)
        let d = power.update(stats: stats)
        expect(d <= safeDutyCap + 1e-9,
               "PowerController при \(Int(temp)) °C не выше 90%")
    }
    expect(power.duty <= safeDutyCap + 1e-9,
           "PowerController после 13 замеров не выше 90%")

    // Пауза при 90% обязана быть положительной: именно она отдаёт 10%
    // времени WindowServer и не даёт повториться watchdog-перезагрузке.
    let guard90 = GpuDutyGuard(powerPercent: 100)
    expect(guard90.pauseMs(gpuMs: 150) > 0,
           "даже запрос 100% оставляет паузу на остывание")
}

// MARK: - main

func runAll() {
    testHTTP()
    testCheckpoints()
    testBackups()
    testGPUOps()
    testConvPrimitives()
    testBufferLifecycle()
    testConvVAEShape()
    testVAEPrimitives()
    testVAEVariational()
    testVAEGradients()
    testVAEConvergence()
    testVRAMBudget()
    testProcessedMarks()
    testReproducibility()
    testCheckpointCompat()
    testTrainStepLogging()
    testThermalGovernor()
    testPowerController()
    testPureLogic()
    testLayerRenderer()
    testUniversalLayers()
    testSceneQuality()
    testLayeredLossMath()
    testLayeredLossGPU()
    testLayeredVAEShape()
    testLayeredVAEGradients()
    testLayeredTrainer()
    testDutyCap90()
    testHeatDutyCap()
}

let started = Date()
runAll()
let elapsed = Date().timeIntervalSince(started)

print("")
print("────────────────────────────────────────")
print("Проверок: \(gChecks), провалено: \(gFailures), время: \(String(format: "%.2f", elapsed)) c")
if gFailures == 0 {
    print("РЕЗУЛЬТАТ: PASS")
    exit(0)
} else {
    print("РЕЗУЛЬТАТ: FAIL (\(gFailures))")
    exit(1)
}
