// YumeLayers — группа тестов (выделено из main.swift).
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import Metal
import AVFoundation
import UniformTypeIdentifiers
import Accelerate
import VideoToolbox
import IOKit

func testGradSanitizeBatched() {
    section("GradSanitize: батчевая санация без аллокаций на тензор")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared

    // 12 тензоров — столько градиентов у ConvVAE (6 слоёв × W,b).
    var grads: [GPUTensor] = []
    for i in 0..<12 {
        var data = (0..<64).map { Float($0 % 7) * 0.001 + Float(i) * 0.0001 }
        if i == 0 {
            data[0] = .nan
            data[1] = .infinity
            data[2] = -.infinity
        }
        grads.append(GPUTensor(engine: engine, data: data, rows: 8))
    }

    // Прогрев: первый вызов наполняет пул staging-буферов.
    sanitizeGrads(grads, maxNorm: 1.0)

    let before = engine.totalStagingAllocations
    let calls = 50
    for _ in 0..<calls { sanitizeGrads(grads, maxNorm: 1.0) }
    let delta = engine.totalStagingAllocations - before
    expect(delta <= 2,
           String(format: "санация %d раз: %d новых staging-буферов (ожидается ≤ 2)",
                  calls, delta))

    // Семантика: NaN/Inf → 0 (проверяем после последнего вызова).
    let a0 = grads[0].toArray()
    expectEqual(a0[0], 0, "NaN заменён на 0")
    expectEqual(a0[1], 0, "+Inf заменён на 0")
    expectEqual(a0[2], 0, "-Inf заменён на 0")

    // Клиппинг: суммарная норма не превышает maxNorm, направление сохраняется.
    var big: [GPUTensor] = []
    for _ in 0..<4 {
        big.append(GPUTensor(engine: engine,
                             data: [Float](repeating: 10, count: 64), rows: 8))
    }
    sanitizeGrads(big, maxNorm: 1.0)
    var norm: Float = 0
    for g in big { for v in g.toArray() { norm += v * v } }
    let n = sqrt(norm)
    expect(n <= 1.0001, String(format: "клиппинг: норма %.4f ≤ 1", n))
    expect(n > 0.9, String(format: "клиппинг сохраняет направление: норма %.4f ≈ 1", n))
}


func testTrainingControlLifecycle() {
    section("TrainingControl: жизненный цикл остановки")

    let tc = TrainingControl.shared
    tc.reset()
    expect(!tc.isRunning, "до старта прогон не активен")
    expect(!tc.isCancelled, "до старта отмена сброшена")

    tc.beginRun()
    expect(tc.isRunning, "beginRun: прогон активен")

    // graceSeconds большой: watchdog не должен успеть выстрелить за тест.
    // Проверяем только состояние флагов, а не аварийный exit.
    tc.cancel(graceSeconds: 60)
    expect(tc.isCancelled, "cancel: флаг отмены выставлен")

    tc.endRun()
    expect(!tc.isRunning, "endRun: прогон завершён")

    tc.reset()
    expect(!tc.isCancelled, "reset: отмена снята")
}


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


func testPrepareCheckpointForTraining() {
    section("Подготовка чекпоинта: resume=false → старый удаляется")

    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_ckpt_prep_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let existing = dir.appendingPathComponent("conv.safetensors").path
    let payload = Data("weights".utf8)
    try? payload.write(to: URL(fileURLWithPath: existing))

    // resume=true: файл НЕ трогается.
    let kept = prepareCheckpointForTraining(convPath: existing, resume: true)
    guard case .keep = kept else {
        expect(false, "resume=true → чекпоинт сохраняется"); return
    }
    expect(FileManager.default.fileExists(atPath: existing),
           "resume=true: файл не удалён")

    // resume=false + файл есть: старый удаляется, бэкап создаётся.
    let outcome = prepareCheckpointForTraining(convPath: existing, resume: false)
    guard case .removed(let backup) = outcome else {
        expect(false, "resume=false с существующим файлом → удаление"); return
    }
    expect(!FileManager.default.fileExists(atPath: existing),
           "старый чекпоинт удалён")
    expect(FileManager.default.fileExists(atPath: backup),
           "бэкап создан")
    expect((try? Data(contentsOf: URL(fileURLWithPath: backup))) == payload,
           "содержимое бэкапа совпадает")

    // resume=false + файла нет: свежий старт, без ошибок.
    let missing = dir.appendingPathComponent("missing.safetensors").path
    guard case .fresh = prepareCheckpointForTraining(convPath: missing, resume: false) else {
        expect(false, "нет файла → свежий старт"); return
    }
    expect(!FileManager.default.fileExists(atPath: missing),
           "файл не создавался")
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


