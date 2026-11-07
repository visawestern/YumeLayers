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
var gFilter: String? = {
    let args = Array(CommandLine.arguments.dropFirst())
    if let i = args.firstIndex(of: "--filter"), i + 1 < args.count { return args[i + 1] }
    return args.isEmpty ? nil : args[0]
}()

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

func testParamAPI() {
    section("Параметры API: приведение значений JSON → поля/переключатели")

    // paramStringValue: число/строка/bool → строка текстового поля.
    expectEqual(paramStringValue("0.01"), "0.01", "строка передаётся как есть")
    expectEqual(paramStringValue(0.01), "0.010000", "дробное число → 6 знаков")
    expectEqual(paramStringValue(80), "80", "целое число → без точки")
    expectEqual(paramStringValue(2000), "2000", "крупное целое → без точки")
    expectEqual(paramStringValue(true), "1", "bool → '1'")
    expectEqual(paramStringValue(3.5), "3.500000", "дробное 3.5 → 6 знаков")
    expect(paramStringValue([1]) == nil, "несовместимый тип → nil")

    // paramBoolValue: bool/0/1/строка → Bool.
    expectEqual(paramBoolValue(true), true, "bool true")
    expectEqual(paramBoolValue(false), false, "bool false")
    expectEqual(paramBoolValue(1), true, "число 1 → true")
    expectEqual(paramBoolValue(0), false, "число 0 → false")
    expectEqual(paramBoolValue("true"), true, "'true' → true")
    expectEqual(paramBoolValue("off"), false, "'off' → false")
    expectEqual(paramBoolValue("yes"), true, "'yes' → true")
    expect(paramBoolValue("abc") == nil, "непарсимо → nil")
    expect(paramBoolValue(2) == nil, "2 не является bool-значением → nil")
}

func testTensorBufferPool() {
    section("TensorBufferPool: переиспользование приватных буферов")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let session = GpuSession(engine)

    // 1. Пул: take → retire → reclaim отдаёт ТОТ ЖЕ буфер.
    do {
        let b1 = engine.takeBuffer(4096)
        let b2 = engine.takeBuffer(4096)
        // Оба nil: пул пуст.
        expect(b1 == nil && b2 == nil, "пустой пул не отдаёт буферы")
        // Создаём два буфера вручную и кладём их в пул.
        if let m1 = engine.device.makeBuffer(length: 4096, options: .storageModePrivate),
           let m2 = engine.device.makeBuffer(length: 4096, options: .storageModePrivate) {
            engine.retireBuffer(m1)
            engine.retireBuffer(m2)
            engine.reclaimRetiredBuffers()
            let r1 = engine.takeBuffer(4096)
            let r2 = engine.takeBuffer(4096)
            // Переиспользование: вернулись те же объекты (в любом порядке).
            expect((r1 === m1 || r1 === m2), "takeBuffer вернул буфер из пула (1)")
            expect((r2 === m1 || r2 === m2), "takeBuffer вернул буфер из пула (2)")
            expect(r1 !== r2, "два takeBuffer дают разные буферы")
        }
    }

    // 2. Поведенческий: на длинном прогоне число РЕАЛЬНЫХ аллокаций не растёт
    //    линейно. Моделируем ритм тренировки: каждый шаг = begin/create/end,
    //    и create/end синхронизирует GPU и возвращает освобождённые буферы
    //    в пул переиспользования.
    do {
        let outerSteps = 40      // «шаги» обучения
        let innerPerStep = 10    // тензоров на шаг
        let iterations = outerSteps * innerPerStep

        // Прогрев: пара итераций наполняет пул.
        for _ in 0..<2 {
            session.begin()
            let t = GPUTensor(engine: engine, rows: 64, cols: 64)
            _ = t.n
            _ = session.end()
        }
        let startAlloc = engine.totalBufferAllocations
        for _ in 0..<outerSteps {
            session.begin()
            for _ in 0..<innerPerStep {
                let t = GPUTensor(engine: engine, rows: 64, cols: 64)
                _ = t.n
            }
            _ = session.end()
        }
        let delta = engine.totalBufferAllocations - startAlloc
        // После прогрева аллокации должны быть ~первый шаг (внутри шага буферы
        // живут одновременно и не переиспользуются), а не расти линейно.
        expect(delta < iterations / 8,
               String(format: "пул: %d созданий породили %d аллокаций (< %d)",
                      iterations, delta, iterations / 8))
    }
}

func testTensorStagingPool() {
    section("TensorStagingPool: staging-буферы переиспользуются")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let t = GPUTensor(engine: engine, rows: 8, cols: 8)
    let vals = (0..<64).map { Float($0) * 0.01 }

    // Прогрев: первый setData/toArray могут выделить staging.
    t.setData(vals)
    _ = t.toArray()

    let before = engine.totalStagingAllocations
    let calls = 100
    var lastCount = 0
    for _ in 0..<calls {
        t.setData(vals)
        let a = t.toArray()
        lastCount = a.count
    }
    let delta = engine.totalStagingAllocations - before
    expectEqual(lastCount, 64, "toArray вернул 64 значения")
    expect(delta <= 2,
           String(format: "%d циклов setData+toArray: %d новых staging-буферов (ожидается ≤ 2)",
                  calls, delta))
}

func testGpuWaitPolicy() {
    section("GpuWaitPolicy: показания драйвера и ожидание command buffer")

    // 1) Правдоподобное показание драйвера берём как есть.
    expectEqual(boundedGpuMs(reportedMs: 2, wallMs: 10), 2,
                "правдоподобное показание не меняется")

    // 2) Драйвер сообщил 18 с при 12 мс стенного времени. GPU физически
    //    не мог быть занят дольше самого шага — такое показание отбрасываем,
    //    иначе регулятор превращает его в паузу в 2 с на каждом шаге
    //    (обучение замедляется в 100 раз при полностью простаивающей GPU).
    expectEqual(boundedGpuMs(reportedMs: 18_000, wallMs: 12), 12,
                "невозможное показание ограничивается стенным временем")

    // 3) Драйвер не дал времени вовсе — берём стенное.
    expectEqual(boundedGpuMs(reportedMs: 0, wallMs: 8), 8,
                "без показания драйвера — стенное время")

    // 4) Ни показания, ни стенного времени — ноль (не NaN!).
    expectEqual(boundedGpuMs(reportedMs: .nan, wallMs: 0), 0,
                "NaN и отсутствие стенного времени дают 0")

    // 5) Небольшое превышение стенного времени — норма: буфер мог ждать
    //    в очереди (WindowServer, другой клиент).
    expectEqual(boundedGpuMs(reportedMs: 30, wallMs: 12, toleranceMs: 50), 30,
                "превышение в пределах допуска не трогаем")

    // 6) Главное свойство: пауза регулятора не может взорваться из-за
    //    показания драйвера. 18 000 мс при duty 0.9 дали бы 2000 мс паузы;
    //    после ограничения — около 1.3 мс.
    let pause = dutySleepMs(gpuMs: boundedGpuMs(reportedMs: 18_000, wallMs: 12),
                            duty: 0.9, maxPauseMs: 2000)
    expect(pause < 5,
           String(format: "пауза по ограниченному показанию: %.2f мс (ожидается < 5)", pause))

    // 7) Обработчик сработал — ждать больше нечего.
    expectEqual(gpuWaitOutcome(state: .running, handlerSignaled: true,
                               elapsedSeconds: 0.01, deadlineSeconds: 30), .done,
                "сработавший обработчик — готово")

    // 8) Обработчик потерялся, но буфер уже выполнен — это и есть путь
    //    спасения от «GPU не ответила»: ждать handler'а бессмысленно.
    expectEqual(gpuWaitOutcome(state: .completed, handlerSignaled: false,
                               elapsedSeconds: 0.01, deadlineSeconds: 30), .done,
                "выполненный буфер без обработчика — готово")

    // 9) Буфер ещё выполняется, время есть — продолжаем ждать.
    expectEqual(gpuWaitOutcome(state: .running, handlerSignaled: false,
                               elapsedSeconds: 1, deadlineSeconds: 30), .keepWaiting,
                "идущий буфер — продолжаем ждать")

    // 10) Время вышло — таймаут.
    expectEqual(gpuWaitOutcome(state: .running, handlerSignaled: false,
                               elapsedSeconds: 30, deadlineSeconds: 30), .timedOut,
                "превышен предел — таймаут")

    // 11) Ошибка драйвера — дальше ждать нечего.
    expectEqual(gpuWaitOutcome(state: .failed, handlerSignaled: false,
                               elapsedSeconds: 1, deadlineSeconds: 30), .done,
                "ошибка буфера — ждать бессмысленно")

    // 12) Пауза регулятора режется на короткие срезы: «Стоп» должен
    //     срабатывать сразу, а не через 2 с.
    expectEqual(pauseSlices(totalMs: 250), [100, 100, 50],
                "пауза 250 мс делится на срезы по 100 мс")
    expectEqual(pauseSlices(totalMs: 0), [Double](),
                "нулевая пауза — без срезов")
    expectEqual(pauseSlices(totalMs: 40), [40],
                "пауза меньше среза — один срез")
}

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

    // --- Доля работы: сколько спать после шага ---
    // GPU занят `gpuMs`, значит доля занятости = gpu/(gpu+sleep).
    // Чтобы уложиться в duty: sleep = gpu·(1/duty − 1).
    expectClose(dutySleepMs(gpuMs: 170, duty: 0.55, maxPauseMs: 2000), 170 * (1 / 0.55 - 1),
                0.01, "170 мс при duty 55% → пауза 139 мс")
    expectClose(dutySleepMs(gpuMs: 170, duty: 0.5, maxPauseMs: 2000), 170,
                0.01, "duty 50% → пауза равна времени работы")

    // Главная регрессия: пауза не должна быть нулевой на штатном режиме.
    // Раньше в цикле шага стоял жёсткий Thread.sleep(0.03) — 30 мс на кадр
    // 854×480 при 170 мс работы, то есть доля работы ~85% на много часов.
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
    // потолок доли работы урезали, а расчёт паузы остался прежним.
    let sleepAt99 = dutySleepMs(gpuMs: 170, duty: 0.99, maxPauseMs: 5000)
    expectClose(sleepAt99, 170 * (1 / 0.99 - 1), 1e-9, "99% → пауза по формуле")
    expect(sleepAt99 < 2.5, "99% → пауза меньше 2.5 мс на шаге 170 мс")

    // --- Целевая доля работы по уровню нагрева ---
    let dn = targetDuty(for: .nominal)
    let dw = targetDuty(for: .warm)
    let dh = targetDuty(for: .hot)
    let dc = targetDuty(for: .critical)
    expect(dn < 1 && dn > 0.2, "штатная доля работы в разумных пределах")
    expect(dn > dw && dw > dh && dh > dc, "чем жарче, тем ниже доля работы")
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

    // Без пользовательской паузы работает чистая доля работы.
    let d2 = throttleDecision(gpuMs: 170, wallMs: 170, baselineMs: 0, thermalRaw: 0,
                              floorMs: 0, maxPauseMs: 2000)
    expectClose(d2.sleepMs, 139.09, 0.01, "пауза по доле работы 55%")

    // Тот же кадр, но GPU разогрелась: тепловой уровень режет долю работы.
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
    expectClose(maxSafeDuty, 0.98, 1e-9, "100% мощности → 98% доли работы")

    // Без телеметрии режим не меняем: гадать хуже, чем держать текущий.
    var none = PowerController(ceiling: 0.765, startDuty: 0.5)
    expectClose(none.update(stats: nil), 0.5, 1e-9, "нет датчика — доля работы та же")

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
    // что шум частоты долю работы НЕ меняет.
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

// MARK: - Потолок GPU: изменяемый параметр (до физического предела 98%)

func testDutyCap() {
    section("Предохранитель GPU: параметр мощности соблюдается")

    expectClose(safeDutyCap, 0.98, 1e-9, "физический потолок доли работы = 98%")
    expectEqual(defaultGpuPowerPercent, 90, "мощность по умолчанию = 90%")
    expectClose(effectiveDutyCeiling(powerPercent: 98), 0.98, 1e-9,
                "запрошенные 98% не режутся до 90%")
    expectClose(effectiveDutyCeiling(powerPercent: 100), 0.98, 1e-9,
                "100% упираются только в физический предел 98%")
    expectClose(effectiveDutyCeiling(powerPercent: 200), 0.98, 1e-9,
                "некорректные 200% тоже упираются в 98%")
    expectClose(effectiveDutyCeiling(powerPercent: 85), 0.85, 1e-9,
                "85% проходят как есть")
    expect(effectiveDutyCeiling(powerPercent: 85) < safeDutyCap,
           "85% остаётся ниже потолка")
    expect(effectiveDutyCeiling(powerPercent: 100) <= maxSafeDuty,
           "предохранитель не слабее физического предела")

    let guard90 = GpuDutyGuard()
    expectClose(guard90.duty, 0.90, 1e-9, "guard по умолчанию: доля работы 90%")
    let pause = guard90.pauseMs(gpuMs: 100)
    expect(pause > 0, "guard: после шага 100 мс есть пауза")
    expectClose(100.0 / (100.0 + pause), 0.90, 1e-6,
                "guard: фактическая занятость GPU = 90%")
    expectEqual(guard90.pauseMs(gpuMs: 0), 0, "guard: нулевой шаг → без паузы")
    expectEqual(guard90.emergencyPauseMs(tempC: 70), 0, "guard: 70 °C без аварии")
    expect(guard90.emergencyPauseMs(tempC: emergencyTempC + 1) > 0,
           "guard: аварийная температура → принудительное остывание")

    let guard98 = GpuDutyGuard(powerPercent: 98)
    expectClose(guard98.duty, 0.98, 1e-9, "guard: запрошенные 98% соблюдаются")
    let pause98 = guard98.pauseMs(gpuMs: 100)
    expectClose(100.0 / (100.0 + pause98), 0.98, 1e-6,
                "guard: фактическая занятость GPU = 98%")
    expect(Params().gpuPowerPercent == 90, "Params по умолчанию = 90%")
}

/// `runGpuHeat` — диагностический прогрев, но он гоняет тот же 856×480
/// forward+backward, что и обучение, и поэтому не имеет права обходить потолок.
/// До исправления там стоял `ceiling: auto ? maxSafeDuty : 1.0`, то есть 98–100%.
func testHeatDutyCap() {
    section("Прогрев GPU: физический потолок без обходных путей")

    expectClose(clampedGpuDuty(1.0), safeDutyCap, 1e-9,
                "gpuheat: ручной режим 100% упирается в 98%")
    expectClose(clampedGpuDuty(0.99), safeDutyCap, 1e-9,
                "gpuheat: 99% упирается в 98%")
    expectClose(clampedGpuDuty(0.90), 0.90, 1e-9,
                "gpuheat: ровно 90% проходят как есть")
    expectClose(clampedGpuDuty(0.75), 0.75, 1e-9,
                "gpuheat: 75% проходят как есть")
    let autoDuty = clampedGpuDuty(-1)
    expect(autoDuty.isFinite && autoDuty >= 0.05 && autoDuty <= safeDutyCap,
           "gpuheat: авто-режим (duty ≤ 0) остаётся в коридоре")
    let nanDuty = clampedGpuDuty(Double.nan)
    expect(nanDuty.isFinite && nanDuty <= safeDutyCap,
           "gpuheat: NaN не превращается в полную нагрузку")

    // Сам регулятор с новым потолком не должен выходить за 98% ни при
    // каких показаниях датчика, включая «холодно» (там он разгоняется).
    var power = PowerController(ceiling: safeDutyCap, startDuty: 0.5)
    expectClose(power.ceiling, safeDutyCap, 1e-9,
                "PowerController: потолок = 98%")
    for temp in stride(from: 40.0, through: 100.0, by: 5.0) {
        let stats = GpuStats(temperatureC: temp, coreClockMHz: 400,
                             activityPercent: 90, powerW: 30)
        let d = power.update(stats: stats)
        expect(d <= safeDutyCap + 1e-9,
               "PowerController при \(Int(temp)) °C не выше 98%")
    }
    expect(power.duty <= safeDutyCap + 1e-9,
           "PowerController после 13 замеров не выше 98%")

    // Пауза при 98% обязана быть положительной: 2% времени уходят системе
    // и это не даёт повториться watchdog-перезагрузке.
    let guard90 = GpuDutyGuard(powerPercent: 100)
    expect(guard90.pauseMs(gpuMs: 150) > 0,
           "даже запрос 100% оставляет паузу на остывание")
}

// MARK: - Итерация 6: реальные кадры → слои → обучение под ключ

func testLayerAnnotation() {
    section("Разметка реальных кадров: точная факторизация 8 каналов")
    let w = 16, h = 16
    let n = w * h

    // Статичный фон + движущийся квадрат: классическое отделение объектов
    // от фона, но без единой строки на Python и без ML-модели.
    func backdrop() -> [Float] {
        var f = [Float](repeating: 0, count: n * 3)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 3
                let t = Float(y) / Float(h - 1)
                f[i] = 0.20 + 0.50 * t
                f[i + 1] = 0.30 + 0.40 * t
                f[i + 2] = 0.60 - 0.30 * t
            }
        }
        return f
    }
    func shot(_ sx: Int) -> [Float] {
        var f = backdrop()
        for y in 4..<8 {
            for x in sx..<(sx + 4) {
                let i = (y * w + x) * 3
                f[i] = 0.95; f[i + 1] = 0.10; f[i + 2] = 0.10
            }
        }
        return f
    }
    // Позиции объекта подобраны так, чтобы ни один пиксель не был закрыт
    // объектом более чем в 2 из 5 кадров: иначе временная медиана ПО ПРАВУ
    // становится объектом, и требовать от неё чистый фон было бы ошибкой
    // самого теста, а не алгоритма.
    let frames = [shot(0), shot(4), shot(8), shot(12), shot(2)]
    let params = LayerAnnotationParams(diffThreshold: 0.08, minLight: 0.70,
                                       blurRadius: 2, dilateRadius: 1)

    let bg = temporalMedianBackground(frames, w: w, h: h)
    expect(maxAbsDiff(bg, backdrop()) < 0.02,
           "временная медиана восстановила статичный фон")

    let scenes = annotateFrames(frames, w: w, h: h, params: params)
    expectEqual(scenes.count, frames.count, "по одной сцене на кадр")
    var totalMask = 0
    for i in 0..<scenes.count {
        let sc = scenes[i]
        let got = composeLayers(w: w, h: h, background: sc.background,
                                objects: sc.objects, mask: sc.mask, light: sc.light)
        expect(maxAbsDiff(got, frames[i]) < 1e-5,
               "кадр \(i): compose(слои) == исходный RGB")
        expect(sc.mask.allSatisfy { $0 == 0 || $0 == 1 },
               "кадр \(i): маска бинарна")
        totalMask += sc.mask.filter { $0 == 1 }.count
        let lo = params.minLight - 1e-6
        expect(sc.light.allSatisfy { $0 >= lo && $0 <= 1 + 1e-6 },
               "кадр \(i): свет в [minLight, 1]")
        var objectsClean = true
        for p in 0..<n where sc.mask[p] == 0 {
            let a = sc.objects[p * 3]
            let b = sc.objects[p * 3 + 1]
            let c = sc.objects[p * 3 + 2]
            if a != 0 || b != 0 || c != 0 { objectsClean = false; break }
        }
        expect(objectsClean, "кадр \(i): объекты нулевые вне маски")
    }
    expect(totalMask > 0, "движущийся объект обнаружен")

    let flat = [Float](repeating: 0.5, count: n)
    expect(maxAbsDiff(boxBlur(flat, w: w, h: h, radius: 3), flat) < 1e-6,
           "boxBlur константы не меняет её")
    let empty = differenceMask(frame: frames[0], background: frames[0],
                               w: w, h: h, threshold: 0.08, dilate: 1)
    expect(empty.allSatisfy { $0 == 0 }, "нулевая разница → пустая маска")

    let light = illuminationMap(frame: frames[0], w: w, h: h,
                                minLight: params.minLight, blurRadius: 2)
    var lightSafe = true
    for i in 0..<n {
        let r = frames[0][i * 3]
        let g = frames[0][i * 3 + 1]
        let b = frames[0][i * 3 + 2]
        let mx = max(r, max(g, b))
        if light[i] < mx - 1e-6 { lightSafe = false; break }
    }
    expect(lightSafe, "свет >= max канала: деление не выходит за [0,1]")

    let single = annotateFrames([frames[0]], w: w, h: h, params: params)
    expect(single[0].mask.allSatisfy { $0 == 0 },
           "один кадр → маска пуста")
    let one = composeLayers(w: w, h: h, background: single[0].background,
                            objects: single[0].objects, mask: single[0].mask,
                            light: single[0].light)
    expect(maxAbsDiff(one, frames[0]) < 1e-5,
           "один кадр: compose всё равно == RGB")
}

func testLayeredCapacityAndSchedule() {
    section("Ёмкость латента и расписание lr")
    let base: Float = 0.01
    let total = 100
    expectClose(Double(cosineLR(step: 0, totalSteps: total, baseLR: base)),
                Double(base), 1e-6, "lr на старте = baseLR")
    expectClose(Double(cosineLR(step: total, totalSteps: total, baseLR: base)),
                Double(base * 0.05), 1e-6, "lr в конце = 5% от baseLR")
    var prev = Double(base) + 1
    var monotonic = true
    for s in stride(from: 0, through: total, by: 10) {
        let v = Double(cosineLR(step: s, totalSteps: total, baseLR: base))
        if v > prev + 1e-9 { monotonic = false }
        prev = v
    }
    expect(monotonic, "lr монотонно убывает")
    expectClose(Double(cosineLR(step: total * 2, totalSteps: total, baseLR: base)),
                Double(base * 0.05), 1e-6, "шаг вне расписания не повышает lr")

    guard gHasMetal else { skip("Metal недоступен"); return }
    let engine = MetalEngine.shared
    let vae16 = ConvVAEGPU(engine: engine, channels: 8)
    let vae32 = ConvVAEGPU(engine: engine, channels: 8, latentChannels: 32)
    expectEqual(vae16.latentChannels, ConvVAEGPU.latentC, "по умолчанию C = 16")
    expectEqual(vae32.latentChannels, 32, "layered-путь: C = 32")
    expectEqual(vae32.e3.oc, 64, "e3 отдаёт 2·C каналов")
    expectEqual(vae32.d1.ic, 32, "d1 принимает C каналов")
    expectClose(ConvVAEGPU.compressionRatio(channels: 8, latentChannels: 32), 16, 1e-9,
                "сжатие 8 каналов при C=32 = 16×")

    let session = GpuSession(engine)
    let data = [Float](repeating: 0.5, count: 8 * 16 * 16)
    let x = GPUTensor(engine: engine, data: data, rows: 8)
    session.begin()
    let cache = vae32.forward(session, image: x, w: 16, h: 16, deterministic: true)
    _ = session.end()
    expectEqual(cache.z.rows, 32, "латент реально 32 канала")
    expectEqual(cache.recon.rows, 8, "выход по-прежнему 8 каналов")

    let b16 = convVAEBytesPerSample(w: 64, h: 48, channels: 8, layered: true)
    let b32 = convVAEBytesPerSample(w: 64, h: 48, channels: 8, layered: true,
                                    latentChannels: 32)
    expect(b32 > b16, "бюджет VRAM учитывает ёмкость латента")
    expectThrows("чекпоинт C=16 отвергнут моделью C=32") {
        try vae32.load(from: vae16.matrices)
    }
}

func testLayeredDatasetTraining() {
    section("Обучение на датасете сцен + чекпоинт")
    guard gHasMetal else { skip("Metal недоступен"); return }
    let scenes = [makeLayeredScene(w: 16, h: 16, seed: 101),
                  makeLayeredScene(w: 16, h: 16, seed: 202),
                  makeLayeredScene(w: 16, h: 16, seed: 303)]
    let ckpt = NSTemporaryDirectory() + "yume_layered_" + UUID().uuidString + ".safetensors"
    let report = trainLayeredVAE(steps: 30, w: 16, h: 16,
                                 scenes: scenes, latentChannels: 32,
                                 checkpointPath: ckpt)
    expectEqual(report.sceneCount, 3, "тренер увидел все 3 сцены")
    expect(report.lastLoss < report.firstLoss, "потеря по датасету упала")
    expect(report.psnr.isFinite && report.ssim.isFinite, "качество конечно")
    expectClose(Double(report.duty), 0.90, 1e-9, "потолок GPU = 90%")
    do {
        let mats = try loadState(path: ckpt)
        expectEqual(mats.count, 12, "чекпоинт: 12 матриц")
        expectEqual(mats[4].rows, 64, "чекпоинт: e3 = 2·C = 64")
        try? FileManager.default.removeItem(atPath: ckpt)
    } catch {
        expect(false, "чекпоинт не читается: \(error)")
    }
}

func testLayerPipeline() {
    section("Конвейер под ключ: кадры → слои → обучение → чекпоинт → PNG")
    let w = 16, h = 16
    let n = w * h
    func frame(_ sx: Int) -> [Float] {
        var f = [Float](repeating: 0, count: n * 3)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 3
                let t = Float(y) / Float(h - 1)
                f[i] = 0.2 + 0.5 * t
                f[i + 1] = 0.3 + 0.4 * t
                f[i + 2] = 0.6 - 0.3 * t
            }
        }
        for y in 4..<8 {
            for x in sx..<(sx + 4) {
                let i = (y * w + x) * 3
                f[i] = 0.9; f[i + 1] = 0.1; f[i + 2] = 0.1
            }
        }
        return f
    }
    let frames = [frame(0), frame(4), frame(8), frame(12)]

    // Моменты времени: чистая функция, проверяется без видеофайла.
    let times = sampleVideoTimes(count: 4, duration: 10)
    expectEqual(times.count, 4, "4 момента времени")
    expect(times[0] < times[1] && times[1] < times[2] && times[2] < times[3],
           "времена монотонно возрастают")
    expect(times[0] >= 0 && times[3] <= 10, "внутри длительности ролика")
    expectClose(sampleVideoTimes(count: 1, duration: 10, windowSeconds: 8,
                                 startSeconds: 0)[0], 4, 1e-9,
                "один кадр = середина заданного отрывка")
    // Ключевое для 30-минутного ролика: НЕ разбрасывать кадры по всему
    // фильму, иначе временная медиана фона превращается в кашу из сцен.
    let longVideo = sampleVideoTimes(count: 16, duration: 1800)
    expect(longVideo[15] - longVideo[0] <= 6.0001,
           "стек берётся из короткого непрерывного отрывка, а не из всего фильма")

    let params = LayerAnnotationParams(diffThreshold: 0.08, minLight: 0.70,
                                       blurRadius: 1, dilateRadius: 0)
    let ds = makeLayerDataset(frames: frames, w: w, h: h, params: params)
    expectEqual(ds.scenes.count, frames.count, "датасет: по сцене на кадр")
    expect(ds.compositeError < 1e-5, "датасет: обратная сборка точна")
    expect(ds.meanCoverage > 0 && ds.meanCoverage < 1,
           "датасет: маска не пустая и не сплошная")

    // Путь decodeImageRGB, который использует AVAssetImageGenerator:
    // проверяем через PNG в памяти, чтобы не тащить видеофайл в юнит-тесты.
    if let png = encodePNG(width: w, height: h, rgb: frames[0]),
       let back = loadImageRGB(data: png, w: w, h: h) {
        expect(maxAbsDiff(back, frames[0]) < 0.02,
               "PNG → decodeImageRGB сохраняет цвет кадра")
    } else {
        expect(false, "PNG → decodeImageRGB: декодирование не вернуло кадр")
    }

    guard gHasMetal else { skip("Metal недоступен"); return }
    let tmp = NSTemporaryDirectory() + "yume_pipe_" + UUID().uuidString
    let ckpt = tmp + ".safetensors"
    do {
        let r = try runLayerPipeline(frames: frames, w: w, h: h,
                                     frameCount: frames.count, steps: 12,
                                     latentChannels: 32,
                                     checkpointPath: ckpt,
                                     previewPrefix: tmp + "_preview")
        expectEqual(r.frameCount, frames.count, "pipeline: все кадры дошли")
        expectEqual(r.train.sceneCount, frames.count, "pipeline: тренер учился на всех сценах")
        expectEqual(r.train.latentChannels, 32, "pipeline: ёмкость латента C=32")
        expect(r.train.checkpointSaved, "pipeline: чекпоинт записан")
        expect(r.train.lastLoss < r.train.firstLoss, "pipeline: потеря упала")
        expectEqual(r.previewPaths.count, 7, "pipeline: 7 PNG-превью")
        expect(r.previewPaths.allSatisfy { FileManager.default.fileExists(atPath: $0) },
               "pipeline: превью реально на диске")
        expect(r.train.reconstructionHWC.count == n * 3,
               "pipeline: реконструкция той же размерности, что кадр")
        let mats = try loadState(path: ckpt)
        expectEqual(mats.count, 12, "pipeline: чекпоинт содержит 12 матриц")
        expectEqual(mats[4].rows, 64, "pipeline: e3 = 2·C = 64")
        for path in r.previewPaths { try? FileManager.default.removeItem(atPath: path) }
        try? FileManager.default.removeItem(atPath: ckpt)
    } catch {
        expect(false, "pipeline завершился ошибкой: \(error)")
    }
}

func testLayeredBatching() {
    section("Мини-батч: один command buffer на несколько сцен")
    guard gHasMetal else { skip("Metal недоступен"); return }
    // Корректная эквивалентность: ОДИНАКОВОЕ число обновлений Adam.
    // Батч 2 по [S, S] (сцена продублирована) с 20 обновлениями должен дать
    // ту же траекторию, что одиночный батч по [S] с теми же 20 обновлениями:
    // среднее из двух ТОЖДЕСТВЕННЫХ градиентов равно самому градиенту.
    // (Это изолирует проверку накопления/нормировки от разных расписаний лr:
    // сравнение «батч 4 за 8 шагов vs одиночный за 32» было бы некорректно.)
    let S = makeLayeredScene(w: 16, h: 16, seed: 99)
    let dupe = [S, S]
    let updates = 20
    let rBatch = trainLayeredVAE(steps: updates, w: 16, h: 16,
                                 scenes: dupe, latentChannels: 32,
                                 cosineSchedule: true, batchSize: 2)
    let rSingle = trainLayeredVAE(steps: updates, w: 16, h: 16,
                                  scenes: [S], latentChannels: 32,
                                  cosineSchedule: true, batchSize: 1)
    expectEqual(rBatch.sceneCount, 2, "батч-тренер видел обе сцены")
    expectEqual(rSingle.sceneCount, 1, "одиночный видел одну сцену")
    expect(rBatch.lastLoss < rBatch.firstLoss, "батч: потеря упала")
    expect(rSingle.lastLoss < rSingle.firstLoss, "одиночный: потеря упала")
    // Допуск не жёсткий: шаги батча берут разные сэмплы z (seed на forward),
    // так что дублированная пара не побитово равна одиночной, но статистика
    // одинаковых градиентов обязана сойтись.
    expectClose(rBatch.lastLoss, rSingle.lastLoss, 0.02,
                String(format: "батч 2 по дупликату == одиночный: %.5f vs %.5f",
                       rBatch.lastLoss, rSingle.lastLoss))
    expectClose(rBatch.duty, 0.90, 1e-9, "батч сохраняет потолок 90%")
    expect(rBatch.psnr.isFinite && rBatch.ssim.isFinite, "качество конечно")
}

func testLayeredOverlap() {
    section("Double-buffering CPU↔GPU: overlap даёт те же числа, что синхронный")
    guard gHasMetal else { skip("Metal недоступен"); return }
    // Overlap только перекрывает кодирование следующего шага со счётом
    // текущего на GPU. Математика не меняется: те же сцены, те же seed'ы
    // (step*batch+b), тот же порядок очереди command buffer'ов, те же
    // обновления Adam. Значит траектория обучения обязана совпасть — это
    // и есть доказательство, что перекрытие не ломает зависимости параметров.
    let S = makeLayeredScene(w: 24, h: 24, seed: 7)
    let steps = 40
    let sync = trainLayeredVAE(steps: steps, w: 24, h: 24,
                               scenes: [S], latentChannels: 32,
                               cosineSchedule: true, overlap: false)
    let ovl = trainLayeredVAE(steps: steps, w: 24, h: 24,
                              scenes: [S], latentChannels: 32,
                              cosineSchedule: true, overlap: true)
    expect(ovl.steps == steps, "overlap отработал заявленное число шагов")
    expectClose(ovl.firstLoss, sync.firstLoss, 1e-6,
                String(format: "firstLoss совпадает: %.6f vs %.6f",
                       ovl.firstLoss, sync.firstLoss))
    expectClose(ovl.lastLoss, sync.lastLoss, 1e-6,
                String(format: "lastLoss совпадает: %.6f vs %.6f",
                       ovl.lastLoss, sync.lastLoss))
    expectClose(Double(ovl.psnr), Double(sync.psnr), 1e-4,
                String(format: "PSNR совпадает: %.2f vs %.2f",
                       ovl.psnr, sync.psnr))
    expectClose(Double(ovl.ssim), Double(sync.ssim), 1e-5,
                String(format: "SSIM совпадает: %.5f vs %.5f",
                       ovl.ssim, sync.ssim))
}

func testDutyPacer() {
    section("Пейсер доли работы: пакетные паузы без потери 90%")
    let pacer = GpuDutyPacer(powerPercent: 90, maxPauseMs: 2000, minSleepMs: 2.0)
    expectClose(pacer.duty, 0.90, 1e-9, "пейсер наследует жёсткие 90%")
    // Короткий шаг: пауза копится в долг, но НЕ будит планировщик каждый раз.
    _ = pacer.wait(gpuMs: 1.0)
    expect(pacer.pendingPauseMs > 0, "короткий шаг копит долг паузы")
    expectEqual(pacer.sleepCalls, 0, "на коротком шаге сна ещё нет")
    // Накопим до порога и проверим, что суммарная доля работы всё та же.
    for _ in 0..<200 { _ = pacer.wait(gpuMs: 1.0) }
    _ = pacer.sleepDebt()
    expectEqual(pacer.pendingPauseMs, 0, "после сброса долга нет")
    expect(pacer.sleepCalls < 20,
           "201 шаг обошёлся \(pacer.sleepCalls) снами вместо 201")
    expectClose(pacer.measuredDuty, 0.90, 0.02,
                "фактическая доля работы осталась ~90%")
    expectEqual(pacer.emergency(tempC: 70), 0, "при 70 °C аварии нет")
    expectEqual(pacer.emergencyPauses, 0, "нормальная температура не считается аварией")
    // Отдельный пейсер с паузой 1 мс, чтобы тест не спал 2 секунды.
    let hot = GpuDutyPacer(powerPercent: 90, maxPauseMs: 1, minSleepMs: 2)
    expect(hot.emergency(tempC: emergencyTempC) > 0, "при 86 °C пауза применена сразу")
    expectEqual(hot.emergencyPauses, 1, "авария учтена один раз")
}

func testFrameStabilization() {
    section("Стабилизация камеры: сдвиг, обрезка, точная сборка")
    let w = 32, h = 32
    func base() -> [Float] {
        var f = [Float](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 3
                // НЕпериодический узор: на периодической «шахматке» сдвиг
                // на целый период даёт такой же ноль невязки, и поиск сдвига
                // становится неоднозначным. Хэш гарантирует единственный минимум.
                var q = UInt64(x) &* 73856093
                q ^= UInt64(y) &* 19349663
                q = q &* 2654435761
                let v = Float((q >> 24) & 255) / 255
                f[i] = v; f[i + 1] = v * 0.6; f[i + 2] = 0.3
            }
        }
        return f
    }
    let b = base()
    /// Кадр со сдвинутой «камерой»: пиксель (x,y) берёт базу из (x+s, y).
    func shifted(_ s: Int) -> [Float] {
        var f = [Float](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            for x in 0..<w {
                let sx = min(w - 1, max(0, x + s))
                let si = (y * w + sx) * 3
                let di = (y * w + x) * 3
                f[di] = b[si]; f[di + 1] = b[si + 1]; f[di + 2] = b[si + 2]
            }
        }
        return f
    }
    let frames = [shifted(-3), shifted(-1), shifted(0), shifted(2), shifted(4)]
    let params = LayerAnnotationParams(diffThreshold: 0.08, minLight: 0.70,
                                       blurRadius: 1, dilateRadius: 0)
    let naive = makeLayerDataset(frames: frames, w: w, h: h, params: params)
    let stab = makeLayerDataset(frames: frames, w: w, h: h, params: params,
                                stabilize: true, maxShift: 6)
    expect(naive.meanCoverage > 0.2,
           "без выравнивания сдвиг камеры раздувает маску")
    expect(stab.alignment != nil, "отчёт стабилизации заполнен")
    expectEqual(stab.alignment?.droppedFrames ?? -1, 0,
                "однородный отрывок: ни один кадр не выброшен")
    expect(stab.meanCoverage < naive.meanCoverage,
           String(format: "выравнивание снизило долю объектов: %.1f%% → %.1f%%",
                  naive.meanCoverage * 100, stab.meanCoverage * 100))
    expect(stab.w % 8 == 0 && stab.h % 8 == 0,
           "размер после обрезки кратен 8 (латент /8)")
    expect(stab.compositeError < 1e-5,
           "после перестановки пикселей сборка всё ещё точна")
    expectEqual(stab.scenes.count, frames.count, "все кадры остались в датасете")
}

// MARK: - Извлечение кадров: многопоточный ffmpeg в RAM

func testFrameExtractionPlan() {
    section("Извлечение кадров: план чанков, fps и JPEG 85%")

    let full = ParallelExtractor.makePlan(duration: 10, sourceFPS: 24,
                                          maxFrames: 0, cores: 4)
    expectClose(full.sourceFPS, 24, 1e-9, "fps берётся из исходного видео")
    expectEqual(full.allFrames, 240, "все кадры = длительность × fps")
    expectEqual(full.targetFrames, 240, "без лимита извлекаются все кадры")
    expectClose(full.outputFPS, 24, 1e-9, "выходной fps равен исходному")

    // Чанки идут подряд без дыр и перекрытий и покрывают ролик целиком.
    var nextFrame = 0
    for chunk in full.chunks {
        expectEqual(chunk.startFrame, nextFrame,
                    "чанк начинается там, где закончился предыдущий")
        expect(chunk.frameCount > 0, "чанк не пустой")
        nextFrame += chunk.frameCount
    }
    expectEqual(nextFrame, 240, "чанки покрывают весь ролик")

    // Первый чанк маленький: первые кадры видны в интерфейсе почти сразу,
    // а не после декодирования всего ролика.
    expectEqual(full.chunks.first?.frameCount, 60,
                "первый чанк — 60 кадров (быстрый старт)")
    // Дальше чанки растут до базового размера: на длинном ролике нельзя
    // запускать ffmpeg на каждые 60 кадров.
    expect((full.chunks.dropFirst().first?.frameCount ?? 0) > 60,
           "второй чанк крупнее первого")

    let limited = ParallelExtractor.makePlan(duration: 10, sourceFPS: 24,
                                             maxFrames: 100, cores: 4)
    expectEqual(limited.allFrames, 240, "лимит не меняет общее число кадров ролика")
    expectEqual(limited.targetFrames, 100, "лимит ограничивает число кадров")
    // Лимит — это ПЕРВЫЕ N кадров подряд, а не равномерная выборка по всему
    // ролику: частота остаётся исходной, иначе тест на 2000 кадров был бы
    // прореженным по всей длине, а не эмуляцией полной работы на начале.
    expectClose(limited.outputFPS, 24, 1e-9, "при лимите fps остаётся исходным")
    expectEqual(limited.chunks.map { $0.frameCount }.reduce(0, +), 100,
                "лимит делится по чанкам")
    if let last = limited.chunks.last {
        expectClose(limited.timeRange(of: last).end, 100.0 / 24.0, 1e-9,
                    "лимит занимает первые 100 кадров, а не весь ролик")
    } else {
        expect(false, "у лимита есть чанки")
    }

    // Времена чанков берутся из номеров кадров: стык двух чанков — ровно
    // кадр, а не «примерно секунда».
    if full.chunks.count >= 2 {
        let first = full.chunks[0], second = full.chunks[1]
        let r1 = full.timeRange(of: first)
        let r2 = full.timeRange(of: second)
        expectClose(r1.start, 0, 1e-9, "первый чанк начинается с нуля")
        expectClose(r1.end, 60.0 / 24.0, 1e-9, "конец = кадр / fps")
        expectClose(r2.start, r1.end, 1e-9, "следующий чанк начинается в стык")
    }

    // Длинный ролик (≈44 800 кадров, 10 ядер): чанков достаточно много, чтобы
    // рабочие не простаивали, но и не по одному на каждые 60 кадров.
    let long = ParallelExtractor.makePlan(duration: 1866.6, sourceFPS: 24.001,
                                          maxFrames: 0, cores: 10)
    expectEqual(long.targetFrames, long.allFrames, "длинный ролик: все кадры")
    expect(long.chunks.count >= 10,
           "чанков хватает на все ядра (\(long.chunks.count))")
    expect(long.chunks.count <= 120,
           "чанков не слишком много — запуск ffmpeg не съедает время (\(long.chunks.count))")
    expect(long.chunks.first?.frameCount == 60, "быстрый первый кадр на длинном ролике")
    expect(long.chunks.map { $0.frameCount }.reduce(0, +) == long.targetFrames,
           "длинный ролик покрыт без пропусков")
    let maxChunk = long.chunks.map { $0.frameCount }.max() ?? 0
    expect(maxChunk <= 1920, "размер чанка ограничен (память): \(maxChunk)")

    // Маленький ролик не дробится на бессмысленные куски.
    let tiny = ParallelExtractor.makePlan(duration: 1, sourceFPS: 10,
                                          maxFrames: 0, cores: 10)
    expectEqual(tiny.chunks.count, 1, "10 кадров — один чанк")

    expectEqual(ParallelExtractor.jpegQuality, 5,
                "JPEG ≈85% (ffmpeg q:v 5)")
}

/// Собирает эталон: один процесс ffmpeg, непрерывное декодирование.
private func referenceFrames(video: URL, fps: Double, count: Int,
                             dir: URL) -> [Data] {
    let ff = ParallelExtractor.ffmpegPath!
    let p = Process()
    p.executableURL = URL(fileURLWithPath: ff)
    p.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                   "-i", video.path,
                   "-vf", String(format: "fps=%.4f", fps),
                   "-frames:v", "\(count)",
                   "-f", "image2", "-vcodec", "mjpeg", "-q:v", "5",
                   "-vsync", "0", dir.appendingPathComponent("%05d.jpg").path]
    p.standardOutput = FileHandle.nullDevice
    p.standardError = FileHandle.nullDevice
    try? p.run()
    p.waitUntilExit()
    let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
    return files.sorted().compactMap { try? Data(contentsOf: dir.appendingPathComponent($0)) }
}

/// Стыки чанков не должны сдвигать кадры: байты каждого кадра обязаны
/// совпасть с непрерывным декодированием того же ролика.
func testChunkBoundariesAreExact() {
    section("Извлечение: чанки не сдвигают и не теряют кадры")

    guard let ffmpeg = ParallelExtractor.ffmpegPath else {
        skip("ffmpeg не установлен"); return
    }
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_boundary_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let video = root.appendingPathComponent("clip.mp4")

    let gen = Process()
    gen.executableURL = URL(fileURLWithPath: ffmpeg)
    gen.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc=size=96x64:rate=30:duration=20",
                     "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     video.path]
    gen.standardOutput = FileHandle.nullDevice
    gen.standardError = FileHandle.nullDevice
    do { try gen.run() } catch { skip("ffmpeg не запустился"); return }
    gen.waitUntilExit()
    guard gen.terminationStatus == 0 else {
        skip("ffmpeg не собрал тестовое видео"); return
    }

    // 20 с × 30 fps = 600 кадров, план даёт 4+ чанка (несколько стыков).
    func extract(maxFrames: Int) -> (frames: [Data], plan: ParallelExtractor.Plan) {
        let plan = ParallelExtractor.makePlan(duration: 20, sourceFPS: 30,
                                              maxFrames: maxFrames, cores: 10)
        var out: [Data] = []
        let sem = DispatchSemaphore(value: 0)
        let ex = FrameExtractor(url: video,
                                options: FrameExtractor.Options(maxFrames: maxFrames))
        ex.run(onFrame: { _, _, _, jpeg in out.append(jpeg) },
               progress: { _ in },
               completion: { _ in sem.signal() })
        _ = sem.wait(timeout: .now() + 180)
        return (out, plan)
    }

    let full = extract(maxFrames: 0)
    expect(full.plan.chunks.count >= 4,
           "тестовый ролик режется на несколько чанков (\(full.plan.chunks.count))")
    expectEqual(full.frames.count, 600, "получены все кадры ролика")

    let refDir = root.appendingPathComponent("ref_full")
    try? FileManager.default.createDirectory(at: refDir, withIntermediateDirectories: true)
    let ref = referenceFrames(video: video, fps: 30, count: 600, dir: refDir)
    expectEqual(ref.count, 600, "эталон собран")
    expectEqual(full.frames, ref,
                "кадры на стыках чанков байт-в-байт совпадают с эталоном")

    // Лимит кадров — это первые N кадров при исходном fps: они обязаны
    // байт-в-байт совпасть с началом полного извлечения. Раньше лимит
    // пересчитывал fps и прореживал ролик по всей длине — для теста на
    // 2000 кадров это была бы другая выборка, а не эмуляция полной работы.
    let limited = extract(maxFrames: 300)
    expectEqual(limited.frames.count, 300, "лимит: ровно 300 кадров")
    expectClose(limited.plan.outputFPS, 30, 1e-9, "лимит сохраняет исходный fps")
    expectEqual(limited.frames, Array(full.frames.prefix(300)),
                "лимит берёт первые 300 кадров подряд, без прореживания")
}

func testSoftwareFrameExtractionInMemory() {
    section("Извлечение кадров: ffmpeg параллельно, JPEG в оперативке")

    guard let ffmpeg = ParallelExtractor.ffmpegPath else {
        skip("ffmpeg не установлен"); return
    }
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_frame_test_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let video = dir.appendingPathComponent("clip.mp4")

    let gen = Process()
    gen.executableURL = URL(fileURLWithPath: ffmpeg)
    gen.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc=size=64x48:rate=20:duration=2",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", video.path]
    gen.standardOutput = FileHandle.nullDevice
    gen.standardError = FileHandle.nullDevice
    do { try gen.run() } catch { skip("ffmpeg не запустился"); return }
    gen.waitUntilExit()
    guard gen.terminationStatus == 0 else {
        skip("ffmpeg не собрал тестовое видео"); return
    }

    let extractor = FrameExtractor(url: video)
    expectEqual(extractor.estimatedCount(), 40, "оценка = 2 с × 20 fps")

    let sem = DispatchSemaphore(value: 0)
    var frames: [Data] = []
    var ordinals: [Int] = []
    var outcome: Result<Void, Error>?
    extractor.run(onFrame: { _, _, ordinal, jpeg in
        frames.append(jpeg)
        ordinals.append(ordinal)
    }, progress: { _ in }, completion: { r in
        outcome = r
        sem.signal()
    })
    _ = sem.wait(timeout: .now() + 60)

    switch outcome {
    case .success?:
        expect(frames.count >= 38 && frames.count <= 42,
               "кадров ≈ 40, получено \(frames.count)")
        expect(frames.allSatisfy { $0.count > 100 && $0.first == 0xFF && $0.dropFirst().first == 0xD8 },
               "каждый кадр — JPEG (FFD8) в памяти")
        expect(ordinals == Array(0..<frames.count), "порядок кадров строгий")
        // Тот же путь, которым обучается VAE: JPEG из памяти декодируется
        // сразу в размер обучения (80×56), исходный кадр остаётся полным.
        if let first = frames.first, let rgb = loadImageRGB(data: first, w: 80, h: 56) {
            expectEqual(rgb.count, 80 * 56 * 3,
                        "кадр декодируется в размер обучения 80×56")
            expect(rgb.allSatisfy { $0 >= 0 && $0 <= 1 },
                   "значения после декодирования в 80×56 нормализованы")
        } else {
            expect(false, "кадр не декодировался в 80×56")
        }
    case .failure(let e):
        expect(false, "извлечение упало: \(e)")
    case nil:
        expect(false, "извлечение не завершилось за 60 с")
    }
}

func testParallelRGBExtraction() {
    section("Извлечение RGB: программный декод параллельно")

    guard let ffmpeg = ParallelExtractor.ffmpegPath else {
        skip("ffmpeg не установлен"); return
    }
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_rgb_test_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let video = dir.appendingPathComponent("clip.mp4")

    let gen = Process()
    gen.executableURL = URL(fileURLWithPath: ffmpeg)
    gen.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", video.path]
    gen.standardOutput = FileHandle.nullDevice
    gen.standardError = FileHandle.nullDevice
    do { try gen.run() } catch { skip("ffmpeg не запустился"); return }
    gen.waitUntilExit()
    guard gen.terminationStatus == 0 else {
        skip("ffmpeg не собрал тестовое видео"); return
    }

    let w = 32, h = 24
    let times = [0.1, 0.5, 1.0, 1.5]
    let frames = try! FrameRGBExtractor.loadFrames(url: video, times: times,
                                                   w: w, h: h)
    expectEqual(frames.count, times.count, "кадров столько же, сколько моментов")
    expect(frames.allSatisfy { $0.count == w * h * 3 },
           "каждый кадр — HWC RGB нужного размера")
    expect(frames.allSatisfy { $0.allSatisfy { $0 >= 0 && $0 <= 1 } },
           "значения нормализованы в 0...1")
    expect(frames[0] != frames[3], "кадры из разных моментов различаются")

    // Тот же путь, что используют датасет и шоты (без синхронного моста).
    let datasetFrames = try! loadVideoFrames(url: video, w: 16, h: 16, count: 4)
    expectEqual(datasetFrames.count, 4, "датасет получает все кадры")
    expect(datasetFrames.allSatisfy { $0.count == 16 * 16 * 3 },
           "датасет получает кадры нужного размера")

    let shotFrames = try! loadShotFrames(url: video, w: 16, h: 16,
                                         start: 0.2, end: 1.8, count: 3)
    expectEqual(shotFrames.count, 3, "шот получает все кадры")
}

func testExtractionStreamsFramesAndProgress() {
    section("Извлечение: прогресс и кадры идут во время декодирования")

    guard let ffmpeg = ParallelExtractor.ffmpegPath else {
        skip("ffmpeg не установлен"); return
    }
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("yume_stream_test_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let video = dir.appendingPathComponent("clip.mp4")

    let gen = Process()
    gen.executableURL = URL(fileURLWithPath: ffmpeg)
    gen.arguments = ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc=size=480x270:rate=30:duration=10",
                     "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     video.path]
    gen.standardOutput = FileHandle.nullDevice
    gen.standardError = FileHandle.nullDevice
    do { try gen.run() } catch { skip("ffmpeg не запустился"); return }
    gen.waitUntilExit()
    guard gen.terminationStatus == 0 else {
        skip("ffmpeg не собрал тестовое видео"); return
    }

    let start = Date()
    var firstFrameAt: TimeInterval?
    var completionAt: TimeInterval?
    var sawIntermediate = false
    var frames = 0
    let sem = DispatchSemaphore(value: 0)
    let extractor = FrameExtractor(url: video,
                                   options: FrameExtractor.Options(maxFrames: 120))
    extractor.run(onFrame: { _, _, _, _ in
       frames += 1
        if firstFrameAt == nil { firstFrameAt = Date().timeIntervalSince(start) }
    }, progress: { p in
        if p > 0.02 && p < 0.98 { sawIntermediate = true }
    }, completion: { _ in
        completionAt = Date().timeIntervalSince(start)
        sem.signal()
    })
    _ = sem.wait(timeout: .now() + 120)

    expectEqual(frames, 120, "все 120 кадров получены")
    expect(sawIntermediate, "прогресс меняется до завершения (0 < p < 1)")
    if let first = firstFrameAt, let done = completionAt {
        expect(first < done, "первый кадр приходит раньше завершения")
    } else {
        expect(false, "извлечение не завершилось")
    }
}

// MARK: - Асинхронная выдача кадров в интерфейс

func testExtractBatchPolicy() {
    section("Извлечение: пачки кадров и первая страница коллажа")
    let policy = ExtractBatchPolicy()

    // Первая пачка — ровно страница коллажа (10 кадров), чтобы первые
    // кадры появились сразу, а не после всех 44 000.
    expect(policy.shouldFlush(pending: 9, sinceLastFlushMs: 0,
                              flushedFirstBatch: false) == false,
           "9 кадров меньше первой страницы — не сбрасываем")
    expect(policy.shouldFlush(pending: 10, sinceLastFlushMs: 0,
                              flushedFirstBatch: false),
           "10 кадров — первая пачка готова")

    // Дальше крупные пачки: main thread не дёргается на каждый кадр.
    expect(policy.shouldFlush(pending: 63, sinceLastFlushMs: 0,
                              flushedFirstBatch: true) == false,
           "63 кадра — ещё не полная пачка")
    expect(policy.shouldFlush(pending: 64, sinceLastFlushMs: 0,
                              flushedFirstBatch: true),
           "64 кадра — пачка готова")

    // Даже неполная пачка уходит по таймеру, иначе последние кадры
    // висят в буфере до конца извлечения.
    expect(policy.shouldFlush(pending: 1, sinceLastFlushMs: 250,
                              flushedFirstBatch: true),
           "неполная пачка уходит по времени")
    expect(policy.shouldFlush(pending: 1, sinceLastFlushMs: 50,
                              flushedFirstBatch: true) == false,
           "неполная пачка ждёт таймер")
    expect(policy.shouldFlush(pending: 0, sinceLastFlushMs: 999,
                              flushedFirstBatch: true) == false,
           "пустую пачку не сбрасываем")

    // Коллаж перерисовывается только на первой странице и только пока
    // она заполняется; остальные кадры — по пагинации.
    expect(policy.shouldRenderCollage(page: 0, totalBefore: 0, totalAfter: 10,
                                      pageSize: 10),
           "первая страница заполнилась — рендер")
    expect(policy.shouldRenderCollage(page: 0, totalBefore: 5, totalAfter: 10,
                                      pageSize: 10),
           "кадры первой страницы появляются по мере поступления")
    expect(policy.shouldRenderCollage(page: 0, totalBefore: 10, totalAfter: 74,
                                      pageSize: 10) == false,
           "после 10 кадров коллаж не дёргаем")
    expect(policy.shouldRenderCollage(page: 2, totalBefore: 10, totalAfter: 74,
                                      pageSize: 10) == false,
           "пока пользователь листает, коллаж не перерисовывается")
    expect(policy.shouldRenderCollage(page: 0, totalBefore: 10, totalAfter: 10,
                                      pageSize: 10) == false,
           "без новых кадров рендера нет")
}

// MARK: - Проверка (тест) текущих кадров в памяти

func testFrameValidation() {
    section("Проверка кадров: лимит и целостность")

    let ok = Data([0xFF, 0xD8, 0x00, 0x01, 0xFF, 0xD9])
    let frames = (0..<5).map { i -> MemoryFrame in
        MemoryFrame(name: String(format: "frame_%04d.jpg", i), jpeg: ok)
    }

    // Все кадры валидны, лимит 0 = проверить всё.
    let all = frameValidation(frames: frames, limit: 0, decode: { _ in true })
    expectEqual(all.checked, 5, "проверены все кадры")
    expectEqual(all.passed, 5, "все прошли")
    expectEqual(all.failed, 0, "провалов нет")
    expectEqual(all.issues.count, 0, "проблем нет")

    // Лимит ограничивает число проверяемых кадров.
    let limited = frameValidation(frames: frames, limit: 2, decode: { _ in true })
    expectEqual(limited.checked, 2, "лимит 2: проверены 2 кадра")
    expectEqual(limited.passed, 2, "оба прошли")
    expectEqual(limited.failed, 0, "провалов нет")

    // Лимит больше числа кадров — проверяются все.
    let over = frameValidation(frames: frames, limit: 999, decode: { _ in true })
    expectEqual(over.checked, 5, "лимит больше массива — все кадры")

    // Пустой массив — ничего не проверяем.
    let empty = frameValidation(frames: [], limit: 0, decode: { _ in true })
    expectEqual(empty.checked, 0, "пустой массив: 0 проверок")
    expectEqual(empty.failed, 0, "пустой массив: 0 провалов")

    // Битая сигнатура JPEG ловится.
    var broken = frames
    broken[2] = MemoryFrame(name: "frame_0002.jpg", jpeg: Data([0x00, 0x01, 0x02]))
    let b = frameValidation(frames: broken, limit: 0, decode: { _ in true })
    expectEqual(b.failed, 1, "битая сигнатура — 1 провал")
    expect(b.issues.contains { $0.contains("JPEG") },
           "причина указана: JPEG-сигнатура")

    // Сигнатура есть, но декодирование не проходит.
    let undecodable = frameValidation(frames: frames, limit: 0,
                                      decode: { _ in false })
    expectEqual(undecodable.failed, 5, "все не декодируются")
    expect(undecodable.issues.first?.contains("декод") == true,
           "причина указана: декодирование")
}

// MARK: - Математика и защита от краевых входов (ревью 2026-11)

func testMathAndGuards() {
    section("Математика: smoothstep, метрики кадра, нормализация")

    // smoothstep: обычный порядок краёв (монотонный рост).
    expect(smoothstep(0, 1, 0) == 0, "smoothstep: левый край = 0")
    expect(smoothstep(0, 1, 1) == 1, "smoothstep: правый край = 1")
    expect(smoothstep(0, 1, 0.5) == 0.5, "smoothstep: центр = 0.5")
    expect(smoothstep(0, 1, -1) == 0, "smoothstep: кламп слева")
    expect(smoothstep(0, 1, 2) == 1, "smoothstep: кламп справа")

    // Обращённые края (e0 > e1) — так рендерер зовёт солнце:
    // smoothstep(sunR, sunR*0.6, d). Раньше знаменатель клампился как
    // max(e1-e0, 1e-6): обращённый диапазон инвертировался — диск солнца
    // не закрашивался, а всё небо вокруг заливалось солнечным цветом.
    expect(smoothstep(1, 0.5, 1) == 0, "smoothstep (обратный): t=e0 → 0")
    expect(smoothstep(1, 0.5, 0.5) == 1, "smoothstep (обратный): t=e1 → 1")
    expect(smoothstep(1, 0.5, 2) == 0, "smoothstep (обратный): кламп справа")
    expect(smoothstep(1, 0.5, 0) == 1, "smoothstep (обратный): кламп слева")
    // Вырожденные края — без деления на ноль.
    expect(smoothstep(0.5, 0.5, 0.4) == 0, "smoothstep: e0==e1, t меньше")
    expect(smoothstep(0.5, 0.5, 0.6) == 1, "smoothstep: e0==e1, t больше")

    // Метрики кадра (FrameMetrics).
    expect(lumaAt([1, 1, 1], 0) == 1, "lumaAt: белый пиксель = 1")
    expect(abs(lumaAt([1, 0, 0], 0) - 0.299) < 1e-6, "lumaAt: красный = 0.299")
    expect(maxPixelDelta([0, 1, 0.5], [1, 1, 0.5]) == 1, "maxPixelDelta: максимум разности")
    expect(maxPixelDelta([0], [0, 1, 2]) == 0, "maxPixelDelta: короткий кадр безопасен")

    // rgbToFloats: короткий буфер не читает за границей (остаток — нули).
    let short = FrameRGBExtractor.rgbToFloats(Data([255, 128, 0]), pixels: 2)
    expect(short[0] == 1 && abs(short[1] - 128.0 / 255) < 1e-6 && short[2] == 0,
           "rgbToFloats: доступные байты нормализованы")
    expect(short[3] == 0 && short[4] == 0 && short[5] == 0,
           "rgbToFloats: недостающие пиксели — нули (без крэша)")

    // padToMultipleOf8: продление края (не растяжение) до кратного 8.
    // Вход — RGB: на пиксель три значения, иначе функция читает за границей.
    let (padded, pw, ph) = padToMultipleOf8([Float](repeating: 0.5, count: 3 * 3 * 3), w: 3, h: 3)
    expectEqual(pw, 8, "pad: ширина до кратного 8")
    expectEqual(ph, 8, "pad: высота до кратного 8")
    expectEqual(padded.count, 8 * 8 * 3, "pad: размер кадра")
    expect(padded.allSatisfy { $0 == 0.5 }, "pad: край продлён значением края")
}

// MARK: - Планировщик чанков и бюджеты памяти

func testChunkPlanAndMemoryBudgets() {
    section("Планировщик чанков ffmpeg и бюджеты памяти")

    // План чанков: покрывает [0, target) без дыр, растёт 60→120→240→1920.
    do {
        let plan = ParallelExtractor.chunkPlan(target: 500)
        expectEqual(plan.map { $0.frameCount }, [60, 120, 240, 80],
                    "чанки: рост 60→120→240, хвост неполный")
        expectEqual(plan.map { $0.startFrame }, [0, 60, 180, 420],
                    "чанки: стыки без дыр и перекрытий")
        expectEqual(plan.reduce(0) { $0 + $1.frameCount }, 500, "чанки: сумма = target")

        let one = ParallelExtractor.chunkPlan(target: 50)
        expectEqual(one.count, 1, "чанки: маленький ролик — один чанк")
        expect(ParallelExtractor.chunkPlan(target: 0).isEmpty, "чанки: target=0 → пусто")

        let big = ParallelExtractor.chunkPlan(target: 100_000)
        expect(big.allSatisfy { $0.frameCount <= 1920 }, "чанки: потолок размера 1920")
        expectEqual(big.reduce(0) { $0 + $1.frameCount }, 100_000,
                    "чанки: 100k покрывается целиком")
    }

    // Порция дозагрузки кадров в VRAM: [64, 1024], от бюджетов RAM/GPU.
    do {
        let fb = 854 * 480 * 3 * 4
        expectEqual(frameLoadChunkSize(frameBytes: 0, availableRAM: 32 << 30,
                                       maxWorkingSet: 8 << 30),
                    64, "порция: frameBytes=0 → консервативные 64")
        expectEqual(frameLoadChunkSize(frameBytes: fb, availableRAM: 64 << 30,
                                       maxWorkingSet: 64 << 30),
                    1024, "порция: огромные бюджеты → потолок 1024")
        // Нижняя граница — когда бюджета не хватает даже на пару кадров:
        // 50% от 64 МБ = 32 МБ; 32 МБ / (2 × 4.9 МБ) ≈ 3 → зажим до 64.
        expectEqual(frameLoadChunkSize(frameBytes: fb, availableRAM: 64 << 30,
                                       maxWorkingSet: 64 << 20),
                    64, "порция: малый GPU-бюджет → нижняя граница 64")
        // 25% RAM против 50% рабочего набора: берётся минимум.
        // 32 ГБ RAM → 8 ГиБ; 8 ГБ GPU → 4 ГиБ; 4 ГиБ / (2 × 4.9 МБ) = 436.
        expectEqual(frameLoadChunkSize(frameBytes: fb, availableRAM: 32 << 30,
                                       maxWorkingSet: 8 << 30),
                    436, "порция: 32 ГБ RAM / 8 ГБ GPU → 436 кадров")
    }
}

// MARK: - Рендер: солнце (регрессия инвертированного smoothstep)

func testSunDiscRegression() {
    section("Рендер: солнце — компактный диск, а не всё небо")

    let sc = makeLayeredScene(w: 160, h: 120, seed: 2026)
    let n = sc.w * sc.h
    var warm = 0
    // Солнце — слой группы .background: в sc.objects его нет (там только
    // фигуры), поэтому ищем тёплый диск именно в фоне. Раньше проверка
    // смотрела в objects и не могла увидеть регрессию вообще.
    for i in 0..<n {
        let r = sc.background[i * 3], g = sc.background[i * 3 + 1], b = sc.background[i * 3 + 2]
        // Солнце: (1.0, 0.97, 0.88). Облака (0.98, 0.99, 1.0) отсекаются
        // синим каналом, небо — тоже (синий канал высокий).
        if r > 0.95 && g > 0.92 && b < 0.95 { warm += 1 }
    }
    let frac = Double(warm) / Double(n)
    expect(warm >= 5, "солнце: тёплые пиксели в фоне есть (\(warm))")
    expect(frac < 0.15,
           "солнце: компактный диск — \(String(format: "%.3f", frac)) кадра (< 0.15); раньше инвертированный smoothstep заливал солнечным цветом всё небо")
}

// MARK: - Стабилизация: крошечные кадры (регрессия выхода за границы)

func testStabilizeTinyFrames() {
    section("Стабилизация: крошечные кадры не ломают границы выборки")

    // 12×12: раньше availW=6, cw=max(8, 0)=8 > availW → offX<0 → sx выходил
    // за [0, w) и чтение шло за границей буфера (крэш). Теперь сдвиг зажат
    // запасом на кратность 8.
    do {
        var rng = SplitMix64(seed: 3)
        let frames = (0..<5).map { _ in makeSyntheticScene(w: 12, h: 12, rng: &rng) }
        let (aligned, cw, ch, report) = stabilizeFrames(frames, w: 12, h: 12, maxShift: 3)
        expect(cw % 8 == 0 && ch % 8 == 0, "12×12: выход кратен 8 (\(cw)×\(ch))")
        expect(cw <= 12 && ch <= 12, "12×12: выход не больше исходного кадра")
        expect(aligned.allSatisfy { $0.count == cw * ch * 3 },
               "12×12: размер выровненных кадров сходится")
        expect(!aligned.isEmpty, "12×12: хотя бы один кадр остался")
        expectEqual(report.sourceWidth, 12, "12×12: отчёт ссылается на источник")
    }

    // 16×16: поведение прежнее (умещается и сдвиг, и кратность 8).
    do {
        var rng = SplitMix64(seed: 7)
        let frames = (0..<4).map { _ in makeSyntheticScene(w: 16, h: 16, rng: &rng) }
        let (aligned, cw, ch, _) = stabilizeFrames(frames, w: 16, h: 16, maxShift: 8)
        expectEqual(cw, 8, "16×16: выход 8×8 (сдвиг ±4, область 8)")
        expectEqual(ch, 8, "16×16: выход 8×8")
        expect(!aligned.isEmpty, "16×16: кадры остались")
    }

    // 8×8: на сдвиг места нет вовсе — честный ранний выход без изменений.
    do {
        var rng = SplitMix64(seed: 11)
        let frames = (0..<3).map { _ in makeSyntheticScene(w: 8, h: 8, rng: &rng) }
        let (aligned, cw, ch, _) = stabilizeFrames(frames, w: 8, h: 8, maxShift: 8)
        expectEqual(aligned.count, 3, "8×8: кадры без изменений")
        expectEqual(cw, 8, "8×8: размер без изменений")
        expectEqual(ch, 8, "8×8: размер без изменений")
    }
}

// MARK: - GPUFramePool: битый кадр (регрессия «фантома»)

func testChannelLayout() {
    section("Канальная раскладка: HWC ↔ CHW")

    // 1. hwcToChw ставит каналы в планарный (CHW) порядок.
    do {
        // Кадр 2×1: пиксель 0 = (R0,G0,B0) = (1,2,3), пиксель 1 = (R1,G1,B1).
        let hwc: [Float] = [1, 2, 3, 4, 5, 6]
        let chw = hwcToChw(hwc)
        // CHW: R-плоскость [1,4], G [2,5], B [3,6].
        expectEqual(chw, [1, 4, 2, 5, 3, 6], "hwcToChw: планарная раскладка")
    }

    // 2. Круговая поездка — тождество.
    do {
        var rng = SplitMix64(seed: 99)
        let img = makeSyntheticScene(w: 16, h: 12, rng: &rng)
        expectEqual(maxAbsDiff(chwToHwc(hwcToChw(img)), img), 0,
                    "chwToHwc∘hwcToChw = тождество")
        expectEqual(maxAbsDiff(hwcToChw(chwToHwc(img)), img), 0,
                    "hwcToChw∘chwToHwc = тождество")
    }

    // 3. Поведенческий: с правильной CHW-раскладкой и малым β модель учится
    //    сохранять яркость кадра. В HWC-кормушке (каналы перепутаны) этого
    //    не происходило — реконструкция схлопывалась в тёмный шум.
    guard gHasMetal else { skip("Metal недоступен"); return }
    do {
        let engine = MetalEngine.shared
        let session = GpuSession(engine)
        let w = 32, h = 24
        var rng = SplitMix64(seed: 7)
        let hwc = makeSyntheticScene(w: w, h: h, rng: &rng)
        let chw = hwcToChw(hwc)
        let x = GPUTensor(engine: engine, data: chw, rows: 3)
        let vae = ConvVAEGPU(engine: engine)
        let beta: Float = 0.01
        let steps = 200
        for step in 1...steps {
            session.begin()
            let c = vae.forward(session, image: x, w: w, h: h,
                                deterministic: false, seed: UInt32(step))
            let (_, g) = vae.backward(session, c, target: x, beta: beta)
            vae.applyAdam(session, grads: g, t: step, lr: 0.01)
            _ = session.end()
        }
        func lumaMean(_ hwc: [Float]) -> Double {
            let n = hwc.count / 3
            var s = 0.0
            for i in 0..<n {
                s += 0.299 * Double(hwc[i * 3])
                    + 0.587 * Double(hwc[i * 3 + 1])
                    + 0.114 * Double(hwc[i * 3 + 2])
            }
            return s / Double(n)
        }
        session.begin()
        let c = vae.forward(session, image: x, w: w, h: h, deterministic: true)
        _ = session.end()
        let reconHWC = chwToHwc(c.recon.toArray())
        let targetLuma = lumaMean(hwc)
        let reconLuma = lumaMean(reconHWC)
        expect(abs(reconLuma - targetLuma) < 0.30,
               String(format: "CHW-кормушка: яркость реконструкции близка к исходной (%.2f vs %.2f)",
                      reconLuma, targetLuma))
    }
}

// MARK: - GPUFramePool: битый кадр (регрессия «фантома»)

func testGPUFramePoolBrokenFrame() {
    section("GPUFramePool: битый кадр → чёрная заглушка, не мусор")

    guard MTLCreateSystemDefaultDevice() != nil else {
        skip("Metal недоступен"); return
    }
    let engine = MetalEngine.shared

    func solidFrame(_ v: Float) -> MemoryFrame {
        let img = [Float](repeating: v, count: 8 * 8 * 3)
        let png = encodePNG(width: 8, height: 8, rgb: img)!
        return MemoryFrame(name: "f\(v)", jpeg: png)
    }
    // Кадр 1 — мусор вместо JPEG: раньше слот пула оставался с байтами
    // предыдущего окна («фантом»), а непроинициализированный приватный
    // буфер вообще содержал случайные значения.
    let frames = [solidFrame(0.1),
                  MemoryFrame(name: "bad", jpeg: Data([0x12, 0x34, 0x56])),
                  solidFrame(0.9)]
    guard let pool = GPUFramePool(engine: engine, frames: frames, w: 8, h: 8, size: 2) else {
        expect(false, "пул не создался"); return
    }

    let a0 = pool.next().toArray()
    expect(a0.allSatisfy { abs($0 - 0.1) < 0.02 },
           "пул: валидный кадр декодирован (значение ~0.1)")

    let a1 = pool.next().toArray()
    expect(a1.allSatisfy { $0 == 0 },
           "пул: битый кадр — чёрная заглушка (нули), не мусор и не чужой кадр")
}

// MARK: - main

func runAll() {
    let tests: [(String, () -> Void)] = [
        ("HTTP", testHTTP),
        ("TensorBufferPool", testTensorBufferPool),
        ("TensorStagingPool", testTensorStagingPool),
        ("GradSanitize", testGradSanitizeBatched),
        ("GpuWaitPolicy", testGpuWaitPolicy),
        ("TrainingControl", testTrainingControlLifecycle),
        ("ParamAPI", testParamAPI),
        ("Checkpoints", testCheckpoints),
        ("Backups", testBackups),
        ("Backups", testPrepareCheckpointForTraining),
        ("GPUOps", testGPUOps),
        ("ConvPrimitives", testConvPrimitives),
        ("BufferLifecycle", testBufferLifecycle),
        ("ConvVAEShape", testConvVAEShape),
        ("VAEPrimitives", testVAEPrimitives),
        ("VAEVariational", testVAEVariational),
        ("VAEGradients", testVAEGradients),
        ("VAEConvergence", testVAEConvergence),
        ("VRAMBudget", testVRAMBudget),
        ("ProcessedMarks", testProcessedMarks),
        ("Reproducibility", testReproducibility),
        ("CheckpointCompat", testCheckpointCompat),
        ("TrainStepLogging", testTrainStepLogging),
        ("ThermalGovernor", testThermalGovernor),
        ("PowerController", testPowerController),
        ("PureLogic", testPureLogic),
        ("LayerRenderer", testLayerRenderer),
        ("UniversalLayers", testUniversalLayers),
        ("SceneQuality", testSceneQuality),
        ("LayeredLossMath", testLayeredLossMath),
        ("LayeredLossGPU", testLayeredLossGPU),
        ("LayeredVAEShape", testLayeredVAEShape),
        ("LayeredVAEGradients", testLayeredVAEGradients),
        ("LayeredTrainer", testLayeredTrainer),
        ("DutyCap", testDutyCap),
        ("HeatDutyCap", testHeatDutyCap),
        ("LayerAnnotation", testLayerAnnotation),
        ("LayeredCapacityAndSchedule", testLayeredCapacityAndSchedule),
        ("LayeredDatasetTraining", testLayeredDatasetTraining),
        ("LayerPipeline", testLayerPipeline),
        ("DutyPacer", testDutyPacer),
        ("LayeredBatching", testLayeredBatching),
        ("LayeredOverlap", testLayeredOverlap),
        ("FrameStabilization", testFrameStabilization),
        ("FrameExtraction", testFrameExtractionPlan),
        ("FrameExtraction", testChunkBoundariesAreExact),
        ("FrameExtraction", testSoftwareFrameExtractionInMemory),
        ("FrameExtraction", testParallelRGBExtraction),
        ("FrameExtraction", testExtractionStreamsFramesAndProgress),
        ("ExtractBatch", testExtractBatchPolicy),
        ("FrameValidation", testFrameValidation),
        ("MathGuards", testMathAndGuards),
        ("ChunkPlan", testChunkPlanAndMemoryBudgets),
        ("SunDisc", testSunDiscRegression),
        ("StabilizeTiny", testStabilizeTinyFrames),
        ("ChannelLayout", testChannelLayout),
        ("PoolBrokenFrame", testGPUFramePoolBrokenFrame),
    ]
    if let filter = gFilter {
        print("Фильтр: \(filter)")
    }
    for (name, body) in tests {
        if let filter = gFilter,
           !name.localizedCaseInsensitiveContains(filter) {
            continue
        }
        body()
    }
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
