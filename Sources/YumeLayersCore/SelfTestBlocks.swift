// YumeLayers — блоки самопроверки (декомпозиция runSelfTest).
// Каждая функция — независимый набор проверок; вызывается из runSelfTest.
import Foundation
import CoreGraphics
import ImageIO

func selftestDeliveryPlanner(_ expect: (Bool, String) -> Void) {
    // 1. пока пропуск чёрных неизвестен — ничего не отдаём
    var pl = FrameDeliveryPlanner()
    expect(pl.emittableRanges(chunkDone: [true, true]).isEmpty, "пока skip неизвестен — не отдаём")

    // 2. первый чанк готов → отдаём первый
    pl.learnSkipIfNeeded(leadingBlacks: 0)
    let r1 = pl.emittableRanges(chunkDone: [true, false, false])
    expect(r1.count == 1 && r1[0] == 0..<1, "готов только чанк 0 → отдаём его")

    // 3. строгий порядок: чанк 1 готов, но 0 нет → ничего
    var pl2 = FrameDeliveryPlanner(); pl2.learnSkipIfNeeded(leadingBlacks: 0)
    expect(pl2.emittableRanges(chunkDone: [false, true]).isEmpty,
           "чанк 0 не готов → не перепрыгиваем через него")

    // 4. подряд готовы 0 и 1 → отдаём оба разом
    var pl3 = FrameDeliveryPlanner(); pl3.learnSkipIfNeeded(leadingBlacks: 0)
    let r3 = pl3.emittableRanges(chunkDone: [true, true, false])
    expect(r3.count == 1 && r3[0] == 0..<2, "подряд готовы 0..1 → оба вместе")

    // 5. продвижение: после отдачи следующий чанк становится доступен
    pl3.markEmittedChunks(2)
    let r4 = pl3.emittableRanges(chunkDone: [true, true, true])
    expect(r4.count == 1 && r4[0] == 2..<3, "после продвижения отдаём следующий")

    // 6. фиксация пропуска лидирующих чёрных
    var pl4 = FrameDeliveryPlanner(); pl4.learnSkipIfNeeded(leadingBlacks: 2)
    expect(pl4.skip == 2, "пропуск лидирующих чёрных зафиксирован")

    // 7. правило обновления коллажа: первые 3 сразу, далее каждые 200
    expect(shouldRefreshCollage(atFrame: 1) && shouldRefreshCollage(atFrame: 2)
           && shouldRefreshCollage(atFrame: 3) && shouldRefreshCollage(atFrame: 200)
           && shouldRefreshCollage(atFrame: 400), "обновление на 1/2/3/200/400")
    expect(!shouldRefreshCollage(atFrame: 201) && !shouldRefreshCollage(atFrame: 199),
           "без обновления на 201/199")


}

func selftestBatchesAndClips(_ expect: (Bool, String) -> Void) {
    // 8. разбиение на батчи
    expect(batchRanges(total: 10, batch: 4) == [0..<4, 4..<8, 8..<10], "батчи 4/4/2")
    expect(batchRanges(total: 0, batch: 4).isEmpty, "пусто при total=0")
    expect(batchRanges(total: 3, batch: 8) == [0..<3], "один батч целиком")


    // 9. клипы: метка источника и строгий порядок сохраняются
    let frames = [MemoryFrame(name: "frame_0000.jpg", jpeg: Data([1]), source: "a.mp4"),
                  MemoryFrame(name: "frame_0001.jpg", jpeg: Data([2]), source: "a.mp4"),
                  MemoryFrame(name: "frame_0000.jpg", jpeg: Data([3]), source: "b.mp4")]
    expect(frames.count == 3, "клип: кадры из источников")
    expect(frames.allSatisfy { !$0.source.isEmpty }, "клип: у каждого кадра есть источник")
    let clipA = frames.filter { $0.source == "a.mp4" }
    expect(clipA.count == 2 && clipA[0].name == "frame_0000.jpg" && clipA[1].name == "frame_0001.jpg",
           "клип: порядок внутри источника сохранён")


}

func selftestGpuMseAndControl(_ expect: (Bool, String) -> Void) {
    // 10. GPU-MSE совпадает с CPU-расчётом
    do {
        let engine = MetalEngine.shared
        let session = GpuSession(engine)
        let n = 200_000
        var a = [Float](repeating: 0, count: n)
        var b = [Float](repeating: 0, count: n)
        for i in 0..<n {
            a[i] = sin(Float(i) * 0.001)
            b[i] = cos(Float(i) * 0.001)
        }
        let tA = GPUTensor(engine: engine, data: a, rows: 1)
        let tB = GPUTensor(engine: engine, data: b, rows: 1)
        session.begin()
        let buf = session.encodeMSE(recon: tA, target: tB)
        _ = session.end()
        let gpuMse = Double(buf.contents().assumingMemoryBound(to: Float.self)[0]) / Double(n)
        var cpuSum = 0.0
        for i in 0..<n {
            let d = Double(a[i] - b[i])
            cpuSum += d * d
        }
        let cpuMse = cpuSum / Double(n)
        // MSE единицы: sin/cos дают ~0.5±; допустимая погрешность 1e-3
        expect(abs(gpuMse - cpuMse) < 1e-3, "GPU-MSE (\(gpuMse)) ≈ CPU (\(cpuMse))")
    }

    // 11. контрольная выборка: 1 кадр на секунду видео (по fps)
    expect(isControlFrame(index: 0, fps: 24), "выборка: кадр 0 (24 fps) всегда контрольный")
    expect(isControlFrame(index: 24, fps: 24), "выборка: кадр 24 (24 fps) — 1-я секунда")
    expect(!isControlFrame(index: 1, fps: 24), "выборка: кадр 1 (24 fps) — не контрольный")
    expect(!isControlFrame(index: 23, fps: 24), "выборка: кадр 23 (24 fps) — не контрольный")
    expect(isControlFrame(index: 30, fps: 30), "выборка: кадр 30 (30 fps) — 1-я секунда")
    expect(!isControlFrame(index: 29, fps: 30), "выборка: кадр 29 (30 fps) — не контрольный")
    expect(!isControlFrame(index: 24, fps: 30), "выборка: кадр 24 (30 fps) — не контрольный")
    expect(isControlFrame(index: 60, fps: 30), "выборка: кадр 60 (30 fps) — 2-я секунда")
    expect(isControlFrame(index: 0, fps: 0), "выборка: fps=0 → шаг 1 (безопасно)")

    // 12. контрольная выборка не пересекается с освобождением основного потока
    // (каждый контрольный кадр существует и в общем наборе, но живёт отдельно).
    let allNames = (0..<100).map { String(format: "frame_%04d.jpg", $0) }
    let controlIdx = (0..<100).filter { isControlFrame(index: $0, fps: 24) }
    let expectedControls = stride(from: 0, to: 100, by: 24).map { $0 }
    expect(controlIdx == expectedControls, "выборка: индексы 0,24,48,72,96 (24 fps)")
    expect(controlIdx.allSatisfy { allNames.indices.contains($0) }, "выборка: каждый контрольный кадр есть в общем наборе")
    expect(controlIdx.count == 5, "выборка: для 100 кадров (24 fps) — 5 контрольных")

}

func selftestJpegReleaseAndCheckpoint(_ expect: (Bool, String) -> Void) {
    // 13. освобождение JPEG из RAM после загрузки в VRAM
    var poolFrames = (0..<64).map { i -> MemoryFrame in
        MemoryFrame(name: String(format: "frame_%04d.jpg", i),
                    jpeg: Data(repeating: UInt8(i & 0xFF), count: 1000 + i), source: "a.mp4")
    }
    let freed1 = releaseFrameJPEG(&poolFrames, range: 0..<32)
    expect(freed1 > 0, "освобождение: возвращает число освобождённых байт")
    expect(poolFrames[0].jpeg.isEmpty && poolFrames[31].jpeg.isEmpty,
           "освобождение: JPEG 0 и 31 стали пустыми")
    expect(!poolFrames[32].jpeg.isEmpty, "освобождение: кадр 32 (вне диапазона) не тронут")
    // Повторное освобождение — ничего не освобождает (уже пусто).
    let freed2 = releaseFrameJPEG(&poolFrames, range: 0..<32)
    expect(freed2 == 0, "освобождение: повторно — 0 байт (защита от двойного)")
    // Полный диапазон за пределами массива — безопасно (без падения).
    let freed3 = releaseFrameJPEG(&poolFrames, range: 100..<200)
    expect(freed3 == 0, "освобождение: диапазон вне массива — безопасно")

    // 14. бэкап чекпоинтов перед стартом новой сессии (TDD)
    do {
        let tmp = NSTemporaryDirectory() + "yume_tdd_backup_\(UUID().uuidString)"
        let fm = FileManager.default
        try fm.createDirectory(atPath: tmp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: tmp) }

        let existing = tmp + "/conv.safetensors"
        let payload = Data("test-weights".utf8)
        try payload.write(to: URL(fileURLWithPath: existing))

        let empty = tmp + "/empty.safetensors"
        try Data().write(to: URL(fileURLWithPath: empty))

        let missing = tmp + "/missing.safetensors"

        // Непустой файл → бэкап создаётся, содержимое совпадает.
        let b1 = createCheckpointBackup(fromPath: existing)
        expect(b1 != nil, "бэкап: непустой файл → бэкап создан")
        if let b1 = b1 {
            expect(fm.fileExists(atPath: b1), "бэкап: файл существует на диске")
            expect((try? Data(contentsOf: URL(fileURLWithPath: b1))) == payload,
                   "бэкап: содержимое совпадает с оригиналом")
            // Бэкап не равен оригинальному пути.
            expect(b1 != existing, "бэкап: путь отличается от оригинала (суффикс времени)")
        }

        // Коллизия: повторный вызов в ту же секунду — создаётся ДРУГАЯ копия
        // (а не упавший nil). Это фикс реального бага.
        let b2 = createCheckpointBackup(fromPath: existing)
        expect(b2 != nil, "бэкап: повторный вызов не падает (коллизия)")
        if let b1 = b1, let b2 = b2 {
            expect(b2 != b1, "бэкап: повторный вызов создаёт другую копию (уникальное имя)")
        }

        // Непустой файл детектится isNonEmptyFile.
        expect(isNonEmptyFile(existing), "бэкап: isNonEmptyFile(непустой) = true")
        expect(!isNonEmptyFile(empty), "бэкап: isNonEmptyFile(пустой) = false")
        expect(!isNonEmptyFile(missing), "бэкап: isNonEmptyFile(нет файла) = false")

        // Пустой/отсутствующий файл → бэкап не создаётся.
        expect(createCheckpointBackup(fromPath: empty) == nil, "бэкап: пустой файл → nil")
        expect(createCheckpointBackup(fromPath: missing) == nil, "бэкап: нет файла → nil")

        // createBackups: бэкапит только существующие непустые.
        let backs = createBackups(forPaths: [existing, empty, missing])
        expect(backs.count == 1, "бэкап: createBackups бэкапит только 1 непустой из 3")
        expect(backs[0] != existing, "бэкап: createBackups возвращает путь копии")
    } catch {
        expect(false, "бэкап: настройка временных файлов — ошибка: \(error.localizedDescription)")
    }

}

func selftestVramPoolUiModes(_ expect: (Bool, String) -> Void) {
    // 15. редактируемое поле: видимость элементов в зависимости от режима
    let normal = editFieldVisibility(isEditing: false)
    expect(normal.showValue && normal.showEditButton && !normal.showInputWithActions,
           "поле: в обычном режиме виден лейбл и карандаш, поле скрыто")
    let editing = editFieldVisibility(isEditing: true)
    expect(!editing.showValue && !editing.showEditButton && editing.showInputWithActions,
           "поле: в режиме правки видно поле с ✓/✕, лейбл и карандаш скрыты")

    // 16. размер порции дозагрузки в VRAM-пул (подстраивается под память/GPU)
    // frameBytes для 856×480 float32 ≈ 4.93 МБ; RAM 32 ГБ; рабочий набор GPU ~6.5 ГБ.
    let frameBytes856 = 856 * 480 * 3 * 4
    let ram32 = UInt64(32 * 1024 * 1024 * 1024)
    let gpu65 = UInt64(6_500_000_000)

    // Реальный случай: порция заметно больше 64 (консервативного значения), но < 1024.
    let realChunk = frameLoadChunkSize(frameBytes: frameBytes856,
                                       availableRAM: ram32, maxWorkingSet: gpu65)
    expect(realChunk >= 64 && realChunk <= 1024,
           "порция: реальное значение в диапазоне [64, 1024]")
    expect(realChunk > 64, "порция: подстраивается под память (больше консервативных 64)")
    expect(realChunk < 1024, "порция: не упирается в потолок 1024")

    // Логика бюджета: больше RAM → не меньше порция.
    let bigRAM = frameLoadChunkSize(frameBytes: frameBytes856,
                                    availableRAM: UInt64(128 * 1024 * 1024 * 1024),
                                    maxWorkingSet: gpu65)
    let smallRAM = frameLoadChunkSize(frameBytes: frameBytes856,
                                      availableRAM: UInt64(8 * 1024 * 1024 * 1024),
                                      maxWorkingSet: gpu65)
    expect(bigRAM >= smallRAM, "порция: больше RAM → порция не меньше")

    // Безопасные краевые случаи.
    expect(frameLoadChunkSize(frameBytes: 0, availableRAM: ram32, maxWorkingSet: gpu65) == 64,
           "порция: frameBytes=0 → дефолт 64")
    expect(frameLoadChunkSize(frameBytes: frameBytes856,
                              availableRAM: 1, maxWorkingSet: 1) == 64,
           "порция: нет ресурсов → нижняя граница 64")
    // Огромная память на крошечном кадре → упирается в потолок 1024.
    let huge = frameLoadChunkSize(frameBytes: 1000,
                                  availableRAM: UInt64(512 * 1024 * 1024 * 1024),
                                  maxWorkingSet: UInt64(128_000_000_000))
    expect(huge == 1024, "порция: огромные ресурсы → потолок 1024")

    // 17. реальный прогресс в кадрах (а не в шагах)
    // При батче 8: шаг 4800 из 5398 = 38400 из 43184 кадров (89%).
    let p1 = progressInFrames(stepDone: 4800, totalSteps: 5398, batch: 8)
    expect(p1 == (38400, 43184), "прогресс: 4800 шагов × 8 = 38400/43184 кадров")
    expect(Double(p1.done) / Double(p1.total) > 0.889
           && Double(p1.done) / Double(p1.total) < 0.890,
           "прогресс: ~89% реального объёма")
    // Батч 1 — количество шагов равно количеству кадров.
    let p2 = progressInFrames(stepDone: 100, totalSteps: 43187, batch: 1)
    expect(p2 == (100, 43187), "прогресс: батч 1 → шаги = кадры")
    // Батч 0/отрицательный — защита (минимум 1).
    let p3 = progressInFrames(stepDone: 10, totalSteps: 100, batch: 0)
    expect(p3 == (10, 100), "прогресс: батч 0 → дефолт 1")
    // Начало работы — 0.
    let p4 = progressInFrames(stepDone: 0, totalSteps: 5398, batch: 8)
    expect(p4 == (0, 43184), "прогресс: шаг 0 → 0 кадров")

}

func selftestProgressPaginationReconsPng(_ expect: (Bool, String) -> Void) {
    // 18. переход в край пагинации (начало/конец)
    expect(edgePage(current: 5, pages: 100, toStart: true) == 0,
           "пагинация: переход в начало → страница 0")
    expect(edgePage(current: 5, pages: 100, toStart: false) == 99,
           "пагинация: переход в конец → последняя страница")
    expect(edgePage(current: 0, pages: 1, toStart: false) == 0,
           "пагинация: одна страница → 0")
    expect(edgePage(current: 0, pages: 0, toStart: false) == 0,
           "пагинация: нет страниц → 0 (безопасно)")
    expect(edgePage(current: 3, pages: 5, toStart: true) == 0,
           "пагинация: в начало на 5 страницах → 0")
    expect(edgePage(current: 0, pages: 5, toStart: false) == 4,
           "пагинация: в конец → 4")

    // 18a. автоматический пересчёт видимых кружков от ширины колонки.
    // Ширина центральной колонки при окне 1360: (1360 − 440 − 2)/2 = 459 pt.
    // Стрелки: 4 × 28 pt, зазор 4 pt → доступно 327 pt.
    expect(dotCapacity(columnWidth: 459, dotWidth: 28) == 10,
           "пагинация: 459 pt, кружок 28 → 10 видимых")
    expect(dotCapacity(columnWidth: 459, dotWidth: 40) == 7,
           "пагинация: 459 pt, кружок 40 (4 цифры) → 7 видимых")
    expect(dotCapacity(columnWidth: 320, dotWidth: 40) == 4,
           "пагинация: узкая колонка 320 → 4")
    expect(dotCapacity(columnWidth: 120, dotWidth: 40) == 1,
           "пагинация: тесно → минимум 1 кружок")
    // Ширина кружка растёт с числом цифр номера страницы, чтобы не обрезать.
    expect(paginatorDotWidth(pageNumber: 1) == 28, "пагинация: 1 цифра → 28")
    expect(paginatorDotWidth(pageNumber: 99) == 28, "пагинация: 2 цифры → 28")
    expect(paginatorDotWidth(pageNumber: 100) == 34, "пагинация: 3 цифры → 34")
    expect(paginatorDotWidth(pageNumber: 999) == 34, "пагинация: 3 цифры (999) → 34")
    expect(paginatorDotWidth(pageNumber: 1000) == 40, "пагинация: 4 цифры → 40")
    expect(paginatorDotWidth(pageNumber: 4319) == 40, "пагинация: 4 цифры (4319) → 40")
    // Окно видимых страниц: текущая держится в центре, у краёв прижимается.
    expect(paginatorVisiblePages(current: 0, pages: 4319, limit: 7) == [0, 1, 2, 3, 4, 5, 6],
           "пагинация: старт → первые 7 страниц")
    expect(paginatorVisiblePages(current: 4318, pages: 4319, limit: 7) == Array(4312...4318),
           "пагинация: конец → последние 7 страниц")
    expect(paginatorVisiblePages(current: 2000, pages: 4319, limit: 7) == Array(1997...2003),
           "пагинация: середина → текущая в центре")
    expect(paginatorVisiblePages(current: 0, pages: 3, limit: 7) == [0, 1, 2],
           "пагинация: страниц меньше лимита → все")
    expect(paginatorVisiblePages(current: 5, pages: 0, limit: 7) == [],
           "пагинация: нет страниц → пусто (безопасно)")

    // 18b. Фактическая ширина строки пагинатора = ширина центральной колонки
    // минус её горизонтальные отступы (24 + 24). Считать от колонки целиком
    // нельзя: строка уже на 48 pt, и на 3–4-значных номерах кружки снова
    // распирают колонку.
    expect(paginatorRowWidth(columnWidth: 494) == 446,
           "пагинация: колонка 494 → строка 446 (минус отступы 48)")
    expect(paginatorRowWidth(columnWidth: 419) == 371,
           "пагинация: колонка 419 (узкое окно) → строка 371")
    expect(paginatorRowWidth(columnWidth: 100) == 120,
           "пагинация: вырожденная ширина → безопасный минимум")
    // Вместимость считается уже от фактической ширины строки.
    expect(dotCapacity(columnWidth: paginatorRowWidth(columnWidth: 494),
                       dotWidth: 28) == 9,
           "пагинация: 446 pt, кружок 28 → 9 видимых")
    expect(dotCapacity(columnWidth: paginatorRowWidth(columnWidth: 494),
                       dotWidth: 34) == 8,
           "пагинация: 446 pt, кружок 34 (3 цифры) → 8 видимых")
    expect(dotCapacity(columnWidth: paginatorRowWidth(columnWidth: 494),
                       dotWidth: 40) == 7,
           "пагинация: 446 pt, кружок 40 (4 цифры) → 7 видимых")

    // 19. прогресс реконструкций: когда писать в журнал
    // Старт и финиш — всегда.
    expect(shouldLogProgress(done: 0, total: 1800, lastLoggedPercent: 0),
           "журнал: старт (0) логируется")
    expect(shouldLogProgress(done: 1800, total: 1800, lastLoggedPercent: 95),
           "журнал: финиш (total) логируется")
    // Каждые 5%: при 5% и 10% логируется; на 4% — нет.
    expect(shouldLogProgress(done: 90, total: 1800, lastLoggedPercent: 0),
           "журнал: 5% логируется")
    expect(shouldLogProgress(done: 180, total: 1800, lastLoggedPercent: 5),
           "журнал: 10% логируется")
    expect(!shouldLogProgress(done: 72, total: 1800, lastLoggedPercent: 0),
           "журнал: 4% — не логируется")
    expect(!shouldLogProgress(done: 100, total: 1800, lastLoggedPercent: 5),
           "журнал: 5.5% (до 10%) — не логируется повторно")
    // После 98% идём к 100% — сам финиш (done==total) логируется отдельно.
    expect(!shouldLogProgress(done: 1799, total: 1800, lastLoggedPercent: 95),
           "журнал: 99% (до границы 100%) — не логируется повторно")
    // Пересечение очередной границы 5% на 95% (после 90%).
    expect(shouldLogProgress(done: 1710, total: 1800, lastLoggedPercent: 90),
           "журнал: 95% логируется после 90%")
    // Безопасный случай: total=0.
    expect(shouldLogProgress(done: 0, total: 0, lastLoggedPercent: 0),
           "журнал: total=0 → логируется")

}

func selftestAutoTuneFingerprint(_ expect: (Bool, String) -> Void) {
    // 20. PNG в памяти (encodePNG) вместо записи на диск
    // Простая картинка 2×2, красный квадрат.
    let red = [Float](repeating: 1.0, count: 2 * 2 * 3)  // 1.0 = красный
    guard let png = encodePNG(width: 2, height: 2, rgb: red) else {
        expect(false, "PNG: encodePNG вернул nil")
        return
    }
    expect(png.count > 8, "PNG: данные не пустые")
    // Магия PNG: 0x89 'P' 'N' 'G'.
    let magic = [UInt8](png.prefix(8))
    expect(magic == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
           "PNG: корректная магия заголовка")
    // Неверный размер → nil.
    expect(encodePNG(width: 0, height: 2, rgb: red) == nil, "PNG: width=0 → nil")
    expect(encodePNG(width: 2, height: 3, rgb: red) == nil, "PNG: мало данных (2×2 на 2×3) → nil")
    // Кодирование и декодирование: пиксель белый воспроизводится (через CGImageSource).
    let white = [Float](repeating: 1.0, count: 1 * 1 * 3)
    if let whitePNG = encodePNG(width: 1, height: 1, rgb: white),
       let src = CGImageSourceCreateWithData(whitePNG as CFData, nil),
       let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
       let data = img.dataProvider?.data as Data? {
        let px = [UInt8](data)
        // RGBA premultiplied: белый → все каналы 255.
        expect(px.count >= 4 && px[0] > 240 && px[1] > 240 && px[2] > 240,
               "PNG: белый пиксель воспроизводится")
    } else {
        expect(false, "PNG: не удалось декодировать обратно")
    }

    // 21. автоподбор параметров с учётом resume
    // Resume включён → НЕ пересчитываем (сохраняем текущие, changed=false),
    // но LR авто-снижается ×0.5 (если не задан вручную).
    let au1 = autoTuneValues(frames: 43187, batch: 8, resume: true,
                             currentSteps: 5000, currentEpochs: 3,
                             currentLr: 0.0003,
                             stepsEditedByUser: false, epochsEditedByUser: false,
                             lrEditedByUser: false)
    expect(au1.steps == 5000 && au1.epochs == 3 && !au1.changed,
           "авто: resume=true → сохраняем текущие шаги/эпохи, без изменений")
    expect(au1.lr != nil && abs(au1.lr! - 0.00015) < 1e-9,
           "авто: resume=true → LR авто-снижен ×0.5 (0.0003 → 0.00015)")
    // Resume + LR задан вручную → LR не трогаем (nil).
    let au6 = autoTuneValues(frames: 43187, batch: 8, resume: true,
                             currentSteps: 5000, currentEpochs: 3,
                             currentLr: 0.0001,
                             stepsEditedByUser: true, epochsEditedByUser: true,
                             lrEditedByUser: true)
    expect(au6.lr == nil, "авто: resume + LR вручную → LR не трогаем")
    // Resume выключён, ничего не редактировали → пересчёт, LR не трогаем.
    let au2 = autoTuneValues(frames: 43187, batch: 8, resume: false,
                             currentSteps: 20, currentEpochs: 20,
                             currentLr: 0.0003,
                             stepsEditedByUser: false, epochsEditedByUser: false,
                             lrEditedByUser: false)
    expect(au2.steps == 5398 && au2.changed, "авто: шаги = frames/batch = 5398, changed=true")
    expect(au2.lr == nil, "авто: resume=false → LR не трогаем")
    // Эпохи: min(10, max(3, 3000/frames)); для 43187 → 3.
    expect(au2.epochs == 3, "авто: эпохи = 3 для 43187 кадров")
    // Resume выключён, но шаги редактированы вручную → эпохи пересчитываются, шаги нет.
    let au3 = autoTuneValues(frames: 43187, batch: 8, resume: false,
                             currentSteps: 1234, currentEpochs: 20,
                             currentLr: 0.0003,
                             stepsEditedByUser: true, epochsEditedByUser: false,
                             lrEditedByUser: false)
    expect(au3.steps == 1234 && au3.epochs == 3 && au3.changed,
           "авто: шаги редактированы → не трогаем, эпохи авто")
    // Оба поля редактированы → ничего не меняется.
    let au4 = autoTuneValues(frames: 43187, batch: 8, resume: false,
                             currentSteps: 100, currentEpochs: 5,
                             currentLr: 0.0003,
                             stepsEditedByUser: true, epochsEditedByUser: true,
                             lrEditedByUser: false)
    expect(!au4.changed, "авто: оба поля редактированы → changed=false")
    // frames=0 → безопасно, без изменений.
    let au5 = autoTuneValues(frames: 0, batch: 8, resume: false,
                             currentSteps: 20, currentEpochs: 20,
                             currentLr: 0.0003,
                             stepsEditedByUser: false, epochsEditedByUser: false,
                             lrEditedByUser: false)
    expect(!au5.changed && au5.steps == 20, "авто: frames=0 → без изменений")

    // 22. Отпечаток данных (детерминированный, реагирует на содержимое).
    let fa = [MemoryFrame(name: "a0.jpg", jpeg: Data([1,2,3]), source: "a.mp4"),
              MemoryFrame(name: "a1.jpg", jpeg: Data([4,5,6]), source: "a.mp4")]
    let fa2 = [MemoryFrame(name: "a0.jpg", jpeg: Data([1,2,3]), source: "a.mp4"),
               MemoryFrame(name: "a1.jpg", jpeg: Data([4,5,6]), source: "a.mp4")]
    let fb = [MemoryFrame(name: "b0.jpg", jpeg: Data([9,9,9]), source: "b.mp4"),
              MemoryFrame(name: "b1.jpg", jpeg: Data([7,7,7]), source: "b.mp4")]
    expect(dataFingerprint(frames: fa) == dataFingerprint(frames: fa2),
           "отпечаток: одинаковые данные → одинаковый отпечаток")
    expect(dataFingerprint(frames: fa) != dataFingerprint(frames: fb),
           "отпечаток: разные данные → разные отпечатки")
    expect(dataFingerprint(frames: []) == "empty",
           "отпечаток: пустой набор → 'empty'")

}

func selftestLiveStatusAndVram(_ expect: (Bool, String) -> Void) {
    // 24. Живой статус обучения: эпоха, текущий кадр, шаг, скорость.
    expect(formatThousands(43187) == "43 187", "статус: 43187 → '43 187'")
    expect(formatThousands(0) == "0", "статус: 0 → '0'")
    expect(formatThousands(1234567) == "1 234 567", "статус: 1234567 → '1 234 567'")

    let live1 = TrainStepReport(done: 12480, total: 129561,
                                epoch: 1, epochs: 3,
                                step: 12480, stepsPerEpoch: 43187,
                                frame: 12480, framesTotal: 43187,
                                batch: 1, secondsPerStep: 0.46)
    expect(trainLiveText(live1) ==
           "Эпоха 1/3 · кадр 12 480 из 43 187 · шаг 12 480/43 187 · 0.46 с/шаг",
           "статус: батч 1 → эпоха/кадр/шаг/скорость")
    let live2 = TrainStepReport(done: 1, total: 129561,
                                epoch: 1, epochs: 3,
                                step: 1, stepsPerEpoch: 5399,
                                frame: 1, framesTotal: 43187,
                                batch: 8, secondsPerStep: nil)
    expect(trainLiveText(live2) ==
           "Эпоха 1/3 · кадр 1 из 43 187 · батч 8 · шаг 1/5 399",
           "статус: батч 8 → кадр + размер батча")
    let live3 = TrainStepReport(done: 5, total: 20,
                                epoch: 1, epochs: 1,
                                step: 5, stepsPerEpoch: 20,
                                frame: 0, framesTotal: 0,
                                batch: 1, secondsPerStep: nil)
    expect(trainLiveText(live3) == "Эпоха 1/1 · шаг 5/20",
           "статус: без кадров → только эпоха и шаг")
    // Скорость округляется до сотых, чтобы не дёргалась на каждом шаге.
    let live4 = TrainStepReport(done: 1, total: 10,
                                epoch: 1, epochs: 1,
                                step: 1, stepsPerEpoch: 10,
                                frame: 1, framesTotal: 10,
                                batch: 1, secondsPerStep: 0.456)
    expect(trainLiveText(live4).hasSuffix("· 0.46 с/шаг"),
           "статус: 0.456 с/шаг → '0.46 с/шаг'")

    // 25. «Идёт или зависло»: состояние по времени с последнего шага.
    expect(trainLivePhase(secondsSinceStep: 5, stepSeconds: 0.46) == .running,
           "статус: 5 с назад → обучение идёт")
    expect(trainLivePhase(secondsSinceStep: 50, stepSeconds: 0.46) == .waiting(seconds: 50),
           "статус: 50 с назад → последний шаг давно")
    expect(trainLivePhase(secondsSinceStep: 95, stepSeconds: 0.46) == .stalled(seconds: 95),
           "статус: 95 с назад → возможно зависло")
    // Медленные шаги (3 с/шаг) не должны давать ложное «зависло» на 95 с.
    expect(trainLivePhase(secondsSinceStep: 95, stepSeconds: 3) == .running,
           "статус: медленные шаги 3 с → 95 с ещё норма")
    expect(trainLivePhaseText(.running) == "Обучение идёт",
           "статус: текст running")
    expect(trainLivePhaseText(.waiting(seconds: 50)) == "Последний шаг 50 с назад",
           "статус: текст waiting")
    expect(trainLivePhaseText(.stalled(seconds: 95)) ==
           "Нет шагов 1 мин 35 с — возможно, зависло",
           "статус: текст stalled с минутами")
    expect(formatShortDuration(125) == "2 мин 5 с",
           "статус: 125 с → '2 мин 5 с'")

}

func selftestPrepAndFrameIndex(_ expect: (Bool, String) -> Void) {
    // 26. Ёмкость VRAM-пула: как в ранних прогонах — 1024 кадра при 8.57 ГБ.
    expect(gpuFramePoolCapacity(framesTotal: 43187, frameBytes: 4_930_560,
                                workingSet: 8_570_000_000) == 1024,
           "VRAM: 8.57 ГБ → 1024 кадра (≈4.8 ГБ)")
    expect(gpuFramePoolCapacity(framesTotal: 300, frameBytes: 4_930_560,
                                workingSet: 8_570_000_000) == 300,
           "VRAM: кадров меньше потолка → все")
    expect(gpuFramePoolCapacity(framesTotal: 43187, frameBytes: 4_930_560,
                                workingSet: 2_000_000_000) == 243,
           "VRAM: 2 ГБ → 243 кадра (60% бюджета)")
    expect(gpuFramePoolCapacity(framesTotal: 43187, frameBytes: 0,
                                workingSet: 8_570_000_000) == 1,
           "VRAM: нулевой кадр → минимум 1 (без деления на 0)")
    expect(gpuFramePoolCapacity(framesTotal: 0, frameBytes: 4_930_560,
                                workingSet: 8_570_000_000) == 0,
           "VRAM: нет кадров → 0")

    // 27. Фаза подготовки: пока кадры грузятся в VRAM, шагов ещё нет —
    // интерфейс обязан показывать прогресс загрузки, а не «зависло».
    expect(trainPreparationNote() == "Подготовка: загрузка модели на GPU…",
           "статус: текст подготовки")
    expect(trainVRAMLoadNote(loaded: 512, total: 1024) == "Загрузка кадров в VRAM: 512/1024",
           "статус: прогресс загрузки VRAM")
    expect(trainVRAMLoadNote(loaded: 1024, total: 1024) == "Загрузка кадров в VRAM: 1024/1024",
           "статус: VRAM загружена полностью")
    let prepReport = TrainStepReport(done: 0, total: 129561,
                                     epoch: 1, epochs: 3,
                                     step: 0, stepsPerEpoch: 43187,
                                     frame: 0, framesTotal: 43187,
                                     batch: 1, secondsPerStep: nil,
                                     note: trainVRAMLoadNote(loaded: 512, total: 1024),
                                     stage: .preparing)
    expect(trainLiveText(prepReport) == "Загрузка кадров в VRAM: 512/1024",
           "статус: отчёт с пометкой показывает пометку")
    expect(trainLiveStateText(phase: .running, stage: .preparing) == "Подготовка данных",
           "статус: состояние «подготовка данных»")
    expect(trainLiveStateText(phase: .stalled(seconds: 95), stage: .preparing) ==
           "Подготовка данных не идёт 1 мин 35 с — возможно, зависло",
           "статус: зависание на фазе подготовки")
    expect(trainLiveStateText(phase: .running, stage: .training) == "Обучение идёт",
           "статус: состояние при обучении")

    // 28. Номер кадра, который обучается прямо сейчас (1-based).
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 0, stepsPerEpoch: 43187,
                             batch: 1, datasetFrames: 43187) == 1,
           "кадр: первый шаг → кадр 1")
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 12479, stepsPerEpoch: 43187,
                             batch: 1, datasetFrames: 43187) == 12480,
           "кадр: шаг 12480 → кадр 12480")
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 43186, stepsPerEpoch: 43187,
                             batch: 1, datasetFrames: 43187) == 43187,
           "кадр: последний кадр эпохи → 43 187")
    expect(currentFrameIndex(epochIndex: 1, stepIndex: 0, stepsPerEpoch: 43187,
                             batch: 1, datasetFrames: 43187) == 1,
           "кадр: вторая эпоха → набор начинается заново (кадр 1)")
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 1, stepsPerEpoch: 5399,
                             batch: 8, datasetFrames: 43187) == 9,
           "кадр: батч 8 → первый кадр следующей пачки (9)")
    expect(currentFrameIndex(epochIndex: 0, stepIndex: 0, stepsPerEpoch: 10,
                             batch: 1, datasetFrames: 0) == 0,
           "кадр: набора нет → 0 (кадр не показываем)")

}

func selftestLocalApi(_ expect: (Bool, String) -> Void) {
    // 23. Локальный API: белый список Host привязан к фактическому порту
    // сервера. Иначе второй экземпляр (диагностика) не может ответить.
    func httpRequest(_ host: String, origin: String? = nil) -> Data {
        var s = "GET /api/state HTTP/1.1\r\nHost: \(host)\r\n"
        if let o = origin { s += "Origin: \(o)\r\n" }
        s += "\r\n"
        return Data(s.utf8)
    }
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17851"), port: 17851)) != nil,
           "API: Host на своём порту (17851) принимается")
    expect((try? LocalHTTPRequest.parse(httpRequest("localhost:17851"), port: 17851)) != nil,
           "API: localhost на своём порту принимается")
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17852"), port: 17852)) != nil,
           "API: Host на своём порту (17852) принимается")
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17851"), port: 17852)) == nil,
           "API: чужой порт в Host отклоняется")
    expect((try? LocalHTTPRequest.parse(httpRequest("evil.example:17851"), port: 17851)) == nil,
           "API: чужой хост отклоняется")
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17852", origin: "http://127.0.0.1:17852"), port: 17852)) != nil,
           "API: Origin совпадает с Host — принимается")
    expect((try? LocalHTTPRequest.parse(httpRequest("127.0.0.1:17852", origin: "http://evil.example"), port: 17852)) == nil,
           "API: чужой Origin отклоняется")
}

