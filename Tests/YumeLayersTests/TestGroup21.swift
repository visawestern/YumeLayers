// YumeLayers — политика логирования недоставленного обработчика завершения GPU.
//
// На AMD Radeon Pro 5500M драйвер регулярно не доставляет addCompletedHandler,
// хотя буфер уже выполнен (status == .completed). Это штатный обход: выходим по
// статусу и продолжаем. Раньше о каждом таком обходе печаталось «⚠️», и длинные
// прогоны тонули в тысячах ложных предупреждений, за которыми не увидеть
// настоящую ошибку. Тест фиксирует троттлинг и классификацию (чистые функции из
// GpuWaitPolicy.swift).
import Foundation

func testGpuHandlerMissPolicy() {
    section("GpuHandlerMissPolicy: троттлинг штатных обходов недоставленного обработчика")

    // completed-обход — штатный, не настоящая проблема; failed — да.
    expect(!gpuHandlerMissIsRealProblem(state: .completed),
           "completed-обход без обработчика — не проблема")
    expect(gpuHandlerMissIsRealProblem(state: .failed),
           "failed-статус — настоящая проблема")

    // Троттлинг: первые 3 штатных обхода печатаем, дальше молчим до 500-го.
    expect(gpuShouldLogHandlerMiss(state: .completed, count: 1),
           "1-й штатный обход — печатаем")
    expect(gpuShouldLogHandlerMiss(state: .completed, count: 3),
           "3-й штатный обход — печатаем")
    expect(!gpuShouldLogHandlerMiss(state: .completed, count: 4),
           "4-й штатный обход — молчим (не flooding)")
    expect(!gpuShouldLogHandlerMiss(state: .completed, count: 42),
           "рядовой обход — молчим")
    expect(gpuShouldLogHandlerMiss(state: .completed, count: 500),
           "каждый 500-й — печатаем (видно нарастание)")
    expect(gpuShouldLogHandlerMiss(state: .failed, count: 9999),
           "ошибка драйвера — печатаем всегда")
    expect(!gpuShouldLogHandlerMiss(state: .completed, count: 0),
           "нулевой счётчик — не печатаем мусор")
}
