// YumeLayers — тестовая сюита (собственный раннер).
//
// Компилируется вместе с Sources/YumeLayersCore/*.swift одним модулем,
// поэтому видит internal-API ядра без @testable. Запуск: scripts/test.sh
//
// Формат вывода намеренно машинно-читаемый: строки "ok/FAIL" + итог,
// код возврата 1 при любом падении (годится для CI и pre-commit).

import Foundation
import Metal
import AVFoundation

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

func runAll() {
    let tests: [(String, () -> Void)] = [
        ("HTTP", testHTTP),
        ("TensorBufferPool", testTensorBufferPool),
        ("TensorStagingPool", testTensorStagingPool),
        ("GradSanitize", testGradSanitizeBatched),
        ("GpuWaitPolicy", testGpuWaitPolicy),
        ("ReconOutput", testReconOutput),
        ("InterpolationPlan", testInterpolationPlan),
        ("InterpolateVideo", testInterpolateVideo),
        ("VideoWriter", testVideoWriterOrientation),
        ("LoadImageOrientation", testLoadImageOrientation),
        ("CheckpointRoundTrip", testCheckpointRoundTrip),
        ("TrainingNoise", testTrainingNoiseSeeds),
        ("FinalCheckpoint", testFinalCheckpointPath),
        ("OptimizerState", testOptimizerStateRoundTrip),
        ("CodeGuide", testCodeGuideLimits),
        ("GPUMse", testGPUMseMatchesCPU),
        ("ConvPerf", testConvPerf),
        ("FinalCheckpoint", testFinalCheckpointPath),
        ("OptimizerState", testOptimizerStateRoundTrip),
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
