// YumeLayers — самопроверка (SELFTEST), вынесена из ConvTrainer.swift.
//
// Единая функция (монолит) — сохранена без потери проверок; оформлена как
// документированный долг (см. docs/ROADMAP.md, Фаза 8).
import Foundation
import CoreGraphics
import ImageIO

public func runSelfTest() {
    var failures = 0
    func expect(_ cond: Bool, _ name: String) {
        yumeLog((cond ? "✅ " : "❌ ") + name)
        if !cond { failures += 1 }
    }

    selftestDeliveryPlanner(expect)
    selftestBatchesAndClips(expect)
    selftestGpuMseAndControl(expect)
    selftestJpegReleaseAndCheckpoint(expect)
    selftestVramPoolUiModes(expect)
    selftestProgressPaginationReconsPng(expect)
    selftestAutoTuneFingerprint(expect)
    selftestLiveStatusAndVram(expect)
    selftestPrepAndFrameIndex(expect)
    selftestLocalApi(expect)

    // Таблица производительности (Фаза 7: 96/128/192px). Только логирует.
    benchmarkResolutions()

    yumeLog(failures == 0 ? "SELFTEST PASS" : "SELFTEST FAIL: \(failures)")
}
