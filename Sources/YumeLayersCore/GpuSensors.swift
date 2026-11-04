// Чтение телеметрии дискретной GPU без root.
//
// На Intel-маках с AMD дискретной картой IORegistry-узел IOAccelerator
// публикует словарь PerformanceStatistics, в котором есть температура ядра,
// частота, загрузка и потребляемая мощность. Привилегий не требуется —
// в отличие от SMC/powermetrics, которые доступны только root.
//
// Это даёт то, чего не хватало для честного троттлинга: раньше единственным
// сигналом был ProcessInfo.thermalState, а он запаздывает на минуты и говорит
// о состоянии всей системы, а не GPU.

import Foundation
import IOKit

/// Снимок телеметрии GPU.
public struct GpuStats: Equatable, Sendable {
    /// Температура ядра, °C. 0, если датчик не отдал значение.
    public var temperatureC: Double
    /// Текущая частота ядра, МГц.
    public var coreClockMHz: Double
    /// Загрузка GPU, % (0…100).
    public var activityPercent: Double
    /// Потребление, Вт.
    public var powerW: Double

    public init(temperatureC: Double, coreClockMHz: Double,
                activityPercent: Double, powerW: Double) {
        self.temperatureC = temperatureC
        self.coreClockMHz = coreClockMHz
        self.activityPercent = activityPercent
        self.powerW = powerW
    }

    /// Есть ли осмысленная температура (иначе датчик молчит).
    public var hasTemperature: Bool { temperatureC > 1 && temperatureC < 150 }
}

/// Разбор словаря PerformanceStatistics.
///
/// Чистая функция: на вход — то, что вернул IORegistry, на выход — снимок.
/// Значения приходят как Int, но на разных драйверах бывают Double и String,
/// поэтому приводим терпимо.
public func parseGpuStats(_ raw: [String: Any]) -> GpuStats? {
    func num(_ key: String) -> Double? {
        if let v = raw[key] as? Int { return Double(v) }
        if let v = raw[key] as? Double { return v }
        if let v = raw[key] as? NSNumber { return v.doubleValue }
        if let v = raw[key] as? String { return Double(v) }
        return nil
    }
    // Без температуры запись бесполезна: это, скорее всего, iGPU или
    // узел без телеметрии.
    guard let t = num("Temperature(C)"), t > 1 else { return nil }
    return GpuStats(temperatureC: t,
                    coreClockMHz: num("Core Clock(MHz)") ?? 0,
                    activityPercent: num("GPU Activity(%)") ?? 0,
                    powerW: num("Total Power(W)") ?? 0)
}

/// Читатель телеметрии. Держит только сопоставление сервиса — без root.
public final class GpuSensor {
    private let lock = NSLock()
    private var cachedService: io_service_t = 0
    /// Последний удачный снимок: если узел на миг пропал, лучше отдать
    /// устаревшее значение, чем сорвать регулирование.
    private var last: GpuStats?

    public init() {}

    deinit {
        if cachedService != 0 { IOObjectRelease(cachedService) }
    }

    public func read() -> GpuStats? {
        lock.lock(); defer { lock.unlock() }
        if let s = readOnce() {
            last = s
            return s
        }
        return last
    }

    private func readOnce() -> GpuStats? {
        if cachedService != 0 {
            if let s = stats(of: cachedService) { return s }
            // Узел пересоздался (смена GPU) — ищем заново.
            IOObjectRelease(cachedService)
            cachedService = 0
        }
        guard let match = IOServiceMatching("IOAccelerator") else { return nil }
        var iter: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, match, &iter) == KERN_SUCCESS else {
            return nil
        }
        defer { IOObjectRelease(iter) }

        var best: (io_service_t, GpuStats)? = nil
        while case let svc = IOIteratorNext(iter), svc != 0 {
            if let s = stats(of: svc) {
                // Берём узел с самой горячей/самой нагруженной картой:
                // у iGPU температура либо нулевая, либо заметно ниже.
                if best == nil || s.temperatureC > best!.1.temperatureC {
                    if best != nil { IOObjectRelease(best!.0) }
                    best = (svc, s)
                } else {
                    IOObjectRelease(svc)
                }
            } else {
                IOObjectRelease(svc)
            }
        }
        guard let b = best else { return nil }
        cachedService = b.0   // владение переходит к кэшу
        return b.1
    }

    private func stats(of service: io_service_t) -> GpuStats? {
        guard let cf = IORegistryEntryCreateCFProperty(
            service, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0) else { return nil }
        guard let d = cf.takeRetainedValue() as? [String: Any] else { return nil }
        return parseGpuStats(d)
    }
}
