// YumeLayers Core — синхронный мост к асинхронным свойствам AVAsset.
//
// С macOS 13 Apple пометила `AVAsset.duration`, `AVAsset.tracks(withMediaType:)`
// и `AVAssetTrack.nominalFrameRate` устаревшими: вместо них нужно использовать
// `load(_:)` / `loadTracks(withMediaType:)`. Публичный API извлечения кадров
// исторически синхронный (его вызывают UI и тесты без async), поэтому здесь
// один аккуратный мост: загрузка выполняется отдельной задачей, а вызывающий
// поток ждёт результат на семафоре.
import Foundation
import AVFoundation

/// Ошибка, которая не должна произойти: задача завершилась без результата.
enum AVAssetSyncLoadError: Error {
    case noResult
}

/// Безопасная передача результата из задачи в ожидающий поток.
private final class LoadResultBox<T>: @unchecked Sendable {
    var value: Result<T, Error>?
}

private func loadSync<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = LoadResultBox<T>()
    Task.detached(priority: .userInitiated) {
        do {
            box.value = .success(try await operation())
        } catch {
            box.value = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.value else { throw AVAssetSyncLoadError.noResult }
    return try result.get()
}

/// Длительность ассета (бывш. `asset.duration`).
func loadAssetDuration(_ asset: AVAsset) throws -> CMTime {
    try loadSync { try await asset.load(.duration) }
}

/// Видеодорожки ассета (бывш. `asset.tracks(withMediaType: .video)`).
func loadVideoTracks(of asset: AVAsset) throws -> [AVAssetTrack] {
    try loadSync { try await asset.loadTracks(withMediaType: .video) }
}

/// Номинальная частота кадров дорожки (бывш. `track.nominalFrameRate`).
func loadNominalFrameRate(of track: AVAssetTrack) throws -> Float {
    try loadSync { try await track.load(.nominalFrameRate) }
}
