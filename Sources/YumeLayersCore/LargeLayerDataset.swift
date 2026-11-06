import Foundation

/// Собрать большой датасет из одного фильма: детекция шотов → плотная
/// аннотация внутри каждого шота → объединение сцен.
///
/// Это путь от «16 кадров одного отрывка» к «сотням разнородных сцен» без
/// потери качества разметки: фон считается в пределах каждого шота, где
/// задник стабилен.
public func buildLargeLayerDataset(url: URL, w: Int, h: Int,
                                   targetFrames: Int = 600,
                                   framesPerShot: Int = 12,
                                   scanCount: Int = 240,
                                   maxShift: Int = 8) throws -> LayerDataset {
    let shots = try detectSceneShots(url: url, w: w, h: h, scanCount: scanCount)

    var allScenes: [LayeredScene] = []
    var allFrames: [[Float]] = []
    var totalCoverage: Double = 0
    var worstComposite: Float = 0
    var firstAlignment: FrameAlignmentReport? = nil
    var dw = 0, dh = 0

    for shot in shots {
        if allFrames.count >= targetFrames { break }
        let f = try loadShotFrames(url: url, w: w, h: h,
                                   start: shot.start, end: shot.end,
                                   count: framesPerShot)
        if f.count < 4 { continue }
        let ds = makeLayerDataset(frames: f, w: w, h: h,
                                  stabilize: true, maxShift: maxShift)
        guard !ds.scenes.isEmpty else { continue }
        if dw == 0 { dw = ds.w; dh = ds.h }
        if ds.w != dw || ds.h != dh { continue }   // все шоты одного размера
        // Проверка лимита ДО добавления: раньше шот добавлял до
        // framesPerShot кадров поверх уже набранного лимита (до +11 лишних).
        for (i, sc) in ds.scenes.enumerated() where allFrames.count < targetFrames {
            allScenes.append(sc)
            allFrames.append(ds.frames[i])
            totalCoverage += objectCoverage(sc)
            let got = composeLayers(w: dw, h: dh, background: sc.background,
                                    objects: sc.objects, mask: sc.mask,
                                    light: sc.light)
            worstComposite = max(worstComposite, maxPixelDelta(got, ds.frames[i]))
        }
        if firstAlignment == nil { firstAlignment = ds.alignment }
    }

    guard !allScenes.isEmpty else { throw LayerDatasetError.emptySource }
    return LayerDataset(w: dw, h: dh, frames: allFrames,
                        scenes: allScenes,
                        meanCoverage: totalCoverage / Double(allScenes.count),
                        compositeError: worstComposite,
                        alignment: firstAlignment)
}
