// YumeLayers — обучение свёрточного VAE на Metal (480p и любое разрешение, кратное 8).
import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import Metal

/// Вес KL-регуляризатора для RGB-свёрточного VAE.
///
/// β=1.0 (по умолчанию в `backward`) доминирует над реконструкцией: KL-градиент
/// по элементам масштабируется как `β/M`, а MSE — как `2/N`; при N≈1.3e4 и
/// M≈1e3 KL подавляет MSE, латент схлопывается к N(0,1), и декодер выдаёт
/// тёмную кашу даже на ярких кадрах. Замер на одном ярком кадре (80×56):
/// β=0.01 → 22.5 дБ, β=0.1 → 13.0 дБ, β=1.0 → 10.0 дБ. 0.01 — рабочая точка.

/// Сколько шагов разгона скорости делать при продолжении обучения с
/// чекпоинта БЕЗ состояния оптимизатора (старый формат файла).
///
/// Моменты Adam в таком чекпоинте отсутствуют → они нулевые → первый шаг
/// равен ±lr по каждому весу. Для обученной модели это разрушительно,
/// поэтому первые шаги идут с плавно растущей скоростью.

// ---------- Загрузка изображений ----------

/// RGB [0,1], row-major сверху вниз, ровно w×h. Читает из файла.

/// RGB [0,1], row-major сверху вниз, ровно w×h. Декодирует из памяти (JPEG).

/// Публичная точка декодирования CGImage → RGB [0,1] top-down.
/// Итерация 6: конвейеру нужно decoded-кадры из `AVAssetImageGenerator`,
/// а не только JPEG из файла/памяти. Логика одна и та же, дублировать её
/// нельзя — иначе превью и обучение будут видеть разные цвета.


/// Дополняет кадр до размеров, кратных 8 (крайним пикселем).

/// Конвертация раскладки кадра HWC (межстрочный RGB: r,g,b,r,g,b,…) в CHW
/// (планарный, канал-major: канал c, пиксель i → c·n + i). Свёрточный VAE
/// (im2col, split2/merge2) работает строго в CHW, а `decodeImageRGB` и
/// синтетические сцены отдают HWC. Раньше кадры кормились сетью как есть —
/// каналы 1 и 2 перемешивались с соседними пикселями, и модель не могла
/// выучить содержание (реконструкции схлопывались в тёмный шум).

/// Обратная конвертация: CHW → HWC. Используется при отрисовке реконструкций
/// и превью (все выходные PNG — HWC).

// ---------- Данные ----------

/// Предварительно декодированный набор кадров в RGB8 (с паддингом до кратного 8).
/// Декодирование выполняется параллельно по всем ядрам один раз до обучения,
/// чтобы в цикле не было ни JPEG-декода, ни аллокаций [Float] на каждый шаг.
/// `FrameCache` вынесен в `FrameData.swift`.

/// `DataSource` вынесен в `FrameData.swift`.

// ---------- Сохранение пары «цель | реконструкция» ----------


/// Сохранение одной картинки (RGB) в PNG.

/// Прогон обученной модели по кадрам и получение реконструкций В ПАМЯТИ
/// (без записи на диск). Возвращает словарь «имя кадра → PNG-данные».
/// Кадры берутся из контрольной выборки (p.controlFrames), если она задана,
/// иначе из памяти (p.memoryFrames), иначе с диска.
/// При наличии второй GPU (встроенная iGPU) работа делится между двумя
/// устройствами параллельно.

/// Прогон forward на одном движке (используется обеими GPU).
/// Возвращает словарь «имя кадра → PNG-данные реконструкции» В ПАМЯТИ
/// (без записи на диск).

// ---------- Обучение ----------

/// Колбэк прогресса: (эпоха, loss, psnr, путь превью, максимум эпох) -> Void
public typealias TrainProgress = (Int, Double, Double, String?) -> Void

/// Аварийная остановка обучения: loss стал NaN/Inf.
/// Отличается от отмены пользователем (CancellationError), чтобы вызывающий
/// код мог показать честную причину, а не «остановлено».
public struct ConvTrainingNaNError: Error, LocalizedError {
    public init() {}
    public var errorDescription: String? {
        "обучение остановлено: loss = NaN (чекпоинт не записан)"
    }
}

/// `TrainingControl` и `sanitizeGrads` вынесены в отдельные файлы
/// (`TrainingControl.swift`, `GradSanitize.swift`).
/// Замена NaN/Inf в градиентах на 0 + клиппинг суммарной нормы (L2).
/// Подробности и реализация — в `GradSanitize.swift`.

/// Загрузка состояния для продолжения обучения из лучшего чекпоинта.
/// Возвращает счётчик шагов Adam и флаг «нет состояния оптимизатора».
/// Эпохи никогда не пропускаются: resume грузит только веса.
func loadResumeState(into vae: ConvVAEGPU, convPath: String,
                             resume: Bool, currentFingerprint: String) throws
    -> (adamStep: Int, resumeWithoutOptimizer: Bool, savedEpoch: Int, dataMatches: Bool) {
    var adamStep = 0
    var resumeWithoutOptimizer = false
    var savedEpoch = 0
    var dataMatches = true
    // Грузим из ЛУЧШЕЙ доступной модели: защищённый финал важнее рабочего
    // файла (рабочий файл мог быть перезаписан промежуточным чекпоинтом
    // прерванного прогона).
    let resumePath = bestCheckpointPath(for: convPath)
    if resume && FileManager.default.fileExists(atPath: resumePath) {
        let st = try loadStateWithMeta(path: resumePath)
        do {
            try vae.load(from: st.mats)
            savedEpoch = max(st.epoch, 0)   // число завершённых эпох в чекпоинте
            if let opt = st.optimizer {
                try vae.loadOptimizerMoments(opt.moments)
                adamStep = max(opt.step, 0)
                yumeLog("Веса и состояние оптимизатора загружены из \(resumePath) (дообучение, шаг Adam = \(adamStep), эпох пройдено = \(savedEpoch))")
            } else {
                resumeWithoutOptimizer = true
                savedEpoch = 0  // без моментов Adam не доверяем пропуску эпох — идём по всем
                yumeLog("⚠️ В чекпоинте \(resumePath) нет состояния оптимизатора (старый формат) — Adam начинается с нуля, первые \(resumeWarmupSteps) шагов идут с разгоном, чтобы не разрушить модель")
            }
            if let fp = st.fingerprint, !fp.isEmpty, fp != currentFingerprint {
                dataMatches = false
                savedEpoch = 0
                yumeLog("⚠️ Данные отличаются от тех, на которых учился чекпоинт (другой файл?) — гружу веса, но эпохи не пропускаю")
            }
        } catch let error where isIncompatibleCheckpoint(error) {
            // Файл не битый — он от другой архитектуры (например, латент был
            // 64 канала до перехода на вариационный путь). Падать на первом же
            // шаге незачем: начинаем с новых весов и говорим об этом прямо.
            yumeLog("⚠️ Чекпоинт \(resumePath) не совпадает по форме с текущей моделью:")
            yumeLog("   \((error as NSError).localizedDescription)")
            yumeLog("   Обучение начнётся с новых весов. Файл не тронут — при необходимости верните его из .backup.")
            savedEpoch = 0
            dataMatches = false
        }
    }
    return (adamStep, resumeWithoutOptimizer, savedEpoch, dataMatches)
}

