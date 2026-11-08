// YumeLayers UI — профессиональная панель обучения на Swift/AppKit.
// Параметры задаются в окне; обучение на Metal GPU; защита от перегрева.
import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers

import YumeLayersCore

final class YumeAppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    // Поля параметров
    let folderField = NSTextField(string: "")
    let widthField = NSTextField(string: "854")
    let heightField = NSTextField(string: "480")
    let extractLimitCtl = EditableValueControl(value: "0")
    let epochsCtl = EditableValueControl(value: "20")
    let stepsCtl = EditableValueControl(value: "20")
    let batchCtl = EditableValueControl(value: "1")
    let lrCtl = EditableValueControl(value: "0.0003")
    let pauseCtl = EditableValueControl(value: "250")
    let powerCtl = EditableValueControl(value: "90")
    /// Размер кадра, на котором обучается VAE. Исходные JPEG остаются
    /// полноразмерными — декодер уменьшает их на лету (как старый стенд
    /// 80×56, на котором сеть сходилась заметно лучше 480p).
    let widthCtl = EditableValueControl(value: "80")
    let heightCtl = EditableValueControl(value: "56")
    let convPathCtl = EditableValueControl(value: "~/YumeLayers/checkpoints/conv.safetensors", width: 0)
    let resumeSwitch = NSSwitch()
    let adaptiveSwitch = NSSwitch()

    let trainButton = NSButton(title: "Обучить VAE", target: nil, action: nil)
    let stopButton = NSButton(title: "Стоп", target: nil, action: nil)
    let browseButton = NSButton(title: "Обзор", target: nil, action: nil)

    // Статус, прогресс, превью, лог
    let statusLabel = NSTextField(labelWithString: "Готово к обучению")
    // Прогресс каждого действия живёт рядом с его кнопкой (см. карточки 1 и 3).
    let extractProgressBar = NSProgressIndicator()
    let extractPercentLabel = NSTextField(labelWithString: "0%")
    let extractEtaLabel = NSTextField(labelWithString: "—")
    var extractProgressBlock: NSStackView!
    let trainProgressBar = NSProgressIndicator()
    let trainPercentLabel = NSTextField(labelWithString: "0%")
    let trainEtaLabel = NSTextField(labelWithString: "—")
    var trainProgressBlock: NSStackView!
    // Живой индикатор: спиннер + состояние + текущий кадр/шаг/скорость.
    let trainSpinner = NSProgressIndicator()
    let trainStateLabel = NSTextField(labelWithString: "Обучение идёт")
    let trainFrameLabel = NSTextField(labelWithString: "Ожидание первого шага…")
    var trainLiveBlock: NSStackView!
    var trainHeartbeatTimer: Timer?
    var lastTrainStepAt: Date?
    var lastTrainReport: TrainStepReport?
    let reconsProgressBar = NSProgressIndicator()
    let reconsPercentLabel = NSTextField(labelWithString: "0%")
    var reconsProgressBlock: NSStackView!
    let epochCard = MetricCard(title: "Эпоха", symbol: "clock")
    let lossCard = MetricCard(title: "Loss", symbol: "chart.line.downtrend.xyaxis")
    let psnrCard = MetricCard(title: "PSNR, дБ", symbol: "speedometer")
    let chart = HistoryChart()
    let previewView = NSImageView()
    let playerView = AVPlayerView()
    let queuePlayer = AVQueuePlayer()
    var looper: AVPlayerLooper?
    var videoEmptyState: NSStackView!
    var videoURLs: [URL] = []
    var isExtracting = false
    var currentExtractor: FrameExtractor?
    var extractCancelled = false
    var extractedCount = 0
    var skippedBlackTotal = 0
    let extractButton = NSButton(title: "Вытащить кадры", target: nil, action: nil)
    let stopExtractButton = NSButton(title: "Стоп", target: nil, action: nil)
    let resetParamsButton = NSButton(title: "Сброс к умолчаниям", target: nil, action: nil)
    let autoTuneButton = NSButton(title: "Авто-расчёт параметров", target: nil, action: nil)
    var extractStartTime = Date()
    var trainStartTime = Date()
    var lastCollageRender = Date.distantPast

    // Асинхронная выдача извлечённых кадров: фон складывает кадры в буфер,
    // main thread забирает их пачками по таймеру. Так на 44 000 кадров
    // не приходится 44 000 обновлений интерфейса.
    struct PendingExtractFrame {
        let name: String
        let jpeg: Data
        let source: String
        let isControl: Bool
    }
    let extractBatch = ExtractBatchPolicy()
    let pendingExtractLock = NSLock()
    var pendingExtractFrames: [PendingExtractFrame] = []
    var pendingExtractProgress: Double = 0
    var pendingExtractDone = 0
    var extractTotalAll = 0
    var extractFlushTimer: Timer?
    var lastExtractFlush = Date.distantPast
    var flushedFirstExtractBatch = false
    let carouselPrev = NSButton(image: NSImage(systemSymbolName: "chevron.left", accessibilityDescription: "Назад")!, target: nil, action: nil)
    let carouselNext = NSButton(image: NSImage(systemSymbolName: "chevron.right", accessibilityDescription: "Вперёд")!, target: nil, action: nil)
    let carouselFirst = NSButton(image: NSImage(systemSymbolName: "chevron.left.to.line", accessibilityDescription: "В начало")!, target: nil, action: nil)
    let carouselLast = NSButton(image: NSImage(systemSymbolName: "chevron.right.to.line", accessibilityDescription: "В конец")!, target: nil, action: nil)
    var pageDots: [NSButton] = []
    /// Ограничения ширины кружочков-страниц (обновляются под число цифр).
    var dotWidthConstraints: [NSLayoutConstraint] = []
    let dotPoolSize = 30   // запас кнопок; видимое число считается по ширине
    let paginatorRow = NSStackView()
    var framesStore: [MemoryFrame] = []
    /// Контрольная выборка (1 кадр/сек по каждому видео) — для визуальной
    /// оценки качества реконструкций на итоговой модели. Живёт отдельно и не
    /// освобождается по ходу обучения.
    var controlFramesStore: [MemoryFrame] = []
    var processedFrames: [Bool] = []
    /// Реконструкции в памяти: имя кадра → PNG-данные (без записи на диск).
    var reconMemory: [String: Data] = [:]
    var carouselPage = 0
    let pageSize = 10   // 5 внизу + 5 справа вокруг большой
    let collageBig = NSImageView()
    let collageBigBadge = NSImageView()
    var collageThumbs: [NSButton] = []
    var thumbBadges: [NSImageView] = []
    var selectedFrameIndex = -1
    let collageEmpty = NSTextField(labelWithString: "Кадры появятся после извлечения видео")
    let sourceSwitch = NSSegmentedControl(labels: ["Оригинал", "Реконструкция"], trackingMode: .selectOne, target: nil, action: nil)
    let framesChip = CountChip()
    var showRecon = false
    var epochsEditedByUser = false
    var stepsEditedByUser = false
    var lrEditedByUser = false
    let videoListScroll = NSScrollView()
    let videoListStack = NSStackView()

    /// Порядок кадров в L: нижний ряд 0-4 (слева → справа), затем правый
    /// столбец 5-9 (снизу → вверх, конец — правый верхний).
    let carouselStack = NSStackView()
    let carouselScroll = NSScrollView()
    var carouselPlaceholder: NSStackView!
    let logView = NSTextView()

    var isTraining = false
    /// Идут ли реконструкции в фоне (управляются кнопкой «Стоп»).
    var isReconstructing = false
    var lastEpoch = 0
    let api = ApiServer()
    var layoutViews: [String: NSView] = [:]
    var debugGrids: [NSView] = []
    var lastParams = Params()
    /// Кратность ширины VAE для следующего обучения (по умолчанию 1).
    var requestedModelWidth = 1
    /// Число каналов латента для следующего обучения.
    var requestedLatentChannels = 16

    // ——— Генерация A→B: интерполяция между двумя опорными картинками ———
    let interpAButton = NSButton(title: "Выбрать…", target: nil, action: nil)
    let interpBButton = NSButton(title: "Выбрать…", target: nil, action: nil)
    let interpALabel = NSTextField(labelWithString: "не выбран")
    let interpBLabel = NSTextField(labelWithString: "не выбран")
    let interpSecondsCtl = EditableValueControl(value: "5")
    let interpFpsCtl = EditableValueControl(value: "24")
    let interpRunButton = NSButton(title: "Создать видео", target: nil, action: nil)
    let interpStopButton = NSButton(title: "Стоп", target: nil, action: nil)
    let interpProgressBar = NSProgressIndicator()
    let interpPercentLabel = NSTextField(labelWithString: "0%")
    var interpProgressBlock: NSStackView!
    let interpResultLabel = NSTextField(wrappingLabelWithString: "")
    var interpImageAPath = ""
    var interpImageBPath = ""
    var isInterpolating = false
    /// Чекпоинт, которым сделана последняя интерполяция.
    var lastInterpolationModel = ""

    // Плеер для ГОТОВОГО видео (результат интерполяции) + вкладки.
    let resultPlayerView = AVPlayerView()
    let resultQueuePlayer = AVQueuePlayer()
    var resultLooper: AVPlayerLooper?
    var resultEmptyState: NSStackView!
    let mediaTabsControl = NSSegmentedControl(
        labels: ["Входящее видео", "Готовое видео"],
        trackingMode: .selectOne, target: nil, action: nil)
    var videoMediaCard: NSView!
    var resultMediaCard: NSView!














    /// Разворачивает окно на весь рабочий стол при старте.
    ///
    /// Это НЕ нативный полноэкранный режим: меню-бар сверху и Dock остаются на
    /// месте, окно просто занимает `visibleFrame` — ровно то же, что даёт
    /// зелёная кнопка (или «Увеличить» в меню «Окно»). `visibleFrame` уже
    /// исключает полосу меню и Dock, поэтому окно ничего не перекрывает.


    // MARK: - Стандартное главное меню (HIG)


    // MARK: - Вспомогательное


    /// Строка формы, которая растягивается на всю ширину карточки:
    /// [надпись | ⓘ | …поле…] — поле занимает всё расстояние от кнопки
    /// info до правого края строки/колонки.
    final class StretchRow: NSStackView {}

    /// Строка формы: [надпись | ⓘ | …поле…] — кнопка info сразу после
    /// надписи, поле растягивается на всё расстояние до правого края.
    /// `info` — полное описание для выпадающей карточки.
    let labelW: CGFloat = 141
    let rowGap: CGFloat = 6
    let accW: CGFloat = 76
    let infoW: CGFloat = 16


    /// Строка с редактируемым значением: [надпись | ⓘ | …поле…].
    /// Кнопка info — сразу после надписи, значение/поле занимают всё
    /// расстояние от неё до правого края строки.


    /// Компактная строка переключателя (как в web-admin-panels): switch + лейбл.
    /// Лейбл — в той же колонке, что и у строк с числами, переключатель —
    /// в колонке значений (по правому краю). Единая сетка карточки.

    /// Блок пути: подпись с info на первой строке, поле — на отдельной
    /// строке во всю ширину карточки, чтобы длинный путь был виден целиком.

    // MARK: - Кнопка info с выпадающей карточкой-описанием
    var infoTexts: [String] = []
    var openPopovers: [Int: NSPopover] = [:]



    // MARK: - UI

    func buildUI() -> NSView {
        let left = buildLeftPanel()
        return buildRemaining(left: left)
    }


    // MARK: - Действия

    /// Выбор одного или нескольких видеофайлов (MP4/MOV).

    /// Единая регистрация выбранных видео: список справа, поле «Файл:»,
    /// активность кнопки извлечения, показ первого видео. Используется и UI,
    /// и API — чтобы оба пути вели себя одинаково.

    /// Блок прогресса действия: [полоса | проценты | ETA]. Один вид для
    /// извлечения кадров, обучения и реконструкций. Ширина задаётся снаружи
    /// под карточку, полоса тянется на свободное место.

    /// Прогресс извлечения — под кнопкой «Вытащить кадры».

    /// Прогресс обучения — под кнопкой «Обучить VAE».

    /// Раз в секунду проверяем, живой ли тренировочный цикл: если отчётов
    /// о шагах давно нет, спиннер останавливается и строка честно говорит,
    /// сколько секунд нет прогресса.



    /// Прогресс реконструкций — отдельная строка под прогрессом обучения.

    /// Обновление прогноза времени (ETA) рядом с прогрессбаром действия.
    /// Считается линейно: (затраченное / сделано) × осталось.


    /// Текущий тепловой уровень macOS (для /api/state и контроля нагрева).

    /// Рендер списка выбранных видео в правой колонке (имя + удалить).

    /// Удаление видео из выбранного списка (кнопка × в правой колонке).








    /// Переключение страницы пагинатора (по 9 кадров: большая + 9 малых).

    /// Клик по кружочку-странице.

    /// Сколько кружочков реально влезает в строку пагинатора.
    /// Считаем от фактической ширины строки (колонка минус отступы 24+24),
    /// а не от ширины самой строки: иначе кружочки распирают строку, строка
    /// «видит» большую ширину и показывает ещё больше кружков — колонка растёт.

    /// Ширина центральной колонки из фактической ширины окна:
    /// (контент − левая колонка 440 − два разделителя по 1) / 2.

    /// Рендер кружочков-страниц: столько, сколько влезает по ширине,
    /// активная выделена. Ширина кружочка адаптивная: под число цифр
    /// (например «1000» не влезает в 28 pt — расширяем и делаем овал).

    /// Извлечение кадров из одного или нескольких MP4.
    /// Кадры нумеруются строго по порядку файлов (глобально), чтобы движение
    /// можно было привязать к seed'у.
    /// - `append`: false — очистить старые кадры; true — продолжить нумерацию с текущих.
    /// - `maxFrames`: лимит кадров (0 — авто от доступной памяти).

    /// Финализация извлечения: общий статус, кнопки, лог.

    /// Кладём кадр в буфер, который main thread забирает пачками.
    /// Вызывается из фонового потока извлечения — без обновления UI.



    /// Таймер на main thread забирает накопленные кадры пачками.
    /// 200 мс достаточно, чтобы первые кадры появились почти сразу,
    /// и при этом на 44 000 кадров не приходится 44 000 обновлений UI.


    /// Забирает пачку из буфера и добавляет кадры в интерфейс.
    /// - Parameter force: забрать всё, даже неполную пачку (финализация/остановка).

    /// Добавление пачки кадров в память. `processed` = true только когда кадр
    /// обработан моделью (тогда ставится галочка ✓).


    /// Путь к реконструкции того же кадра (recons/frame_XXXX.jpg).
    /// PNG-данные реконструкции кадра из памяти (nil, если ещё нет).

    /// Рендер коллажа: большая (выбранная) + 9 малых текущей страницы.

    /// Обновляет счётчик кадров и пагинацию, не трогая картинки.
    /// Используется во время извлечения: новые кадры сразу доступны
    /// по страницам, но коллаж не перерисовывается на каждый кадр.

    /// Клик по малому кадру — ставит его в центр коллажа.

    /// Рендер собственного окна в PNG (для диагностики интерфейса).


    /// Запуск обучения. `maxFrames` > 0 — учиться только на первых N кадрах
    /// (для тестов вместо полного датасета). 0 — все кадры.
    /// `resume` = true — продолжить с существующего чекпоинта без диалога
    /// (используется API; в UI решение принимает человек).

    /// Единая точка входа обучения: и кнопка «Обучить», и API вызывают её.
    /// Кадры для обучения всегда берутся из памяти (единый источник).
    /// `maxFrames` > 0 — обучаться только на первых N кадрах.


    /// Авто-настройка «Эпохи»/«Шагов за эпоху» под фактическое число кадров.
    /// Поле подстраивается, пока пользователь не подтвердил своё значение
    /// вручную: шаги = число кадров (полный проход за эпоху),
    /// эпохи = 3…10 полных проходов (около 3000 сэмплов суммарно).
    /// Повторный запуск извлечения пересчитывает авто-поля под новый размер очереди.

    /// Кнопка «Авто-расчёт параметров»: пересчёт под текущие кадры.




    /// Строка из core-кода (yumeLog): уже с меткой времени и уже записана в
    /// train.log — здесь только показываем её в журнале и отдаём через API.
}

let app = NSApplication.shared
app.appearance = NSAppearance(named: .aqua)
setbuf(stdout, nil)   // немедленный вывод логов API в терминал/файл
let delegate = YumeAppDelegate()
app.delegate = delegate
app.run()
