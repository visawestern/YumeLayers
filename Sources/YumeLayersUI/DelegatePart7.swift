import AppKit
import AVKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import YumeLayersCore

extension YumeAppDelegate {
    func buildLeftPanel() -> NSStackView {
        let (folderRow, dataPage) = buildExtractSection()
        let paramsPage = buildParamsPage()
        let runPage = buildRunPage()
        let interpPage = buildInterpPage()

        for p in [dataPage, paramsPage, runPage, interpPage] {
            p.widthAnchor.constraint(equalToConstant: 392).isActive = true
            // Карточки не растягиваются по вертикали — держатся нативной высоты.
            p.setContentHuggingPriority(.required, for: .vertical)
        }

        let left = NSStackView(views: [dataPage, paramsPage, runPage, interpPage])
        left.orientation = .vertical
        left.spacing = 16
        left.alignment = .leading
        left.edgeInsets = NSEdgeInsets(top: 24, left: 24, bottom: 24, right: 24)
        left.widthAnchor.constraint(equalToConstant: 440).isActive = true

        layoutViews["dataPage"] = dataPage
        layoutViews["paramsPage"] = paramsPage
        layoutViews["runPage"] = runPage
        layoutViews["left"] = left
        layoutViews["folderRow"] = folderRow
        return left
    }

    /// Собирает карточку «1. Видео» и связанные контролы.
    private func buildExtractSection() -> (folderRow: NSStackView, page: NSView) {
        // Кнопка «Обзор»: открывает ту же панель выбора видео, что и
        // перетаскивание/«Выбрать». Раньше target/action не были привязаны —
        // кнопка не работала (регрессия UI).
        browseButton.target = self
        browseButton.action = #selector(chooseVideo)
        let folderRow = fieldRow("Файл:", folderField, accessory: browseButton,
                                 info: "Выберите один или несколько видеофайлов (MP4/MOV). Кадры будут извлечены строго в порядке выбора — это важно для привязки движения к seed.")

        extractButton.target = self
        extractButton.action = #selector(extractFrames)
        extractButton.bezelStyle = .rounded
        extractButton.controlSize = .large
        extractButton.image = NSImage(systemSymbolName: "camera.fill", accessibilityDescription: "Вытащить кадры")
        extractButton.imagePosition = .imageLeading
        extractButton.isEnabled = false

        stopExtractButton.target = self
        stopExtractButton.action = #selector(stopExtraction)
        stopExtractButton.bezelStyle = .rounded
        stopExtractButton.controlSize = .large
        stopExtractButton.image = NSImage(systemSymbolName: "stop.fill", accessibilityDescription: "Остановить извлечение")
        stopExtractButton.imagePosition = .imageLeading
        stopExtractButton.isEnabled = false

        let extractRow = NSStackView(views: [extractButton, stopExtractButton])
        extractRow.orientation = .horizontal
        extractRow.spacing = 10
        extractRow.alignment = .centerY

        let extractHint = NSTextField(wrappingLabelWithString: "Выберите видеофайл, затем нажмите «Вытащить кадры». Каждый кадр сразу появится ниже.")
        extractHint.font = NSFont.systemFont(ofSize: 11)
        extractHint.textColor = .tertiaryLabelColor
        extractHint.alignment = .left
        extractHint.widthAnchor.constraint(equalToConstant: 340).isActive = true

        // Прогресс извлечения — сразу под кнопками «Вытащить кадры» / «Стоп».
        extractProgressBlock = makeProgressBlock(bar: extractProgressBar,
                                                 percent: extractPercentLabel,
                                                 eta: extractEtaLabel)
        extractProgressBlock.widthAnchor.constraint(equalToConstant: 340).isActive = true

        let extractLimitRow = valueRow(
            "Кадров для теста:",
            extractLimitCtl,
            info: "Сколько первых кадров извлечь. 0 — все кадры. Лимит не меняет частоту: берутся первые N кадров подряд при fps исходника, как при полном извлечении, а не прореженная выборка по всему ролику. Нужен для быстрой проверки обучения на 2000 кадрах.")
        let dataPage = groupCard(title: "1. Видео",
                                 contentViews: [folderRow, extractLimitRow, extractRow,
                                                extractProgressBlock, extractHint],
                                 spacing: 12)
        return (folderRow: folderRow, page: dataPage)
    }

    /// Собирает карточку «2. Параметры обучения».
    private func buildParamsPage() -> NSView {
        // Шаг 2 — Параметры обучения
        resetParamsButton.target = self
        resetParamsButton.action = #selector(resetParams)
        resetParamsButton.bezelStyle = .rounded
        resetParamsButton.controlSize = .small
        resetParamsButton.widthAnchor.constraint(equalToConstant: 190).isActive = true
        autoTuneButton.target = self
        autoTuneButton.action = #selector(autoTuneNow)
        autoTuneButton.bezelStyle = .rounded
        autoTuneButton.controlSize = .small
        autoTuneButton.widthAnchor.constraint(equalToConstant: 190).isActive = true
        adaptiveSwitch.state = .on   // авто-пауза при перегреве включена по умолчанию
        // Ручное подтверждение значения отключает авто-подбор только этого поля:
        // второе поле продолжает подстраиваться под число кадров.
        epochsCtl.onCommit = { [weak self] _ in self?.epochsEditedByUser = true }
        stepsCtl.onCommit = { [weak self] _ in self?.stepsEditedByUser = true }
        lrCtl.onCommit = { [weak self] _ in self?.lrEditedByUser = true }
        let paramsPage = groupCard(title: "2. Параметры обучения",
                                   contentViews: [
            switchRow("Продолжить обучение", resumeSwitch,
                      tooltip: "Продолжить с чекпоинта: обучение начнётся с последнего сохранения"),
            valueRow("Эпохи:", epochsCtl,
                     info: "Сколько раз модель полностью пройдёт по всем кадрам набора. Больше эпох — лучше качество, но дольше обучение и выше нагрев GPU."),
            valueRow("Шагов за эпоху:", stepsCtl,
                     info: "Сколько обновлений весов выполняется за одну эпоху. Один шаг = обработка одного кадра + обновление весов."),
            valueRow("Размер батча:", batchCtl,
                     info: "Сколько кадров обрабатывается за один шаг обучения. Больший батч быстрее проходит эпоху при той же тепловой нагрузке: регулятор держит долю работы GPU, а не число шагов. Верхняя граница — по свободной памяти GPU."),
            valueRow("Ширина обучения:", widthCtl,
                     info: "Ширина кадра, до которой уменьшаются кадры при обучении. Исходные JPEG в памяти остаются полноразмерными. По умолчанию 80 — быстрый тест сходимости; 854 — полное разрешение (медленнее и хуже сходится на этой сети)."),
            valueRow("Высота обучения:", heightCtl,
                     info: "Высота кадра, до которой уменьшаются кадры при обучении. По умолчанию 56. Вместе с шириной должна быть кратна 8 (лишнее добивается краем без растяжения)."),
            valueRow("Скорость обучения:", lrCtl,
                     info: "Размер шага градиентного спуска (learning rate). Слишком большое значение — расходимость/NaN, слишком малое — очень медленное обучение. По умолчанию 0.0003."),
            valueRow("Пауза, мс:", pauseCtl,
                     info: "Фиксированная пауза после каждого шага, в миллисекундах. Работает только когда «Авто-пауза (нагрев)» выключена: тогда нагрузка постоянная и равна этой паузе. При включённом авто-режиме паузу считает регулятор по температуре, и это поле не используется."),
            valueRow("Нагрузка GPU, % (≤98):", powerCtl,
                     info: "Потолок нагрузки: сколько процентов времени GPU занята работой. Значение читается как есть (например, 98 — GPU занята 98% времени). Верхняя граница 98% — физический предел, при котором WindowServer получает слоты между нашими command buffer'ами; полностью непрерывная очередь приводила к перезагрузке 4 ноября 2026. Фактическую нагрузку регулятор подбирает по температуре: рабочая точка 82 °C, потолок 85 °C, при 86 °C — принудительное остывание."),
            switchRow("Авто-пауза (нагрев)", adaptiveSwitch,
                      tooltip: "Авто-пауза при перегреве: регулятор сам подбирает нагрузку по температуре GPU — держит максимум, который не ведёт к перезагрузке"),
            autoTuneButton,
            resetParamsButton,
        ], spacing: 10)
        return paramsPage
    }

    /// Собирает карточку «3. Запуск».
    private func buildRunPage() -> NSView {
        trainButton.target = self
        trainButton.action = #selector(startTraining as () -> Void)
        trainButton.keyEquivalent = "\r"
        trainButton.title = "▶ Обучить VAE"
        // Надёжный рендер: bordered-кнопка, тонированная bezelColor (violet).
        // Безрамный стиль с фоновым слоем (applyPill) не отрисовывался — кнопка
        // становилась невидимой. bezelColor гарантированно рисует заполнение.
        trainButton.bezelStyle = .rounded
        trainButton.isBordered = true
        trainButton.bezelColor = Design.violet
        trainButton.contentTintColor = .white
        trainButton.font = NSFont.systemFont(ofSize: 15, weight: .medium)
        trainButton.controlSize = .large

        stopButton.target = self
        stopButton.action = #selector(stopTraining)
        stopButton.keyEquivalent = "\u{1b}"   // Esc
        stopButton.title = "■ Стоп"
        stopButton.bezelStyle = .rounded
        stopButton.isBordered = true
        stopButton.controlSize = .regular
        stopButton.isEnabled = false

        let runButtons = NSStackView(views: [trainButton, stopButton])
        runButtons.orientation = .horizontal
        runButtons.spacing = 10
        runButtons.alignment = .centerY

        // Прогресс обучения и реконструкций — прямо под кнопками.
        trainProgressBlock = makeProgressBlock(bar: trainProgressBar,
                                               percent: trainPercentLabel,
                                               eta: trainEtaLabel)
        trainProgressBlock.widthAnchor.constraint(equalToConstant: 360).isActive = true

        // Живой индикатор под прогрессом обучения: спиннер + две строки
        // (состояние и реальный кадр/шаг). Скрыт вне обучения.
        trainSpinner.style = .spinning
        trainSpinner.controlSize = .small
        trainSpinner.isDisplayedWhenStopped = false
        trainSpinner.widthAnchor.constraint(equalToConstant: 16).isActive = true
        trainSpinner.heightAnchor.constraint(equalToConstant: 16).isActive = true

        trainStateLabel.font = NSFont.systemFont(ofSize: 11, weight: .semibold)
        trainStateLabel.textColor = .labelColor
        trainStateLabel.lineBreakMode = .byTruncatingTail

        trainFrameLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .regular)
        trainFrameLabel.textColor = .tertiaryLabelColor
        trainFrameLabel.lineBreakMode = .byTruncatingTail
        trainFrameLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        let trainLiveColumn = NSStackView(views: [trainStateLabel, trainFrameLabel])
        trainLiveColumn.orientation = .vertical
        trainLiveColumn.alignment = .leading
        trainLiveColumn.spacing = 2
        trainLiveColumn.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        trainLiveBlock = NSStackView(views: [trainSpinner, trainLiveColumn])
        trainLiveBlock.orientation = .horizontal
        trainLiveBlock.alignment = .centerY
        trainLiveBlock.spacing = 8
        trainLiveBlock.widthAnchor.constraint(equalToConstant: 360).isActive = true
        trainLiveBlock.isHidden = true
        reconsProgressBlock = makeProgressBlock(bar: reconsProgressBar,
                                                percent: reconsPercentLabel,
                                                eta: nil)
        reconsProgressBlock.widthAnchor.constraint(equalToConstant: 360).isActive = true

        let hint = NSTextField(wrappingLabelWithString: "Не запускайте несколько обучений одновременно. При перегреве система сама увеличит паузы.")
        hint.font = NSFont.systemFont(ofSize: 11)
        hint.textColor = .tertiaryLabelColor
        hint.widthAnchor.constraint(equalToConstant: 360).isActive = true

        // Шаг 3 — Запуск
        let runPage = groupCard(title: "3. Запуск",
                                contentViews: [
            runButtons,
            trainProgressBlock,
            trainLiveBlock,
            reconsProgressBlock,
            pathBlock("Файл VAE:", convPathCtl,
                      info: "Путь к файлу, в который сохраняется обученная модель (чекпоинт, safetensors). Можно указать любой файл, например ~/YumeLayers/checkpoints/my_vae.safetensors."),
            hint,
        ], spacing: 10)
        return runPage
    }

    /// Собирает карточку «4. Генерация A→B».
    private func buildInterpPage() -> NSView {
        interpAButton.target = self
        interpAButton.action = #selector(chooseReferenceA)
        interpAButton.bezelStyle = .rounded
        interpAButton.controlSize = .small
        interpBButton.target = self
        interpBButton.action = #selector(chooseReferenceB)
        interpBButton.bezelStyle = .rounded
        interpBButton.controlSize = .small
        interpRunButton.target = self
        interpRunButton.action = #selector(runInterpolation)
        interpStopButton.target = self
        interpStopButton.action = #selector(stopInterpolation)
        // Скриншот текущего выбранного кадра → опорная A/B.
        shotToAButton.target = self
        shotToAButton.action = #selector(shotToReferenceA)
        shotToAButton.bezelStyle = .rounded
        shotToAButton.controlSize = .small
        shotToBButton.target = self
        shotToBButton.action = #selector(shotToReferenceB)
        shotToBButton.bezelStyle = .rounded
        shotToBButton.controlSize = .small
        updateShotButtons()
        // При старте интерполяция не идёт — «Стоп» неактивна.
        // Активность переключает updateInterpolateButtons() при старте/завершении.
        interpStopButton.isEnabled = false
        interpResultLabel.font = NSFont.systemFont(ofSize: 11)
        interpResultLabel.textColor = .secondaryLabelColor
        interpResultLabel.widthAnchor.constraint(equalToConstant: 340).isActive = true
        let interpHint = NSTextField(wrappingLabelWithString:
            "A и B должны быть РАЗНЫМИ картинками — между ними строится движение. Промежуточные кадры — реконструкция модели (внутреннее разрешение 80×56).")
        interpHint.font = NSFont.systemFont(ofSize: 11)
        interpHint.textColor = .tertiaryLabelColor
        interpHint.widthAnchor.constraint(equalToConstant: 340).isActive = true
        interpProgressBlock = makeProgressBlock(bar: interpProgressBar,
                                                percent: interpPercentLabel, eta: nil)
        interpProgressBlock.widthAnchor.constraint(equalToConstant: 360).isActive = true
        interpProgressBlock.isHidden = true
        let interpButtons = NSStackView(views: [interpRunButton, interpStopButton])
        interpButtons.orientation = .horizontal
        interpButtons.spacing = 8
        let shotRow = NSStackView(views: [shotToAButton, shotToBButton])
        shotRow.orientation = .horizontal
        shotRow.spacing = 8
        shotRow.alignment = .centerY
        let interpPage = groupCard(title: "4. Генерация A→B",
                                   contentViews: [
            referenceRow("Кадр A:", interpAButton, interpALabel),
            referenceRow("Кадр B:", interpBButton, interpBLabel),
            shotRow,
            valueRow("Длительность, с:", interpSecondsCtl,
                     info: "Число кадров = длительность × кадров в секунду."),
            valueRow("Кадров в секунду:", interpFpsCtl,
                     info: "Частота кадров итогового MP4."),
            interpButtons,
            interpProgressBlock,
            interpResultLabel,
            interpHint,
        ], spacing: 10)
        return interpPage
    }

}
