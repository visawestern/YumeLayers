# YumeLayers

**YumeLayers** — экспериментальная модель генерации аниме-сцен по слоям (фон → объекты → свет) в латентном пространстве. **Чистый Swift, без PyTorch и других ML-фреймворков** — только Accelerate/vDSP + Metal (GPU, AMD Radeon Pro 5500M).

## Статус (честно)

| Компонент | Статус |
|---|---|
| Metal GPU-движок (private-буферы, тайловый matmul, свёрточные ядра) | ✅ работает: ~440 ГФЛОП/с, 61 ГБ/с |
| Позитивность всех GPU-операций vs CPU | ✅ max delta < 1e-7 |
| Численная проверка градиентов свёрточного VAE (конечные разности) | ✅ 100/100 (расхождение < 1%) |
| Обучение свёрточного VAE 480p (854×480 → латент 107×60×64) | ✅ работает, ~0.3 с/шаг |
| Тренировочный GUI (параметры в окне, выбор папки, resume, стоп, превью) | ✅ работает |
| MLP-прототипы (vae/diffusion/generate) | ✅ оставлены для сверки |
| Свёрточный VAE под 360p (640×360, старт ConvVAE.swift) | ⏳ запасной вариант |
| Текстовый conditioning | ⏳ после безусловной генерации |

## Запуск

```bash
cd /Users/apple/projects/YumeLayers
swift build -c release        # release обязателен (debug в 30–60× медленнее)

# GUI — параметры задаются в окне (рекомендуется):
open .build/release/YumeLayersUI

# CLI без флагов (дефолты: 854×480, 20 эпох, паузы для разгрузки GPU):
.build/release/YumeLayers conv
.build/release/YumeLayers check       # проверка GPU-операций
.build/release/YumeLayers convcheck   # проверка градиентов свёрток
```

## GUI (YumeLayersUI)

- **Папка с кадрами** — выбор через «Обзор…» (пусто = синтетические сцены)
- **Ширина/высота** кадра (по умолчанию 854×480)
- **Эпохи, шаги за эпоху, батч, скорость обучения, пауза** — задаются в окне
- **Продолжить с чекпоинта** — resume обучения (checkpoints/conv.json)
- **Обучить VAE** — запуск на Metal GPU (в фоне, интерфейс не блокируется)
- **Стоп** — остановка обучения
- Прогресс, лог и превью «цель | реконструкция» после каждой эпохи

## Структура

```
YumeLayers/
├── Package.swift
├── Sources/YumeLayersCore/      # библиотека: Metal, свёртки, обучение, проверки
│   ├── MetalEngine.swift        # шейдеры (matmul, conv, relu, adam…)
│   ├── GpuTraining.swift        # GPUTensor (private-память), GpuSession
│   ├── ConvGPU.swift            # свёрточный VAE (forward/backward на GPU)
│   ├── ConvTrainer.swift        # обучение 480p, прогресс, контроль остановки
│   ├── GpuTrainer.swift         # MLP-прототипы на GPU
│   ├── Matrix.swift, Models.swift, Train.swift, ConvVAE.swift
│   └── Common.swift, LegacyCLI.swift
├── Sources/YumeLayers/          # CLI (тонкий диспатчер)
├── Sources/YumeLayersUI/        # GUI (AppKit)
├── docs/                        # план и критика (адвокат дьявола)
├── outputs/                     # превью «цель | реконструкция» по эпохам
└── checkpoints/                 # веса (JSON)
```

## Следующие шаги

1. Доработка качества реконструкции 480p (больше эпох на реальных кадрах)
2. Диффузия на латентном пространстве свёрточного VAE (GPU)
3. Слои: фон → объекты → свет с раздельными латентными каналами
4. Текстовый conditioning

Подробный план — `docs/PLAN.md`, критика архитектуры — `docs/CRITIQUE.md`.
