# YumeLayers — фиксированные ассеты для тестов создания видео

> Записано по просьбе пользователя (2026-11-09): «запомни эти файлы для создания
> видео … использовать каждый раз для тестов создания видео в 3 секунды».

## Канонический набор (реальные файлы пользователя)

| Роль | Путь |
|---|---|
| Опорная картинка A (начало ролика) | `/Users/apple/Downloads/Screenshot 2026-10-20 at 19.12.07.png` |
| Опорная картинка B (конец ролика) | `/Users/apple/Downloads/Screenshot 2026-10-20 at 19.15.29.png` |
| Чекпоинт обученного VAE | `~/YumeLayers/checkpoints/conv.safetensors` (через `bestCheckpointPath`) |

Подтверждено: оба PNG существуют (по ~1.6–1.9 МБ, 2532×1170), чекпоинт
создаётся обучением через `POST /api/train`.

## Эталонный рецепт: видео 3 секунды

- `seconds = 3`, `fps = 24` → **72 кадра** (`interpolationFrameCount(3, 24) = 72`).
- Выход: 2532×1170 (как у опорной картинки A, чётные стороны).
- Длительность MP4 ≈ 3.0 с (проверяется `ffprobe`).

### Живой прогон через API

```bash
curl -s -X POST http://127.0.0.1:17851/api/interpolate \
  -H "Content-Type: application/json" \
  -d '{
    "imageA": "/Users/apple/Downloads/Screenshot 2026-10-20 at 19.12.07.png",
    "imageB": "/Users/apple/Downloads/Screenshot 2026-10-20 at 19.15.29.png",
    "seconds": 3,
    "fps": 24,
    "out": "/tmp/yume_test_3s.mp4"
  }'
```

### Проверка результата

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,avg_frame_rate,duration \
  -of default=noprint_wrappers=1 /tmp/yume_test_3s.mp4
# Ожидание: nb_frames=72, avg_frame_rate=24/1, duration≈3.0
```

## Регрессия в сюите

- `testInterpolateVideo` — сквозной путь (1 с × 12 fps = 12 кадров, длительность
  MP4 ≈ 1 с через `AVURLAsset`).
- `testInterpolationThreeSecondsVideo` — фиксирует ровно жалобу пользователя:
  **3 с × 24 fps = 72 кадра**, длительность MP4 ≈ 3.0 с (допуск 0.25 с).
- Юнит: `testInterpolationPlan` — `interpolationFrameCount(2, 24) = 48` и т.д.

## Как использовать каждый раз

1. Для быстрой проверки «видео создаётся корректно» — `curl` из «Живой прогон»
   с `seconds=3, fps=24`, затем `ffprobe` (должно быть 72 кадра).
2. Для автоматической регрессии — тест `testInterpolationThreeSecondsVideo`.
3. Кадры извлекаются из тех же путей (`/Users/apple/Downloads/…`) при
   необходимости: они остаются источником истины для ручных тестов.
