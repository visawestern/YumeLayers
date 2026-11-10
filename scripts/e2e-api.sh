#!/bin/bash
# YumeLayers — E2E через API (тождественно кликам по кнопкам UI).
#
# Маршруты /api/* вызывают ТЕ ЖЕ функции, что и кнопки окна:
#   /api/extract    → startExtraction (кнопка «Извлечь кадры»)
#   /api/train      → startTrainingInternal (кнопка «Обучить VAE»)
#   /api/interpolate→ startInterpolation (кнопка «Создать видео»)
# Поэтому E2E-проверку можно гонять по API, не требуя прав Accessibility
# для программных кликов.
#
# Использование: scripts/e2e-api.sh [BASE_URL]
set -euo pipefail
BASE="${1:-http://127.0.0.1:17851}"
VIDEO="/Users/apple/Downloads/8446035036876.mp4"
IMG_A="/Users/apple/Downloads/Screenshot 2026-10-20 at 19.12.07.png"
IMG_B="/Users/apple/Downloads/Screenshot 2026-10-20 at 19.15.29.png"
OUT="/tmp/yume_e2e_api.mp4"

fail() { echo "FAIL: $1"; exit 1; }
ok()   { echo "ok: $1"; }
# Чистый shell: достаёт скалярное поле из плоского JSON-ответа API.
# Без Python/внешних интерпретаторов — grep по `"key":value`, затем trim.
jq_get() { grep -o "\"$1\":[^,}]*" | head -n 1 | cut -d: -f2- | tr -d ' "'; }
wait_until() { # $1=поле json, $2=ожидаемое значение, $3=макс секунд
  local field="$1" want="$2" max="$3" i=0 v=""
  while [ $i -lt "$max" ]; do
    v=$(curl -s --max-time 3 "$BASE/api/status" | jq_get "$field")
    [ "$v" = "$want" ] && return 0
    sleep 5; i=$((i+5))
  done
  fail "таймаут ожидания $field=$want (последнее: $v)"
}

curl -s --max-time 3 "$BASE/api/state" | grep -q '"training":false' || fail "API недоступен/занят"
ok "API up"

curl -s -X POST "$BASE/api/set" -H "Content-Type: application/json" \
  -d '{"width":"128","height":"72","epochs":"1","steps":"200","lr":"0.0003","lossKind":"mse","edgeWeight":"0.5","residual":true,"groupNorm":false}' \
  | grep -q '"ok":true' || fail "set"
ok "set params"

curl -s -X POST "$BASE/api/extract" -H "Content-Type: application/json" \
  -d "{\"path\":\"$VIDEO\",\"maxFrames\":200}" | grep -q '"ok":true' || fail "extract старт"
wait_until extracting False 120
ok "extract 200 кадров"

curl -s -X POST "$BASE/api/train" -H "Content-Type: application/json" \
  -d '{"maxFrames":200,"resume":false}' | grep -q '"ok":true' || fail "train старт"
wait_until training False 300
PSNR=$(curl -s --max-time 3 "$BASE/api/status" | jq_get psnr)
echo "psnr = $PSNR дБ"
awk -v v="$PSNR" 'BEGIN{ exit !(v+0 > 0) }' || fail "psnr > 0"

curl -s -X POST "$BASE/api/interpolate" -H "Content-Type: application/json" \
  -d "{\"imageA\":\"$IMG_A\",\"imageB\":\"$IMG_B\",\"seconds\":3,\"fps\":24,\"out\":\"$OUT\"}" \
  | grep -q '"ok":true' || fail "interpolate старт"
wait_until interpolating False 120
ok "interpolate"

FF=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,avg_frame_rate,duration \
     -of default=noprint_wrappers=1 "$OUT" 2>/dev/null || true)
echo "$FF" | grep -q "nb_frames=72" || fail "ожидалось nb_frames=72: $FF"
echo "$FF" | grep -q "avg_frame_rate=24/1" || fail "ожидалось 24 fps: $FF"
ok "mp4: 72 кадра @ 24 fps"

echo "E2E API: PASS"
