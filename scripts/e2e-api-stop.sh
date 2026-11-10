#!/bin/bash
# YumeLayers — E2E через API: ПОКАДРОВАЯ остановка обучения по /api/stop.
#
# Сквозная live-проверка цепочки: /api/stop → api.onStop → stopTraining() →
# TrainingControl.cancel() → цикл trainConvVAE видит isCancelled на следующем
# шаге и обрывается (фикс покадрового «Стоп»). Отдельно от e2e-api.sh, чтобы не
# утяжелять основной happy-path и явно проверить именно прерывание.
#
# Использование: scripts/e2e-api-stop.sh [BASE_URL]
set -euo pipefail
BASE="${1:-http://127.0.0.1:17851}"
VIDEO="/Users/apple/Downloads/8446035036876.mp4"

fail() { echo "FAIL: $1"; exit 1; }
ok()   { echo "ok: $1"; }
# Чистый shell-парсер скалярного поля плоского JSON (без Python).
jget() { grep -o "\"$1\":[^,}]*" | head -n 1 | cut -d: -f2- | tr -d ' "'; }

# Ждём, пока поле примет значение (или, для want=any-true, станет true).
wait_field() { # $1=поле $2=значение $3=макс_сек
  local field="$1" want="$2" max="$3" i=0 v=""
  while [ "$i" -lt "$max" ]; do
    v=$(curl -s --max-time 3 "$BASE/api/status" | jget "$field")
    [ "$v" = "$want" ] && return 0
    sleep 2; i=$((i+2))
  done
  return 1
}

# 1) API должен быть свободен.
curl -s --max-time 3 "$BASE/api/state" | grep -q '"training":false' || fail "API занят/недоступен"
ok "API up, training=false"

# 2) Долго не научим специально: много эпох×шагов, чтобы успеть остановить в середине.
curl -s -X POST "$BASE/api/set" -H "Content-Type: application/json" \
  -d '{"width":"128","height":"72","epochs":"40","steps":"400","lr":"0.0003","lossKind":"mse"}' \
  | grep -q '"ok":true' || fail "set"
ok "set params (epochs=40, steps=400 → длинный прогон)"

# 3) Кадры для обучения (если ещё не загружены в прошлых e2e — извлечём заново).
curl -s -X POST "$BASE/api/extract" -H "Content-Type: application/json" \
  -d "{\"path\":\"$VIDEO\",\"maxFrames\":200}" | grep -q '"ok":true' || fail "extract старт"
wait_field extracting False 180 || fail "таймаут извлечения"
ok "extract 200 кадров"

# 4) Старт обучения (долгое). Ждём, пока training станет true и пойдёт прогресс.
curl -s -X POST "$BASE/api/train" -H "Content-Type: application/json" \
  -d '{"maxFrames":200,"resume":false}' | grep -q '"ok":true' || fail "train старт"
wait_field training True 60 || fail "обучение не стартовало (training!=true)"
ok "обучение запущено (training=true)"

# 5) Даём几步 пробежать, затем STOP.
sleep 5
curl -s -X POST "$BASE/api/stop" | grep -q '"ok":true' || fail "stop не принят"
ok "POST /api/stop отправлен"

# 6) Покадровый обрыв: training должна стать false быстро (за grace watchdog не убьёт
#    процесс, т.к. цикл обрывается сам на следующем шаге).
if ! wait_field training False 40; then
  fail "после /api/stop обучение не остановилось за 40 с (нет покадровой проверки? зависло?)"
fi
ok "обучение остановлено по /api/stop (training=false)"

# 7) Проверяем, что оборвалось ДОЗАПЕБЕЛЬНО: прогресс/эпоха не дошли до финала.
EPOCH=$(curl -s --max-time 3 "$BASE/api/status" | jget epoch)
PROG=$(curl -s --max-time 3 "$BASE/api/status" | jget progress)
echo "epoch=$EPOCH progress=$PROG"
awk -v p="${PROG:-1}" 'BEGIN{ exit !(p+0 < 1.0) }' \
  || fail "прогресс дошёл до 1.0 — остановка не прервала прогон посреди (progress=$PROG)"
ok "оборвано посреди прогона (progress=$PROG < 1.0, epoch=$EPOCH) → покадровый «Стоп» работает сквозно через HTTP"

echo "E2E API STOP: PASS"
