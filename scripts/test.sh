#!/bin/bash
# Сборка и запуск тестовой сюиты YumeLayers.
#
# Почему не `swift test`: XCTest на macOS запускается только через бандл SwiftPM,
# а нам нужен доступ к internal-API ядра и возможность гонять тесты в любом
# окружении (в т.ч. в песочнице). Поэтому тесты компилируются ВМЕСТЕ с ядром
# одним модулем — так они видят internal, как @testable import.
#
# Использование:
#   scripts/test.sh            # все тесты
#   scripts/test.sh --filter Grad   # только тесты, в имени которых есть "Grad"
#   scripts/test.sh --rebuild  # принудительно пересобрать перед запуском
#
# Компиляция кэшируется: бинарник пересобирается только если изменились
# исходники ядра/тестов или сам этот скрипт. Поэтому повторные прогоны
# с разными --filter не тратят минуту на компиляцию.
set -euo pipefail
cd "$(dirname "$0")/.."

SDK="$(xcrun --show-sdk-path)"
OUT="${TMPDIR:-/tmp}/yume_tests"

FORCE=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--rebuild" ]]; then FORCE=1; else ARGS+=("$a"); fi
done

NEED_BUILD=$FORCE
if [[ $NEED_BUILD -eq 0 ]]; then
  if [[ ! -x "$OUT" ]]; then
    NEED_BUILD=1
  elif [[ -n "$(find Sources/YumeLayersCore Tests/YumeLayersTests -name '*.swift' -newer "$OUT" -print -quit)" ]]; then
    NEED_BUILD=1
  elif [[ scripts/test.sh -nt "$OUT" ]]; then
    NEED_BUILD=1
  fi
fi

if [[ $NEED_BUILD -eq 1 ]]; then
  echo "== Компиляция тестовой сюиты =="
  JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  xcrun swiftc -O -j "$JOBS" \
    -sdk "$SDK" \
    -o "$OUT" \
    Sources/YumeLayersCore/*.swift \
    Tests/YumeLayersTests/*.swift \
    -framework Accelerate \
    -framework AVFoundation \
    -framework ImageIO \
    -framework CoreGraphics \
    -framework Metal \
    -framework VideoToolbox \
    -framework UniformTypeIdentifiers \
    -framework IOKit
else
  echo "== Компиляция не нужна (исходники не менялись) =="
fi

echo "== Запуск =="
"$OUT" ${ARGS[@]+"${ARGS[@]}"}
