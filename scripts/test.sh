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
set -euo pipefail
cd "$(dirname "$0")/.."

SDK="$(xcrun --show-sdk-path)"
OUT="${TMPDIR:-/tmp}/yume_tests"

echo "== Компиляция тестовой сюиты =="
xcrun swiftc -O \
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

echo "== Запуск =="
"$OUT" "$@"
