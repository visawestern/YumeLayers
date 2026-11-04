#!/bin/bash
# Сборка настоящего macOS-приложения YumeLayers.app (с иконкой, Info.plist, подписью).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Сборка release =="
swift build -c release

BIN=".build/release/YumeLayersUI"
APP="YumeLayers.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# Иконка
if [ ! -f "assets/AppIcon.icns" ]; then
    echo "== Генерация иконки =="
    mkdir -p assets
    swift assets/make_icon.swift
    ICONSET="/tmp/YumeLayers.iconset"
    rm -rf "$ICONSET"; mkdir -p "$ICONSET"
    for sz in 16 32 128 256 512; do
        d=$((sz * 2))
        sips -z "$sz" "$sz" assets/AppIcon.png --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null
        sips -z "$d" "$d" assets/AppIcon.png --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
    done
    iconutil -c icns "$ICONSET" -o assets/AppIcon.icns
fi

cp "$BIN" "$APP/Contents/MacOS/YumeLayersUI"
cp assets/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>YumeLayersV2</string>
    <key>CFBundleDisplayName</key><string>YumeLayersV2</string>
    <key>CFBundleIdentifier</key><string>com.yumelayers.v2</string>
    <key>CFBundleExecutable</key><string>YumeLayersUI</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST

codesign --force --deep --sign - "$APP"
echo "Готово: $(pwd)/$APP"
