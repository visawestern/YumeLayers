#!/bin/bash
# Сборка YumeLayers.app напрямую через swiftc (без SwiftPM).
# Нужна, когда `swift build` недоступен: SwiftPM требует sandbox-exec,
# который в этом окружении запрещён.
set -euo pipefail
cd /Users/apple/projects/YumeLayers

SDK=$(xcrun --show-sdk-path)
APP="YumeLayers.app"
MOD=/tmp/ylapp-mod

rm -rf "$APP" "$MOD"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Frameworks" "$MOD"

CORE_FRAMEWORKS=(
  -framework Accelerate -framework AVFoundation -framework ImageIO
  -framework CoreGraphics -framework Metal -framework VideoToolbox
  -framework UniformTypeIdentifiers -framework IOKit
)

echo "== Ядро → dylib =="
xcrun swiftc -O -module-name YumeLayersCore -emit-module -emit-library \
  -sdk "$SDK" \
  -emit-module-path "$MOD/YumeLayersCore.swiftmodule" \
  -o "$APP/Contents/Frameworks/libYumeLayersCore.dylib" \
  Sources/YumeLayersCore/*.swift \
  "${CORE_FRAMEWORKS[@]}" \
  -Xlinker -install_name -Xlinker @rpath/libYumeLayersCore.dylib

echo "== Интерфейс → исполняемый файл =="
xcrun swiftc -O -sdk "$SDK" \
  -I "$MOD" -L "$APP/Contents/Frameworks" -lYumeLayersCore \
  -o "$APP/Contents/MacOS/YumeLayersUI" \
  Sources/YumeLayersUI/*.swift \
  -framework AppKit -framework Network -framework Metal \
  -Xlinker -rpath -Xlinker @executable_path/../Frameworks

echo "== Info.plist + иконка =="
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

[ -f assets/AppIcon.icns ] && cp assets/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

echo "== Подпись (ad-hoc) =="
codesign --force --deep --sign - "$APP" 2>&1 | tail -2 || true

echo "Готово: $(pwd)/$APP"
