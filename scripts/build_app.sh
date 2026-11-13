#!/bin/sh
# Build dist/MacEcho.app: a thin bundle that runs the menu bar app from this checkout's .venv.
# ponytail: not standalone (needs this repo + .venv in place); py2app if it must be shareable.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/MacEcho.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>MacEcho</string>
  <key>CFBundleDisplayName</key><string>MacEcho</string>
  <key>CFBundleIdentifier</key><string>local.macecho.menubar</string>
  <key>CFBundleExecutable</key><string>MacEcho</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>MacEcho listens to your voice so it can reply.</string>
</dict>
</plist>
EOF

clang -O2 -DROOT="\"$ROOT\"" -o "$APP/Contents/MacOS/MacEcho" "$ROOT/scripts/launcher.c"

codesign --force --sign - "$APP" 2>/dev/null || true  # ad-hoc sign so macOS remembers mic permission
echo "Built $APP"
