#!/bin/sh
# Build dist/MacEcho.app: a thin bundle that runs the app (Dock icon, window, menu bar item)
# on this checkout's .venv.
# ponytail: not standalone (needs this repo + .venv in place); py2app if it must be shareable.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/MacEcho.app"
PY="$ROOT/.venv/bin/python"

# The window's orb is a web page (ui/: React + the ElevenLabs UI orb). Build it into
# src/macecho/ui/index.html, where menubar.py reads it. Needs Node; npm ci only the first time.
(cd "$ROOT/ui" && { [ -d node_modules ] || npm ci --no-audit --no-fund; } && npm run build)

# A running copy keeps the code it started with, and opening the app again only brings that copy
# forward. Quit it, so the next launch is this build.
pkill -f "$APP/Contents/MacOS/MacEcho" 2>/dev/null || true

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>MacEcho</string>
  <key>CFBundleDisplayName</key><string>MacEcho</string>
  <key>CFBundleIdentifier</key><string>local.macecho.menubar</string>
  <key>CFBundleExecutable</key><string>MacEcho</string>
  <key>CFBundleIconFile</key><string>MacEcho</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>MacEcho listens to your voice so it can reply.</string>
</dict>
</plist>
EOF

# The launcher embeds the venv's Python, so it links against that Python's libpython.
PYINC="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')"
PYLIB="$("$PY" -c 'import sysconfig as s; print(s.get_config_var("LIBDIR") + "/" + s.get_config_var("LDLIBRARY"))')"
clang -O2 -DROOT="\"$ROOT\"" -I"$PYINC" -o "$APP/Contents/MacOS/MacEcho" "$ROOT/scripts/launcher.c" "$PYLIB"

# ponytail: the Dock icon is the microphone emoji on a dark tile; swap in a drawn .icns to brand it.
"$PY" - "$APP/Contents/Resources/MacEcho.icns" <<'EOF'
import struct
import sys
from AppKit import (NSBezierPath, NSBitmapImageRep, NSColor, NSDeviceRGBColorSpace, NSFont,
                    NSFontAttributeName, NSGraphicsContext, NSPNGFileType, NSString)

rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
    None, 1024, 1024, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
NSColor.colorWithWhite_alpha_(0.13, 1).setFill()
# Apple's icon grid: an 824 px rounded tile centred in the 1024 px canvas.
NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(((100, 100), (824, 824)), 185, 185).fill()
mic = NSString.stringWithString_("🎙")
attrs = {NSFontAttributeName: NSFont.systemFontOfSize_(520)}
w, h = mic.sizeWithAttributes_(attrs)
mic.drawAtPoint_withAttributes_(((1024 - w) / 2, (1024 - h) / 2), attrs)
png = bytes(rep.representationUsingType_properties_(NSPNGFileType, None))
# An .icns file is a header plus tagged images; "ic10" is the 1024 px PNG, which macOS scales down.
with open(sys.argv[1], "wb") as f:
    f.write(b"icns" + struct.pack(">I", len(png) + 16) + b"ic10" + struct.pack(">I", len(png) + 8) + png)
EOF

codesign --force --sign - "$APP" 2>/dev/null || true  # ad-hoc sign so macOS remembers mic permission
echo "Built $APP"
