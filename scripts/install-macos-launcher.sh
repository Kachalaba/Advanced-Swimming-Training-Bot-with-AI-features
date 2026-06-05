#!/usr/bin/env bash
# Build a double-clickable "Sprint AI.app" with a custom icon (macOS only).
# Run once, from anywhere inside the repo:
#     bash scripts/install-macos-launcher.sh            # -> ~/Desktop
#     bash scripts/install-macos-launcher.sh /Applications
set -euo pipefail

if [ "$(uname)" != "Darwin" ]; then
  echo "ℹ️  This installer is for macOS."
  echo "   On other systems just run:  bash scripts/sprint-ai-run.sh"
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="Sprint AI"
DEST="${1:-$HOME/Desktop}"
APP="$DEST/$APP_NAME.app"
ICON_PNG="$ROOT/assets/sprint-ai-icon.png"

echo "📦  Building $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- Info.plist --------------------------------------------------------------
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Sprint AI</string>
  <key>CFBundleDisplayName</key><string>Sprint AI</string>
  <key>CFBundleIdentifier</key><string>ai.sprint.launcher</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>sprint-ai</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# --- launcher executable -----------------------------------------------------
# Opens Terminal so logs are visible and Ctrl+C / closing the window stops the
# server. The repo path is baked in below via sed.
cat > "$APP/Contents/MacOS/sprint-ai" <<'LAUNCH'
#!/bin/bash
ROOT='__SPRINT_AI_ROOT__'
CMD="cd '$ROOT' && SPRINT_AI_HOME='$ROOT' bash scripts/sprint-ai-run.sh"
/usr/bin/osascript \
  -e 'tell application "Terminal" to activate' \
  -e "tell application \"Terminal\" to do script \"$CMD\""
LAUNCH
sed -i '' "s|__SPRINT_AI_ROOT__|$ROOT|g" "$APP/Contents/MacOS/sprint-ai"
chmod +x "$APP/Contents/MacOS/sprint-ai"

# --- icon (sips + iconutil ship with macOS) ---------------------------------
if [ -f "$ICON_PNG" ] && command -v iconutil >/dev/null 2>&1; then
  ICONSET="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for sz in 16 32 128 256 512; do
    sips -z "$sz" "$sz" "$ICON_PNG" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null
    sips -z "$((sz * 2))" "$((sz * 2))" "$ICON_PNG" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
  rm -rf "$(dirname "$ICONSET")"
  echo "🎨  Icon installed."
else
  echo "ℹ️  Icon skipped (no iconutil or PNG) — the default app icon will be used."
fi

touch "$APP"  # nudge Finder to pick up the new bundle/icon
echo "✅  Done! Desktop shortcut: \"$APP_NAME\""
echo "   Double-click it — Sprint AI opens in your browser at http://localhost:3000"
echo "   (First launch may show a Gatekeeper prompt: right-click → Open once.)"
