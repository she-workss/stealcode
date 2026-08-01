#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHITECTURE="${1:?usage: bundle-mac.sh <x86_64|aarch64> <version> [channel]}"
VERSION="${2:?usage: bundle-mac.sh <x86_64|aarch64> <version> [channel]}"
CHANNEL="${3:-stable}"

case "$ARCHITECTURE" in
  x86_64) TARGET="x86_64-apple-darwin" ;;
  aarch64) TARGET="aarch64-apple-darwin" ;;
  *) echo "unsupported architecture: $ARCHITECTURE" >&2; exit 1 ;;
esac

echo "Building stealcode (features=all) for $TARGET (version $VERSION, channel $CHANNEL)"
STEALCODE_VERSION="$VERSION" STEALCODE_RELEASE_CHANNEL="$CHANNEL" \
  cargo build --release --package cli --features all --target "$TARGET"

APP_DIR="$REPO_ROOT/target/$TARGET/release/StealCode.app"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

cp "$REPO_ROOT/target/$TARGET/release/stealcode" "$APP_DIR/Contents/MacOS/stealcode"
cp "$REPO_ROOT/crates/cli/assets/resources/macos/Info.plist" "$APP_DIR/Contents/Info.plist"
if [ -f "$REPO_ROOT/crates/cli/assets/icons/prod/icon.icns" ]; then
  cp "$REPO_ROOT/crates/cli/assets/icons/prod/icon.icns" "$APP_DIR/Contents/Resources/icon.icns"
fi

DMG_STAGING="$REPO_ROOT/target/$TARGET/dmg-staging"
rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"
cp -R "$APP_DIR" "$DMG_STAGING/StealCode.app"
ln -s /Applications "$DMG_STAGING/Applications"

OUTPUT_DMG="$REPO_ROOT/target/StealCode-$ARCHITECTURE.dmg"
rm -f "$OUTPUT_DMG"
hdiutil create -volname "StealCode" -srcfolder "$DMG_STAGING" -ov -format UDZO "$OUTPUT_DMG"

echo "Built $OUTPUT_DMG"
