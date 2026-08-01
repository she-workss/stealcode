#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHITECTURE="${1:?usage: bundle-linux.sh <x86_64|aarch64> <version> [channel]}"
VERSION="${2:?usage: bundle-linux.sh <x86_64|aarch64> <version> [channel]}"
CHANNEL="${3:-stable}"

case "$ARCHITECTURE" in
  x86_64) TARGET="x86_64-unknown-linux-gnu" ;;
  aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
  *) echo "unsupported architecture: $ARCHITECTURE" >&2; exit 1 ;;
esac

echo "Building stealcode (features=all) for $TARGET (version $VERSION, channel $CHANNEL)"
STEALCODE_VERSION="$VERSION" STEALCODE_RELEASE_CHANNEL="$CHANNEL" \
  cargo build --release --package cli --features all --target "$TARGET"

OUTPUT_TARBALL="$REPO_ROOT/target/stealcode-linux-$ARCHITECTURE.tar.gz"
tar -czf "$OUTPUT_TARBALL" -C "$REPO_ROOT/target/$TARGET/release" stealcode

echo "Built $OUTPUT_TARBALL"
