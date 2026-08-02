#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_PATH/.."

REPO="she-workss/stealcode"
CHANNEL="${1:-}"

case "$CHANNEL" in
  nightly)
    echo "Dispatching nightly release (release_nightly.yml)"
    exec ./script/bump-nightly.sh
    ;;

  stable|"")
    BUMP="${BUMP:-patch}"
    echo "Bumping $BUMP version on main and building a stable release"
    gh workflow run bump_stealcode_version.yml \
      --repo "$REPO" \
      --ref main \
      -f bump="$BUMP" \
      -f channel=stable
    ;;

  *)
    echo "usage: trigger-release.sh [stable|nightly]" >&2
    exit 1
    ;;
esac