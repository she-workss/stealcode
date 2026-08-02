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
    version="$(./script/get-released-version.sh stable "$REPO")"
    branch="v$(echo "$version" | awk -F. '{print $1"."$2}').x"
    echo "Bumping $BUMP version on stable branch $branch (based on released v$version)"
    gh workflow run bump_stealcode_version.yml \
      --repo "$REPO" \
      -f bump="$BUMP" \
      -f channel=stable \
      -f branch="$branch"
    ;;

  *)
    echo "usage: trigger-release.sh [stable|nightly]" >&2
    exit 1
    ;;
esac