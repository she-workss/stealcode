#!/usr/bin/env bash
set -euo pipefail

REPO="${2:-she-workss/stealcode}"
CHANNEL="${1:?usage: get-released-version.sh <stable|nightly>}"

case "$CHANNEL" in
  stable)
    version="$(gh api --jq .tag_name --method GET "repos/$REPO/releases/latest" 2>/dev/null || true)"
    ;;
  nightly)
    version="$(gh api --paginate --jq '[.[] | select(.draft == false and .prerelease == true) | .tag_name][0]' "repos/$REPO/releases" 2>/dev/null || true)"
    ;;
  *)
    echo "usage: get-released-version.sh <stable|nightly>" >&2
    exit 1
    ;;
esac

echo "${version#v}"