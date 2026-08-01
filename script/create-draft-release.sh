#!/usr/bin/env bash
set -euo pipefail

NOTES_FILE="${1:?usage: create-draft-release.sh <notes-file> [channel]}"
CHANNEL="${2:-stable}"
TAG="${GITHUB_REF_NAME:?GITHUB_REF_NAME must be set}"

PRERELEASE_ARGS=()
if [ "$CHANNEL" = "nightly" ]; then
    PRERELEASE_ARGS=(--prerelease)
fi

gh release create "$TAG" \
    --repo he-thinks/stealcode \
    --title "$TAG" \
    --notes-file "$NOTES_FILE" \
    --draft \
    "${PRERELEASE_ARGS[@]}"
