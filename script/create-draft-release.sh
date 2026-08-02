#!/usr/bin/env bash
set -euo pipefail

NOTES_FILE="${1:?usage: create-draft-release.sh <notes-file> [channel] [tag]}"
CHANNEL="${2:-stable}"
TAG="${3:?tag must be passed explicitly (do not rely on GITHUB_REF_NAME: it is a reserved, runner-managed variable)}"

if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-pre)?$ ]]; then
    echo "::error::invalid release tag '$TAG' (expected e.g. v0.2.0 or v0.2.0-pre)" >&2
    exit 1
fi

PRERELEASE_ARGS=()
if [ "$CHANNEL" = "nightly" ]; then
    PRERELEASE_ARGS=(--prerelease)
fi

gh release create "$TAG" \
    --repo she-workss/stealcode \
    --title "$TAG" \
    --notes-file "$NOTES_FILE" \
    --draft \
    "${PRERELEASE_ARGS[@]}"
