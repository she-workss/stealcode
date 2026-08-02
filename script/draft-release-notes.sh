#!/usr/bin/env bash
set -euo pipefail

TAG="${1:?usage: draft-release-notes.sh <tag>}"
REPO="${2:-she-workss/stealcode}"

PRIOR_TAG="$(git describe --tags --abbrev=0 "${TAG}^" 2>/dev/null || true)"

if [ -z "$PRIOR_TAG" ]; then
    echo "No prior tag found - this looks like the first release. Skipping changelog."
    exit 0
fi

echo "Changes since $PRIOR_TAG:"
echo

git log "${PRIOR_TAG}..${TAG}" --format='%H%x01%s' | while IFS=$'\x01' read -r hash subject; do
    pr_number="$(echo "$subject" | grep -oE '\(#[0-9]+\)$' | tr -d '()#' || true)"
    if [ -n "$pr_number" ]; then
        link="https://github.com/$REPO/pull/$pr_number"
    else
        link="https://github.com/$REPO/commit/$hash"
    fi
    echo "- $subject ([details]($link))"
done

echo
echo "[Full diff](https://github.com/$REPO/compare/${PRIOR_TAG}...${TAG})"
