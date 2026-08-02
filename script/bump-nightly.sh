#!/usr/bin/env bash
set -euo pipefail

REPO="she-workss/stealcode"

# Force a nightly release by running the `release_nightly` workflow.
gh workflow run release_nightly.yml --repo "$REPO" --ref main