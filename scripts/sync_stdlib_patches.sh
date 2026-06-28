#!/usr/bin/env bash
set -e
set -o pipefail

# Sequre's `stdlib/numpy/` is a fork of Codon's
# own numpy (Sequre's ndarray is defined differently and is
# incompatible with Codon's — this is intentional for now; the long-term
# plan is to migrate Sequre to use Codon's numpy directly).
#
# `import numpy` always resolve from
# $CODON_PATH/lib/codon/stdlib (the package root), never from the plugin's
# stdlib dir, so local edits only take effect once applied there too. Since
# the two numpy packages are incompatible, Codon's numpy/ directory is
# wholesale replaced by Sequre's (not merged file-by-file) so nothing in
# Codon's original numpy lingers with a stale/incompatible ndarray type.
#
# This script applies that swap to both locations so a local edit takes
# effect immediately, without rebuilding/reinstalling the whole plugin.
#
# Usage: scripts/sync_stdlib_patches.sh [/path/to/.sequre]

SEQURE_PREFIX=${1:-$HOME/.sequre}
SRC=$(cd "$(dirname "$0")/.." && pwd)

PLUGIN_STDLIB="$SEQURE_PREFIX/lib/codon/plugins/sequre/stdlib"
CODON_STDLIB="$SEQURE_PREFIX/lib/codon/stdlib"

if [ ! -d "$CODON_STDLIB" ]; then
  echo "error: $CODON_STDLIB not found. Is Sequre/Codon installed at $SEQURE_PREFIX?" >&2
  exit 1
fi

echo "Replacing numpy package with Sequre's fork"
rm -rf "$PLUGIN_STDLIB/numpy" "$CODON_STDLIB/numpy"
cp -r "$SRC/stdlib/numpy" "$PLUGIN_STDLIB/numpy"
cp -r "$SRC/stdlib/numpy" "$CODON_STDLIB/numpy"

echo "Done."
