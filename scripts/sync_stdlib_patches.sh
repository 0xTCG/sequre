#!/usr/bin/env bash
set -e
set -o pipefail

# Sequre's `stdlib/numpy/` is a fork of Codon's own numpy (Sequre's ndarray is
# defined differently and is incompatible with Codon's — this is intentional
# for now; the long-term plan is to migrate Sequre to use Codon's numpy
# directly).
#
# Sequre imports the fork exclusively via the `sequre.stdlib.numpy` namespace,
# which resolves to the copy bundled inside the plugin
# ($CODON_PATH/lib/codon/plugins/sequre/stdlib/numpy) — never to Codon's
# native `numpy`. So a local edit to stdlib/numpy/ only needs to be mirrored
# into the plugin's stdlib dir to take effect, and Codon's own numpy/ is left
# untouched.
#
# Usage: scripts/sync_stdlib_patches.sh [/path/to/.sequre]

SEQURE_PREFIX=${1:-$HOME/.sequre}
SRC=$(cd "$(dirname "$0")/.." && pwd)

PLUGIN_STDLIB="$SEQURE_PREFIX/lib/codon/plugins/sequre/stdlib"

if [ ! -d "$PLUGIN_STDLIB" ]; then
  echo "error: $PLUGIN_STDLIB not found. Is Sequre installed at $SEQURE_PREFIX?" >&2
  exit 1
fi

echo "Syncing Sequre's numpy fork into the plugin stdlib"
rm -rf "$PLUGIN_STDLIB/numpy"
cp -r "$SRC/stdlib/numpy" "$PLUGIN_STDLIB/numpy"

echo "Done."
