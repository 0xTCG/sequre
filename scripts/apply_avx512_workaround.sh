#!/usr/bin/env bash
set -e
set -o pipefail

# Workaround for a Codon x86-64 heap-vector alignment bug: Codon emits aligned
# vector moves (vmovaps, 256-bit with AVX2 and 512-bit with AVX-512) for
# heap-resident Vec[*,8], but GC_malloc only 16-byte-aligns memory, so those
# moves fault (SIGSEGV) on x86-64 CPUs with AVX2 or AVX-512. Hit by Sequre's
# MHE/CKKS path (e.g. `sequre run ... --local`).
#
# Applies the install-side pieces that a fresh/bundled Codon does not carry:
#   1. Patch Codon's stdlib Ptr load/store to be unaligned (covers Ptr[Vec][i]
#      / Vec._load at arbitrary offsets).
#   2. Build the over-align shim into $PREFIX/lib (covers compiler-emitted
#      aligned stores: Vec struct fields, vectorized loops). The launcher
#      preloads it for `sequre run` and links it into the binaries of
#      `sequre build`. Rebuilt when sequre_align64.c is newer than it, so a
#      source update reaches it.
#
# The third piece (stdlib/prg.codon's @llvm vector loads -> align 1) lives in
# Sequre's own source and ships in the plugin (lib/codon/plugins/sequre/stdlib),
# so it needs no install-time action. The launcher is rebuilt by the build
# scripts from the patched sequre_launcher.c; pass SEQURE_WORKAROUND_SKIP_LAUNCHER=1
# to skip rebuilding it here (e.g. when the build already did).
#
# Idempotent. No-op on non-x86_64 (e.g. macOS/arm64, which is unaffected).
#
# Usage: scripts/apply_avx512_workaround.sh [/path/to/.sequre]
# Env:   CC (compiler, default cc), SEQURE_WORKAROUND_SKIP_LAUNCHER

SEQURE_PREFIX=${1:-$HOME/.sequre}
SRC=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-cc}

ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
  echo "apply_avx512_workaround: arch '$ARCH' is unaffected; skipping."
  exit 0
fi

PTR="$SEQURE_PREFIX/lib/codon/stdlib/internal/types/ptr.codon"
SHIM_SO="$SEQURE_PREFIX/lib/sequre_align64.so"
SHIM_C="$SRC/sequre_align64.c"

# 1. Patch Codon's Ptr.__getitem__/__setitem__ to use unaligned load/store.
#    Append ', align 1' to the @llvm load/store lines that end in the GEP result
#    '%0'. Handles both opaque ('ptr %0') and typed ('{=T}* %0') pointer syntax,
#    and is idempotent (lines already carrying 'align' are skipped).
if [ ! -f "$PTR" ]; then
  echo "apply_avx512_workaround: $PTR not found; skipping Ptr patch (Codon stdlib not present here)."
else
  before=$(grep -cE 'align 1' "$PTR" || true)
  sed -i -E '/(^|[[:space:]])(load|store)[[:space:]]/ { /align/! { /%0$/ s/$/, align 1/ } }' "$PTR"
  after=$(grep -cE 'align 1' "$PTR" || true)
  if [ "$after" -gt "$before" ]; then
    echo "Patched Ptr load/store to unaligned (+$((after - before)) sites): $PTR"
  elif [ "$after" -gt 0 ]; then
    echo "Ptr load/store already unaligned; skipping: $PTR"
  else
    echo "WARNING: could not patch $PTR (unexpected layout)." >&2
    echo "         The x86-64 alignment workaround is incomplete; sequre --local may crash." >&2
  fi
fi

# 2. Build the over-align shim, when it is missing or older than its source (so
#    a stale shim never stays behind a source update). Built into a temporary
#    file first: a failed build keeps the shim that was there.
if [ -f "$SHIM_SO" ] && { [ ! -f "$SHIM_C" ] || [ ! "$SHIM_C" -nt "$SHIM_SO" ]; }; then
  echo "Over-align shim up to date: $SHIM_SO"
elif [ ! -f "$SHIM_C" ]; then
  echo "WARNING: shim missing and source $SHIM_C not found; sequre --local may crash." >&2
elif ! command -v "$CC" >/dev/null 2>&1; then
  if [ -f "$SHIM_SO" ]; then
    echo "WARNING: $SHIM_SO is older than $SHIM_C, and there is no C compiler ('$CC') to rebuild it." >&2
  else
    echo "WARNING: shim missing and no C compiler ('$CC'); sequre --local may crash." >&2
  fi
else
  mkdir -p "$SEQURE_PREFIX/lib"
  if "$CC" -shared -fPIC -O2 -o "$SHIM_SO.tmp.$$" "$SHIM_C" -ldl && mv -f "$SHIM_SO.tmp.$$" "$SHIM_SO"; then
    echo "Built over-align shim: $SHIM_SO"
  else
    rm -f "$SHIM_SO.tmp.$$"
    echo "WARNING: could not build the over-align shim from $SHIM_C; sequre --local may crash." >&2
  fi
fi

# 3. Rebuild the launcher so it preloads and links the shim, unless the caller
#    already built it from the current source.
if [ "${SEQURE_WORKAROUND_SKIP_LAUNCHER:-0}" != "1" ]; then
  LAUNCHER_C="$SRC/sequre_launcher.c"
  if [ -f "$LAUNCHER_C" ]; then
    if "$CC" -O2 -o "$SEQURE_PREFIX/bin/sequre" "$LAUNCHER_C"; then
      echo "Rebuilt launcher: $SEQURE_PREFIX/bin/sequre"
    else
      echo "warning: launcher rebuild failed; set LD_PRELOAD=$SHIM_SO manually." >&2
    fi
  fi
fi

echo "Done. The sequre launcher preloads the shim for \`sequre run\` and links it into built binaries (disable: SEQURE_NO_ALIGN_SHIM=1)."
