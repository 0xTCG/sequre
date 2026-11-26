#!/usr/bin/env bash
set -e
set -o pipefail

# Workaround for a Codon x86-64 heap-vector alignment bug: Codon emits aligned
# vector moves (vmovaps, 256-bit with AVX2 and 512-bit with AVX-512) for
# heap-resident Vec[*,8], but GC_malloc only 16-byte-aligns memory, so those
# moves fault (SIGSEGV) on x86-64 CPUs with AVX2 or AVX-512. Hit by Sequre's
# MHE/CKKS path (e.g. `sequre run ... --local`).
#
# Builds the over-align shim into $PREFIX/lib: it 64-aligns heap blocks, so the
# aligned moves Codon emits for them (tuples of vectors, Vec struct fields,
# generator frames, vectorized loops) are safe. The launcher preloads it for
# `sequre run` and links it into the binaries of `sequre build`. Rebuilt when
# sequre_align64.c is newer than it, so a source update reaches it.
#
# Codon 0.20's own simd.codon makes loads and stores through Ptr[Vec] unaligned
# (Codon 0.19.6 also needed its stdlib's Ptr patched, which the installer still
# does for the archives built for it), and stdlib/prg.codon's @llvm vector loads
# are unaligned in Sequre's own source, which ships in the plugin
# (lib/codon/plugins/sequre/stdlib): neither needs an install-time action. The
# launcher is rebuilt by the build scripts from sequre_launcher.c; pass
# SEQURE_WORKAROUND_SKIP_LAUNCHER=1 to skip rebuilding it here (e.g. when the
# build already did).
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

SHIM_SO="$SEQURE_PREFIX/lib/sequre_align64.so"
SHIM_C="$SRC/sequre_align64.c"

# 1. Build the over-align shim, when it is missing or older than its source (so
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

# 2. Rebuild the launcher so it preloads and links the shim, unless the caller
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
