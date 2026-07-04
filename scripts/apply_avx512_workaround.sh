#!/usr/bin/env bash
set -e
set -o pipefail

# Workaround for a Codon AVX-512 heap-vector alignment bug: Codon emits aligned
# 512-bit moves (vmovaps %zmm) for heap-resident Vec[*,8], but GC_malloc only
# 16-byte-aligns memory, so those moves fault (SIGSEGV) on x86-64 AVX-512 CPUs.
# Hit by Sequre's MHE/CKKS path (e.g. `sequre run ... --local`).
#
# Applies the install-side pieces that a fresh/bundled Codon does not carry:
#   1. Patch Codon's stdlib Ptr load/store to be unaligned (covers Ptr[Vec][i]
#      / Vec._load at arbitrary offsets).
#   2. Build the LD_PRELOAD over-align shim into $PREFIX/lib (covers
#      compiler-emitted aligned stores: Vec struct fields, vectorized loops).
#      The launcher auto-preloads it (maybe_set_align_shim()).
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
    echo "         The AVX-512 workaround is incomplete; sequre --local may crash." >&2
  fi
fi

# 2. Build the over-align LD_PRELOAD shim (skip if the release already bundled it).
if [ -f "$SHIM_SO" ]; then
  echo "Over-align shim already present: $SHIM_SO"
elif [ ! -f "$SHIM_C" ]; then
  echo "WARNING: shim missing and source $SHIM_C not found; sequre --local may crash." >&2
elif command -v "$CC" >/dev/null 2>&1; then
  mkdir -p "$SEQURE_PREFIX/lib"
  "$CC" -shared -fPIC -O2 -o "$SHIM_SO" "$SHIM_C" -ldl
  echo "Built over-align shim: $SHIM_SO"
else
  echo "WARNING: shim missing and no C compiler ('$CC'); sequre --local may crash." >&2
fi

# 3. Rebuild the launcher so it auto-preloads the shim (maybe_set_align_shim()),
#    unless the caller already built it from the patched source.
if [ "${SEQURE_WORKAROUND_SKIP_LAUNCHER:-0}" != "1" ]; then
  LAUNCHER_C="$SRC/sequre_launcher.c"
  if [ -f "$LAUNCHER_C" ]; then
    if "$CC" -O2 -o "$SEQURE_PREFIX/bin/sequre" "$LAUNCHER_C"; then
      echo "Rebuilt launcher with shim auto-preload: $SEQURE_PREFIX/bin/sequre"
    else
      echo "warning: launcher rebuild failed; set LD_PRELOAD=$SHIM_SO manually." >&2
    fi
  fi
fi

echo "Done. The sequre launcher auto-preloads the shim (disable: SEQURE_NO_ALIGN_SHIM=1)."
