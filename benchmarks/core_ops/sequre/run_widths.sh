#!/usr/bin/env bash
# Runs the Sequre benchmark at more than one share width so the report can show
# them side by side.
#
# MPC_INT_SIZE is a compile-time Static in stdlib/sequre/settings.codon, so the
# only way to vary it is to edit that file and rebuild. This script does that
# and restores the original on any exit path, including Ctrl-C.
#
#   ./run_widths.sh            # 128 and 192, the two deployable configurations
#   ./run_widths.sh 128        # just one
#   ./run_widths.sh 64 128 192 # include the 64-bit control
#
# 64 is deliberately NOT in the default set. It exists to match CrypTen's ring
# width for a like-for-like performance comparison and carries only 10 bits of
# statistical security -- see README.md, "Threat models are not identical".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
SETTINGS="${ROOT}/stdlib/sequre/settings.codon"
BACKUP="$(mktemp)"

WIDTHS=("$@")
if [[ ${#WIDTHS[@]} -eq 0 ]]; then
    WIDTHS=(128 192)
fi

cp "${SETTINGS}" "${BACKUP}"
restore() {
    cp "${BACKUP}" "${SETTINGS}"
    rm -f "${BACKUP}"
    echo "restored ${SETTINGS}"
}
trap restore EXIT INT TERM

for width in "${WIDTHS[@]}"; do
    case "${width}" in
        64|128|192|256) ;;
        *) echo "unsupported MPC_INT_SIZE: ${width} (must be 64, 128, 192 or 256)" >&2; exit 1 ;;
    esac
    echo "=== Sequre at MPC_INT_SIZE=${width} ==="
    # Rewrite from the pristine backup each time, never from a previous edit.
    sed "s/^MPC_INT_SIZE: Static\[int\] = .*/MPC_INT_SIZE: Static[int] = ${width}  # Can be either 128, 192, or 256/" \
        "${BACKUP}" > "${SETTINGS}"
    cd "${ROOT}"
    sequre run -release benchmarks/core_ops/sequre/bench_core.codon --local --use-ring
    echo
done
