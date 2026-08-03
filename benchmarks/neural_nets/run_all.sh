#!/usr/bin/env bash
# Runs all three frameworks and builds the comparison report.
#
#   ./run_all.sh                 # everything
#   ./run_all.sh sequre crypten  # a subset
#
# Sequre must be run from the repository root (it resolves its stdlib and the
# results directory relative to it), so this script cds there itself.
#
# Expect this to take a while. The MLP is SecureML's 784-128-128-10 and Sequre
# spends ~41 s per epoch on it at n = 512; MPyC has no neural-network layer and
# needs ~150 s per epoch at n = 8, which is why it is capped by default.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
VENV_PY="${HERE}/.venv/bin/python"
RESULTS="${HERE}/results"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(sequre crypten mpyc)
fi

wants() {
    for t in "${TARGETS[@]}"; do [[ "$t" == "$1" ]] && return 0; done
    return 1
}

# report.py needs the environment too, not just the two Python runners: it
# computes every accuracy column itself, from ref.py's numpy training loop.
if [[ ! -x "${VENV_PY}" ]]; then
    echo "Python environment missing. Run ${HERE}/setup.sh first." >&2
    exit 1
fi

mkdir -p "${RESULTS}"

if wants sequre; then
    echo "=== Sequre (128- and 192-bit shares) ==="
    # -release matters: Sequre compiles with backtraces by default and is
    # substantially slower that way. --use-ring is what Decor's protocols are
    # designed for; the runner also forces the ring internally.
    "${HERE}/sequre/run_widths.sh"
    echo
fi

if wants crypten; then
    # Both providers. TFP is CrypTen's default but is not a secure
    # configuration with two parties -- party 0 generates the Beaver triples
    # and can therefore reconstruct party 1's inputs. TTP adds a real third
    # party and is what the Sequre columns should be compared against.
    echo "=== CrypTen (TFP -- CrypTen default, insecure at 2 parties) ==="
    cd "${HERE}/crypten"
    "${VENV_PY}" bench_nn.py --provider TFP
    echo
    echo "=== CrypTen (TTP -- separate trusted third party) ==="
    "${VENV_PY}" bench_nn.py --provider TTP
    echo
fi

if wants mpyc; then
    echo "=== MPyC ==="
    cd "${HERE}/mpyc"
    # -M3 makes MPyC spawn all three parties itself on localhost.
    "${VENV_PY}" bench_nn.py -M3
    echo
fi

echo "=== report ==="
cd "${HERE}"
"${VENV_PY}" report.py --out "${HERE}/REPORT.md"
cat "${HERE}/REPORT.md"
