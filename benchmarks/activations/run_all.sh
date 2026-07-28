#!/usr/bin/env bash
# Runs all three frameworks and builds the comparison report.
#
#   ./run_all.sh                 # everything
#   ./run_all.sh sequre crypten  # a subset
#
# Sequre must be run from the repository root (it resolves its stdlib and the
# results directory relative to it), so this script cds there itself.
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

if wants crypten || wants mpyc; then
    if [[ ! -x "${VENV_PY}" ]]; then
        echo "Python environment missing. Run ${HERE}/setup.sh first." >&2
        exit 1
    fi
fi

mkdir -p "${RESULTS}"

if wants sequre; then
    echo "=== Sequre (Decor + Fourier paths) ==="
    cd "${ROOT}"
    # -release matters: Sequre compiles with backtraces by default and is
    # substantially slower that way. Benchmarking the debug build would be
    # meaningless.
    #
    # --use-ring is how Decor is meant to be run: its reduced-ring comparison
    # needs a power-of-two modulus. The benchmark also forces the ring
    # internally via mpc.default_modulus, so the flag does not change the
    # protocol timings, but it puts input sharing on the ring too and saves
    # ~9% of the bytes.
    sequre run -release benchmarks/activations/sequre/bench_activations.codon --local --use-ring
    echo
fi

if wants crypten; then
    # Both providers. TFP is CrypTen's default but is not a secure
    # configuration with two parties -- party 0 generates the Beaver triples
    # and can therefore reconstruct party 1's inputs. TTP adds a real third
    # party and is what the Sequre columns should be compared against.
    echo "=== CrypTen (TFP -- CrypTen default, insecure at 2 parties) ==="
    cd "${HERE}/crypten"
    "${VENV_PY}" bench_activations.py --provider TFP
    echo
    echo "=== CrypTen (TTP -- separate trusted third party) ==="
    "${VENV_PY}" bench_activations.py --provider TTP
    echo
fi

if wants mpyc; then
    echo "=== MPyC ==="
    cd "${HERE}/mpyc"
    # -M3 makes MPyC spawn all three parties itself on localhost.
    "${VENV_PY}" bench_activations.py -M3
    echo
fi

echo "=== report ==="
cd "${HERE}"
# report.py only needs the standard library, so a sequre-only run does not
# require the benchmark venv.
REPORT_PY="python3"
[[ -x "${VENV_PY}" ]] && REPORT_PY="${VENV_PY}"
"${REPORT_PY}" report.py --out "${HERE}/REPORT.md"
cat "${HERE}/REPORT.md"
