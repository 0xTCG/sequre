#!/usr/bin/env bash
# Creates the Python environment the CrypTen and MPyC runners need.
#
# CrypTen 0.4.1 is the last release and has not been updated since
# 2022: it imports torch.onnx._internal.registration, which torch removed after
# 2.4, and it declares a dependency on the deprecated `sklearn` shim that
# refuses to build. So the environment is pinned to Python 3.12 + torch 2.4.1,
# and CrypTen is installed with --no-deps against manually pinned dependencies.
# Give CRYPTEN_BENCH_PYTHON a Python 3.12 interpreter, or let this script find
# one. If ../activations/.venv already exists it is reused rather than
# duplicated -- the two suites need exactly the same packages.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv"
SIBLING="${HERE}/../activations/.venv"

find_python312() {
    if [[ -n "${CRYPTEN_BENCH_PYTHON:-}" ]]; then
        echo "${CRYPTEN_BENCH_PYTHON}"; return
    fi
    for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
        if command -v "${candidate}" >/dev/null 2>&1; then echo "${candidate}"; return; fi
    done
    if command -v conda >/dev/null 2>&1; then
        if ! conda env list | grep -q '^sequre-core-bench '; then
            conda create -y -n sequre-core-bench python=3.12 >/dev/null
        fi
        conda run -n sequre-core-bench python -c 'import sys; print(sys.executable)'
        return
    fi
    echo "no Python 3.12 found; set CRYPTEN_BENCH_PYTHON to one" >&2
    exit 1
}

if [[ -x "${SIBLING}/bin/python" ]] && [[ ! -e "${VENV}" ]]; then
    echo "reusing ${SIBLING}"
    ln -s "$(cd "$(dirname "${SIBLING}")" && pwd)/.venv" "${VENV}"
else
    PY="$(find_python312)"
    echo "using interpreter: ${PY}"
    "${PY}" -m venv "${VENV}"
    "${VENV}/bin/pip" install --upgrade pip setuptools wheel
    "${VENV}/bin/pip" install -r "${HERE}/requirements.txt"
    # CrypTen's own metadata is unusable (see above); its real dependencies are
    # in requirements.txt and were just installed.
    "${VENV}/bin/pip" install --no-deps "crypten==0.4.1"
fi

"${VENV}/bin/python" - <<'PY'
import crypten, mpyc, numpy, torch
print(f"crypten {crypten.__version__}  torch {torch.__version__}  "
      f"mpyc {mpyc.__version__}  numpy {numpy.__version__}")
PY

echo
echo "environment ready: ${VENV}"
echo "run the suite with: ${HERE}/run_all.sh"
