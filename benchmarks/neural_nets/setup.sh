#!/usr/bin/env bash
# Creates the Python environment the CrypTen and MPyC runners need.
#
# The dependency set is identical to the activation benchmark's, so if that
# environment already exists this script links to it rather than building a
# second copy of torch. Pass --fresh to force a private one.
#
# CrypTen 0.4.1 is the last release and has not been updated since 2022: it
# imports torch.onnx._internal.registration, which torch removed after 2.4, and
# it declares a dependency on the deprecated `sklearn` shim that refuses to
# build. So the environment is pinned to Python 3.12 + torch 2.4.1, and CrypTen
# is installed with --no-deps against manually pinned dependencies. Give
# CRYPTEN_BENCH_PYTHON a Python 3.12 interpreter, or let this script find one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv"
SHARED="${HERE}/../activations/.venv"

if [[ "${1:-}" != "--fresh" && -x "${SHARED}/bin/python" ]]; then
    ln -sfn "../activations/.venv" "${VENV}"
    echo "linked ${VENV} -> ${SHARED}"
    "${VENV}/bin/python" -c 'import crypten, mpyc, numpy, torch; print("environment ready")'
    exit 0
fi

find_python312() {
    if [[ -n "${CRYPTEN_BENCH_PYTHON:-}" ]]; then
        echo "${CRYPTEN_BENCH_PYTHON}"; return
    fi
    for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
        if command -v "${candidate}" >/dev/null 2>&1; then echo "${candidate}"; return; fi
    done
    echo "no Python 3.12 found; set CRYPTEN_BENCH_PYTHON to one" >&2
    exit 1
}

PY="$(find_python312)"
echo "using interpreter: ${PY}"
rm -rf "${VENV}"
"${PY}" -m venv "${VENV}"

"${VENV}/bin/pip" install --upgrade pip setuptools wheel
"${VENV}/bin/pip" install -r "${HERE}/../activations/requirements.txt"
# CrypTen's own metadata is unusable (see above); its real dependencies are in
# requirements.txt and were just installed.
"${VENV}/bin/pip" install --no-deps "crypten==0.4.1"

"${VENV}/bin/python" - <<'PY'
import crypten, mpyc, numpy, torch
print(f"crypten {crypten.__version__}  torch {torch.__version__}  "
      f"mpyc {mpyc.__version__}  numpy {numpy.__version__}")
PY

echo
echo "environment ready: ${VENV}"
echo "run the suite with: ${HERE}/run_all.sh"
