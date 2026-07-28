"""Single source of truth for the activation-function benchmark.

Every runner -- Sequre/Decor, CrypTen, MPyC -- draws its inputs, its intervals
and its repetition counts from here so the three columns of the report are
comparable. The Codon runner cannot import Python, so it mirrors these
constants in `sequre/bench_activations.codon` and echoes them into its output
header; `report.py` refuses to build a table if the echoed values disagree with
this file.
"""

from __future__ import annotations

import math


SPEC_VERSION = "1"

# Elementwise inputs are linspace(a, b, n) rather than random draws: it is
# reproducible across three languages without having to match RNG streams, and
# it covers the interval endpoints, which is where every approximation-based
# method is at its worst.
FUNCTIONS: dict[str, dict] = {
    "exp": {
        # Decor confines its input to a known interval so the masked value
        # cannot leave the domain where plaintext evaluation is numerically
        # safe; the same interval is fed to the other frameworks.
        "interval": (-3.0, 3.0),
        "description": "e^x -- the building block for sigmoid/tanh/softmax",
    },
    "sigmoid": {
        "interval": (-6.0, 6.0),
        "description": "1 / (1 + e^-x)",
    },
    "tanh": {
        "interval": (-4.0, 4.0),
        "description": "tanh(x)",
    },
    "relu": {
        "interval": (-5.0, 5.0),
        "description": "max(x, 0) -- comparison-bound rather than approximation-bound",
    },
    # Decor's trigonometric protocols. These are periodic, so Decor needs no
    # interval confinement at all -- only rescaling by the period -- and the
    # whole series costs the same three rounds as a single sine.
    "sin": {
        "interval": (-math.pi, math.pi),
        "description": "sin(x) over one full period",
    },
    "cos": {
        "interval": (-math.pi, math.pi),
        "description": "cos(x) over one full period",
    },
    "tan": {
        # Poles at +-pi/2, so the interval stays well inside them. Decor's own
        # unit tests apply the same guard (period < pi, lower bound negative).
        "interval": (-math.pi / 4, math.pi / 4),
        "description": "tan(x), pole-free interval",
    },
    "cot": {
        # cot has its pole at 0, so the interval is shifted positive.
        "interval": (math.pi / 4, 3 * math.pi / 4),
        "description": "cot(x), pole-free interval",
    },
    # Hyperbolics grow exponentially, so the interval is kept narrow; Decor
    # requires period <= 10 for these.
    "sinh": {
        "interval": (-2.0, 2.0),
        "description": "sinh(x)",
    },
    "cosh": {
        "interval": (-2.0, 2.0),
        "description": "cosh(x)",
    },
    "polynomial": {
        "interval": (-2.0, 2.0),
        "description": "degree-3 polynomial, evaluated by binomial expansion over powers of the mask",
    },
}

# Coefficients for the "polynomial" entry, lowest order first:
# 1 + 2x + 3x^2 + 4x^3. Every framework must evaluate exactly this.
POLY_COEFFS: list[float] = [1.0, 2.0, 3.0, 4.0]
POLY_DEGREE = len(POLY_COEFFS) - 1

# Functions no framework outside Sequre implements natively. The other runners
# build them from that framework's own primitives and say so; report.py shows
# them like any other row.
NON_NATIVE_ELSEWHERE = ("tan", "cot", "sinh", "cosh")

# Vector lengths. 8 is a latency probe (protocol round-trips dominate); 8192
# is a throughput probe (local arithmetic dominates).
SIZES: list[int] = [8, 128, 1024, 8192]

# Timed repetitions per (function, size). The median is reported; the min is
# kept as a floor estimate.
REPS = 5

# Untimed repetitions run before the timed ones, to pay for lazy allocation and
# any first-call setup inside a framework.
WARMUP_REPS = 1

# MPyC is a pure-Python runtime and is orders of magnitude slower per element;
# left uncapped it dominates the wall clock of the whole suite. Sizes above
# this are recorded as skipped rather than silently dropped.
MPYC_DEFAULT_MAX_N = 1024

# Party counts. These differ by design -- see README.md, "Threat models are not
# identical". They are recorded in every result row so the report can show them.
PARTIES = {
    "sequre-decor-64": 3,    # 64-bit control, see README
    "sequre-decor-128": 3,   # CP0 offline dealer + CP1 + CP2 online
    "sequre-decor-192": 3,
    "sequre-fourier-64": 3,
    "sequre-fourier-128": 3,
    "sequre-fourier-192": 3,
    "crypten": 2,            # TFP: 2 parties, one of which makes the triples (insecure)
    "crypten-ttp": 3,        # TTP: 2 online parties + a separate TTPServer
    "mpyc": 3,            # Shamir, threshold 1, no dealer
}


def inputs(function: str, n: int) -> list[float]:
    """The exact input vector every framework must evaluate."""
    a, b = FUNCTIONS[function]["interval"]
    if n == 1:
        return [(a + b) / 2]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def ground_truth(function: str, values: list[float]) -> list[float]:
    """float64 reference, computed in the clear."""
    import math

    if function == "exp":
        return [math.exp(v) for v in values]
    if function == "sigmoid":
        return [1.0 / (1.0 + math.exp(-v)) for v in values]
    if function == "tanh":
        return [math.tanh(v) for v in values]
    if function == "relu":
        return [max(v, 0.0) for v in values]
    if function == "sin":
        return [math.sin(v) for v in values]
    if function == "cos":
        return [math.cos(v) for v in values]
    if function == "tan":
        return [math.tan(v) for v in values]
    if function == "cot":
        return [1.0 / math.tan(v) for v in values]
    if function == "sinh":
        return [math.sinh(v) for v in values]
    if function == "cosh":
        return [math.cosh(v) for v in values]
    if function == "polynomial":
        return [sum(c * v**i for i, c in enumerate(POLY_COEFFS)) for v in values]
    raise ValueError(f"unknown function: {function}")


def header() -> dict:
    """Spec fingerprint embedded in every result file, checked by report.py."""
    return {
        "spec_version": SPEC_VERSION,
        "sizes": SIZES,
        "reps": REPS,
        "intervals": {k: list(v["interval"]) for k, v in FUNCTIONS.items()},
        "poly_coeffs": POLY_COEFFS,
    }
