"""Single source of truth for the core-operations benchmark.

`activations/` and `neural_nets/` measure functions -- `exp`, `sigmoid`,
`relu`, a trained MLP. Those numbers mix two things that move independently:
the cost of the underlying protocol, and the quality of the approximation each
framework's function library happens to use. A framework can lose the
activation benchmark by having a slow multiply or by having a wasteful sigmoid,
and the table cannot tell you which.

So this suite drops to the layer underneath and measures only that: sharing,
addition, multiplication by a public constant, multiplication of two secrets,
an inner product, a fixed polynomial, and reconstruction. Every framework here
provides all of them in its own library, so every cell is that framework's own
code and the rows isolate protocol cost from function-library cost.

The operation set is deliberately the *intersection*, not the union. Nothing is
here that any framework has to emulate. What that costs is expressiveness --
this benchmark says nothing about nonlinear functions, which is what the other
two suites are for -- and what it buys is that no cell needs a caveat about who
wrote it.

Every runner reads its inputs, sizes and repetition counts from this file. The
Codon runner cannot import Python, so it mirrors these constants and echoes
them into its output header; `report.py` refuses to build a table if the echoed
values disagree with this file.
"""

from __future__ import annotations


SPEC_VERSION = "1"


# ---------------------------------------------------------------------------
# The operations.
#
# `secret_ops` is the count of secret-by-secret multiplications the operation
# requires per element, which is what actually drives communication: 0 means
# the operation is local after sharing, and the row is a measurement of a
# framework's constant factors rather than of its protocol.
# ---------------------------------------------------------------------------
OPS: dict[str, dict] = {
    "share": {
        "interval": (-1.0, 1.0),
        "secret_ops": 0,
        "description": "secret-share a length-n vector of cleartext values",
    },
    "add": {
        "interval": (-1.0, 1.0),
        "secret_ops": 0,
        "description": "elementwise a + b, both secret -- local in every additive/Shamir scheme",
    },
    "mul_public": {
        "interval": (-1.0, 1.0),
        "secret_ops": 0,
        "description": "elementwise a * c for public c -- local, but incurs a fixed-point truncation",
    },
    "mul": {
        "interval": (-1.0, 1.0),
        "secret_ops": 1,
        "description": "elementwise a * b, both secret -- one Beaver multiplication per element",
    },
    "dot": {
        "interval": (-1.0, 1.0),
        "secret_ops": 1,
        "description": "inner product of two length-n secret vectors -- n multiplications, one truncation",
    },
    "polynomial": {
        "interval": (-2.0, 2.0),
        # Horner is ((4x + 3)x + 2)x + 1. The first product has a public left
        # operand and is local; only the two that follow are secret-by-secret.
        "secret_ops": 2,
        "description": "1 + 2x + 3x^2 + 4x^3 by Horner -- two sequential secret multiplications, a depth probe",
    },
    "open": {
        "interval": (-1.0, 1.0),
        "secret_ops": 0,
        "description": "reconstruct a length-n secret vector to cleartext",
    },
}

# Coefficients for the "polynomial" entry, lowest order first. Identical to
# activations/spec.py, so the two suites' polynomial rows cross-reference.
POLY_COEFFS: list[float] = [1.0, 2.0, 3.0, 4.0]
POLY_DEGREE = len(POLY_COEFFS) - 1

# The public multiplier for `mul_public`. An integer, and not a power of two:
# integer so that the product of two scale-2^f encodings divides back exactly
# and no framework's rounding policy leaks into the row, and not a power of two
# so that none of them can shortcut it into a bit shift.
PUBLIC_SCALAR = 3.0

# Vector lengths. 8 is a latency probe (protocol round-trips dominate); 8192 is
# a throughput probe (local arithmetic dominates).
SIZES: list[int] = [8, 128, 1024, 8192]

# Timed repetitions per (op, size). The median is reported; the min is kept as
# a floor estimate.
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
    "sequre-64": 3,      # CP0 offline dealer + CP1 + CP2 online
    "sequre-128": 3,
    "sequre-192": 3,
    "crypten": 2,        # TFP: 2 parties, one of which makes the triples (insecure)
    "crypten-ttp": 3,    # TTP: 2 online parties + a separate TTPServer
    "mpyc": 3,           # Shamir, threshold 1, no dealer
}


def inputs(op: str, n: int) -> list[float]:
    """The first operand: the exact vector every framework must consume.

    linspace rather than random draws, so it is reproducible across the
    runners' languages without having to match RNG streams, and it covers the interval
    endpoints, which is where fixed-point range problems show up first.
    """
    a, b = OPS[op]["interval"]
    if n == 1:
        return [(a + b) / 2]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def inputs_b(op: str, n: int) -> list[float]:
    """The second operand for the binary ops: the first, reversed.

    Reversal rather than a second linspace keeps the elementwise products
    varying across the vector -- `a * a` would make every product a square and
    hide a sign-handling bug in a truncation.
    """
    return list(reversed(inputs(op, n)))


def ground_truth(op: str, a: list[float], b: list[float]) -> list[float]:
    """float64 reference, computed in the clear."""
    if op in ("share", "open"):
        return list(a)
    if op == "add":
        return [x + y for x, y in zip(a, b)]
    if op == "mul_public":
        return [x * PUBLIC_SCALAR for x in a]
    if op == "mul":
        return [x * y for x, y in zip(a, b)]
    if op == "dot":
        return [sum(x * y for x, y in zip(a, b))]
    if op == "polynomial":
        return [sum(c * x**i for i, c in enumerate(POLY_COEFFS)) for x in a]
    raise ValueError(f"unknown op: {op}")


def header() -> dict:
    """Spec fingerprint embedded in every result file, checked by report.py."""
    return {
        "spec_version": SPEC_VERSION,
        "sizes": SIZES,
        "reps": REPS,
        "intervals": {k: list(v["interval"]) for k, v in OPS.items()},
        "poly_coeffs": POLY_COEFFS,
        "public_scalar": PUBLIC_SCALAR,
    }
