"""Single source of truth for the neural-network benchmark.

Every runner -- Sequre, CrypTen, MPyC -- draws its architecture, its data, its
initial weights and its hyperparameters from here, so the columns of the report
describe the same training run. The Codon runner cannot import Python, so it
mirrors these constants in `sequre/bench_nn.codon` and echoes them into its
output header; `report.py` refuses to build a table if the echoed values
disagree with this file.

Two networks are benchmarked. Both are sequential feed-forward nets, which is
what Sequre's `Sequential` currently supports -- the convolutional and recurrent
layers are open pull requests, not shipped code.

    siren   Sitzmann et al. 2020, "Implicit Neural Representations with
            Periodic Activation Functions". A coordinate network: it maps (x, y)
            to a scalar intensity through sine activations, fitting an image as
            a continuous function. Chosen because its activation is `sin`, which
            Decor evaluates exactly in three rounds -- the one architecture in
            common use whose nonlinearity is the best case for this protocol
            rather than the worst.

    mlp     Mohassel & Zhang 2017, "SecureML", Section 5.1: a 784-128-128-10
            fully-connected ReLU network. This is the reference workload of the
            secure-ML literature -- SecureML, MiniONN, ABY3, Falcon and CrypTen
            all report it -- so it is the architecture on which Sequre's numbers
            are directly comparable to published ones. Its activation is a
            comparison, which is the worst case for Decor and the case every
            other framework has optimised.

Weights are initialized from the deterministic generator below rather than from
each framework's RNG, so all three train from the identical starting point and
their final losses can be compared to each other and to a float64 reference.
"""

from __future__ import annotations

import math


SPEC_VERSION = "1"


# ---------------------------------------------------------------------------
# Deterministic pseudo-randomness.
#
# Reproducing an RNG stream across Codon, PyTorch and pure Python is not
# possible in general, so the benchmark does not try: it defines its own
# generator, small enough to mirror in three languages in five lines, and every
# framework draws data and initial weights from it. A 63-bit LCG is more than
# adequate for values that only need to be arbitrary and identical.
# ---------------------------------------------------------------------------

LCG_MULT = 6364136223846793005
LCG_ADD = 1442695040888963407
LCG_MASK = 0x7FFFFFFFFFFFFFFF
LCG_RANGE = 2000001  # values land on a 1e-6 grid in [-1, 1]


def uniform_pm1(count: int, seed: int) -> list[float]:
    """`count` values in [-1, 1], identical in Codon, Python and PyTorch."""
    out: list[float] = []
    state = seed
    for _ in range(count):
        state = (state * LCG_MULT + LCG_ADD) & LCG_MASK
        out.append((state % LCG_RANGE) / 1000000.0 - 1.0)
    return out


# ---------------------------------------------------------------------------
# Architectures.
#
# `layers` is (activation, width, pre_act_scale, kernel_scale) per layer after
# the input. kernel_scale is the half-width of the uniform init: SIREN's paper
# prescribes 1/fan_in for the first sine layer and sqrt(6/fan_in)/omega for the
# rest, and PyTorch's nn.Linear default 1/sqrt(fan_in) is what the MLP uses.
# ---------------------------------------------------------------------------

SIREN_OMEGA = 30.0
SIREN_HIDDEN = 64
SIREN_IN = 2
SIREN_OUT = 1

MLP_IN = 784
MLP_HIDDEN = 128
MLP_OUT = 10


def siren_layers() -> list[tuple[str, int, float, float]]:
    inner = math.sqrt(6.0 / SIREN_HIDDEN) / SIREN_OMEGA
    return [
        ("sin", SIREN_HIDDEN, SIREN_OMEGA, 1.0 / SIREN_IN),
        ("sin", SIREN_HIDDEN, SIREN_OMEGA, inner),
        ("linear", SIREN_OUT, 1.0, inner),
    ]


def mlp_layers() -> list[tuple[str, int, float, float]]:
    return [
        ("relu", MLP_HIDDEN, 1.0, 1.0 / math.sqrt(MLP_IN)),
        ("relu", MLP_HIDDEN, 1.0, 1.0 / math.sqrt(MLP_HIDDEN)),
        ("linear", MLP_OUT, 1.0, 1.0 / math.sqrt(MLP_HIDDEN)),
    ]


MODELS: dict[str, dict] = {
    "siren": {
        "in_features": SIREN_IN,
        "out_features": SIREN_OUT,
        # n is the number of pixels, i.e. a full res x res grid for
        # res in 8, 16, 32, 64. SIREN trains on every pixel of one image, so a
        # "batch" and "the dataset" are the same thing here.
        "sizes": [64, 256, 1024, 4096],
        "loss": "mse",
        "step": 0.01,
        "description": "SIREN coordinate network, 2-64-64-1, sine activations, omega=30",
    },
    "mlp": {
        "in_features": MLP_IN,
        "out_features": MLP_OUT,
        # n is the number of training samples in the full-batch gradient step.
        "sizes": [8, 32, 128, 512],
        "loss": "mse",
        "step": 0.01,
        "description": "SecureML MLP, 784-128-128-10, ReLU activations",
    },
}

MODELS["siren"]["layers"] = siren_layers()
MODELS["mlp"]["layers"] = mlp_layers()


# Training epochs per timed run. Small on purpose: the benchmark measures the
# per-epoch cost of secure training and the fidelity of the resulting weights,
# not convergence -- nobody trains a network to completion under MPC, and a
# framework that was 20% faster over 3 epochs is 20% faster over 300.
EPOCHS = 2

# Timed repetitions per (model, size). The median is reported. Each repetition
# trains a fresh network, since training is stateful.
REPS = 3
WARMUP_REPS = 1

MOMENTUM = 0.9

# Nesterov momentum, batch gradient descent. Both are what Sequre's Sequential
# implements, and both are reproduced exactly by the other runners rather than
# substituting each framework's stock optimizer -- an optimizer difference would
# show up as an accuracy difference and be misread as a protocol difference.
OPTIMIZER = "bgd-nesterov"

# Seeds. Separate streams so that changing the data size does not change the
# initial weights.
DATA_SEED = 20250728
WEIGHT_SEED = 987654321

# MPyC is a pure-Python runtime with no neural-network layer at all; its runner
# builds the network from primitives. At the MLP's 784x128 first layer that is
# hundreds of thousands of secure multiplications per matmul, so sizes above
# this are recorded as skipped rather than silently dropped.
MPYC_DEFAULT_MAX_N = 8

# Party counts. These differ by design -- see README.md.
PARTIES = {
    "sequre-64": 3,     # width-matched control against CrypTen, see README
    "sequre-128": 3,
    "sequre-192": 3,
    "crypten": 2,       # TFP: 2 parties, one of which makes the triples (insecure)
    "crypten-ttp": 3,   # TTP: 2 online parties + a separate TTPServer
    "mpyc": 3,          # Shamir, threshold 1, no dealer
}


# ---------------------------------------------------------------------------
# Data.
#
# Synthetic and analytic rather than MNIST or a photograph. Three runners in
# three languages have to agree bit-for-bit on the training set, and the thing
# being measured is the cost of the protocol per unit of arithmetic, which a
# real dataset would not change. The SIREN target is a genuine band-limited
# image -- exactly the kind of signal SIREN is designed to fit -- so its fit
# quality is still meaningful.
# ---------------------------------------------------------------------------

def siren_data(n: int) -> tuple[list[list[float]], list[list[float]]]:
    """A res x res coordinate grid over [-1, 1]^2 and the target image on it."""
    res = round(math.sqrt(n))
    assert res * res == n, f"siren size {n} is not a square number of pixels"
    coords: list[list[float]] = []
    target: list[list[float]] = []
    for i in range(res):
        for j in range(res):
            x = -1.0 + 2.0 * i / (res - 1)
            y = -1.0 + 2.0 * j / (res - 1)
            coords.append([x, y])
            target.append([0.5 * math.sin(3.0 * x) * math.cos(4.0 * y)])
    return coords, target


def mlp_data(n: int) -> tuple[list[list[float]], list[list[float]]]:
    """n feature vectors in [-1, 1]^784 and cyclic one-hot labels over 10 classes."""
    flat = uniform_pm1(n * MLP_IN, DATA_SEED)
    X = [flat[i * MLP_IN:(i + 1) * MLP_IN] for i in range(n)]
    y = [[1.0 if (i % MLP_OUT) == c else 0.0 for c in range(MLP_OUT)] for i in range(n)]
    return X, y


def data(model: str, n: int) -> tuple[list[list[float]], list[list[float]]]:
    if model == "siren":
        return siren_data(n)
    if model == "mlp":
        return mlp_data(n)
    raise ValueError(f"unknown model: {model}")


def initial_weights(model: str) -> list[tuple[list[list[float]], list[list[float]]]]:
    """(weights, bias) per layer, in layer order, drawn from the shared stream.

    One stream for the whole network, consumed layer by layer, weights before
    bias -- the order the runners must reproduce.
    """
    spec = MODELS[model]
    fan_in = spec["in_features"]
    state = WEIGHT_SEED
    out: list[tuple[list[list[float]], list[list[float]]]] = []

    for _, width, _, scale in spec["layers"]:
        count = fan_in * width + width
        stream = uniform_pm1(count, state)
        # Advance the stream deterministically for the next layer.
        state = (state + count) & LCG_MASK
        w = [[stream[r * width + c] * scale for c in range(width)] for r in range(fan_in)]
        b = [[stream[fan_in * width + c] * scale for c in range(width)]]
        out.append((w, b))
        fan_in = width

    return out


# ---------------------------------------------------------------------------
# Accuracy witness.
#
# Every runner emits its trained network's predictions at these row indices,
# and report.py compares them against ref.py's float64 run of the identical
# training. Capped rather than complete so a result file stays small: at 4096
# pixels the full prediction vector would dominate the JSONL. The rows are
# evenly spaced over the training set, so for the SIREN grid they sweep the
# whole image rather than one corner of it.
# ---------------------------------------------------------------------------

WITNESS_POINTS = 256


def witness_indices(n: int) -> list[int]:
    if n <= WITNESS_POINTS:
        return list(range(n))
    stride = n / WITNESS_POINTS
    return [int(i * stride) for i in range(WITNESS_POINTS)]


def header() -> dict:
    """Spec fingerprint embedded in every result file, checked by report.py."""
    return {
        "spec_version": SPEC_VERSION,
        "epochs": EPOCHS,
        "reps": REPS,
        "momentum": MOMENTUM,
        "optimizer": OPTIMIZER,
        "witness_points": WITNESS_POINTS,
        "models": {
            name: {
                "sizes": m["sizes"],
                "in_features": m["in_features"],
                "out_features": m["out_features"],
                "loss": m["loss"],
                "step": m["step"],
                "layers": [[a, w, p, s] for a, w, p, s in m["layers"]],
            }
            for name, m in MODELS.items()
        },
    }
