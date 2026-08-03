"""MPyC column of the neural-network benchmark.

Three parties, Shamir secret sharing over a prime field, threshold 1, no
dealer -- MPyC's default configuration. MPyC ships secure fixed-point
arithmetic, secure matrix multiplication and a secure sign test, and nothing
above them: there is no layer, no autograd, no optimizer and no loss. So the
network, its backward pass and its update rule are all written out here from
primitives, and all of it is inside the LOC region. That is the point of the
LOC column -- this is what "no neural-network support" costs in practice.

**SIREN is not implemented, and the reason is a result rather than an
omission.** It needs a secure sine of a pre-activation scaled by omega = 30.
MPyC has no trigonometric function, so the only route is a Taylor series, and a
Taylor series of sin is usable only near zero. The degree-15 series -- the one
the activation benchmark's MPyC runner uses, where it holds ~1e-5 over
[-pi, pi] -- evaluates sin(30) as -8.8e9 against a true value of -0.988, and
-8.8e9 does not fit in SecFxp(64) either: with 32 fractional bits the range
tops out at +-2.1e9, so it wraps.

Making it work would need range reduction -- x mod 2*pi -- which needs a secure
modulo, i.e. a secure division and floor per element per layer per epoch. The
rows are recorded as skipped with this note rather than filled with a number
that would be reporting overflow as inaccuracy.

The MLP is a faithful transcription of the same network the other two columns
train: same architecture, same initial weights, same mean squared error, same
Nesterov update in Sequre's formulation.

Run (MPyC spawns the three parties itself):
    python bench_nn.py -M3 [--max-n 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mpyc.runtime import mpc

import common
import loc
import ref
import spec

FRAMEWORK = "mpyc"

# 64-bit fixed point with 32 fractional bits, matching Sequre's MPC_NBIT_F.
# CrypTen is fixed at 16 fractional bits and cannot be raised without changing
# its ring, which is why its accuracy column differs from both of these.
SECFXP_BITS = 64


# ---------------------------------------------------------------------------
# The model.
#
# Everything between the LOC-BEGIN/LOC-END markers is the code a user writes to
# define and train the network; ../loc.py counts it. Data, weight injection,
# timing and output are deliberately outside.
# ---------------------------------------------------------------------------

# LOC-BEGIN mlp
def relu(x):
    """max(x, 0), from MPyC's secure sign test."""
    return x * (1 - mpc.np_sgn(x, LT=True))


def drelu(x):
    return 1 - mpc.np_sgn(x, LT=True)


def forward(weights, biases, activations, X):
    """Returns the pre-activation and the output of every layer."""
    pre_acts, outputs = [], []
    out = X
    for i in range(len(weights)):
        z = mpc.np_matmul(out, weights[i]) + biases[i]
        out = relu(z) if activations[i] == "relu" else z
        pre_acts.append(z)
        outputs.append(out)
    return pre_acts, outputs


def backward(weights, activations, X, y, pre_acts, outputs):
    """Gradients of the row-normalised squared error, one layer at a time."""
    rows = len(y)
    dhidden = (outputs[-1] - y) / rows
    grad_w, grad_b = [None] * len(weights), [None] * len(weights)
    for i in range(len(weights) - 1, -1, -1):
        if activations[i] == "relu":
            dhidden = dhidden * drelu(pre_acts[i])
        prev = X if i == 0 else outputs[i - 1]
        grad_w[i] = mpc.np_matmul(mpc.np_transpose(prev), dhidden)
        grad_b[i] = mpc.np_sum(dhidden, axis=0).reshape(1, -1)
        if i > 0:
            dhidden = mpc.np_matmul(dhidden, mpc.np_transpose(weights[i]))
    return grad_w, grad_b


def train(weights, biases, activations, X, y, step, epochs, momentum):
    """Batch gradient descent with Nesterov momentum, in Sequre's formulation."""
    vw = [w * 0 for w in weights]
    vb = [b * 0 for b in biases]
    for _ in range(epochs):
        pre_acts, outputs = forward(weights, biases, activations, X)
        grad_w, grad_b = backward(weights, activations, X, y, pre_acts, outputs)
        for i in range(len(weights)):
            vw_prev, vb_prev = vw[i], vb[i]
            vw[i] = vw[i] * momentum - grad_w[i] * step
            vb[i] = vb[i] * momentum - grad_b[i] * step
            weights[i] = weights[i] + vw[i] * (momentum + 1) - vw_prev * momentum
            biases[i] = biases[i] + vb[i] * (momentum + 1) - vb_prev * momentum
    return weights, biases


def predict(weights, biases, activations, X):
    return forward(weights, biases, activations, X)[1][-1]


def loss(weights, biases, activations, X, y):
    """Elementwise (y - out)^2 / 2, normalised by the row count, summed."""
    out = predict(weights, biases, activations, X)
    diff = out - y
    return mpc.np_sum(diff * diff) / (2 * len(y))
# LOC-END mlp


# ---------------------------------------------------------------------------
# Benchmark scaffolding.
# ---------------------------------------------------------------------------

def _bytes_sent() -> int:
    return sum(p.protocol.nbytes_sent for p in mpc.parties if p.pid != mpc.pid)


def _share(secfxp, values) -> object:
    return mpc.input(secfxp.array(np.array(values, dtype=np.float64)), senders=0)


def _shared_weights(secfxp, model_name: str):
    """Fresh shares of the spec's starting point, one pair per layer."""
    weights, biases = [], []
    for w, b in spec.initial_weights(model_name):
        weights.append(_share(secfxp, w))
        biases.append(_share(secfxp, b))
    return weights, biases


async def _run(reps: int, max_n: int, models: list[str]) -> list[common.Record]:
    await mpc.start()
    secfxp = mpc.SecFxp(SECFXP_BITS)
    rank = mpc.pid
    records: list[common.Record] = []
    loc_counts = loc.counts().get(FRAMEWORK, {})

    for model_name in models:
        model_spec = spec.MODELS[model_name]
        activations = [a for a, _, _, _ in model_spec["layers"]]
        step = model_spec["step"]

        for n in model_spec["sizes"]:
            if model_name == "siren":
                records.append(common.skipped(
                    FRAMEWORK, model_name, n, rank,
                    "MPyC has no secure sine; see the module docstring"))
                continue
            if n > max_n:
                records.append(common.skipped(
                    FRAMEWORK, model_name, n, rank,
                    f"n > --max-n={max_n}; MPyC is a pure-Python runtime"))
                continue

            X_raw, y_raw = spec.data(model_name, n)
            X = _share(secfxp, X_raw)
            y = _share(secfxp, y_raw)

            # MPyC is async: the timed region has to include awaiting the
            # result, otherwise it measures how fast coroutines are scheduled.
            # A fresh set of weights per repetition, since training is stateful.
            async def body():
                weights, biases = _shared_weights(secfxp, model_name)
                weights, biases = train(
                    weights, biases, activations, X, y,
                    step, spec.EPOCHS, spec.MOMENTUM)
                await mpc.output(weights[-1])  # force the coroutines to complete
                return weights, biases

            for _ in range(spec.WARMUP_REPS):
                await body()

            times: list[float] = []
            trained = None
            before = _bytes_sent()
            for _ in range(reps):
                start = time.perf_counter()
                trained = await body()
                times.append(time.perf_counter() - start)
            sent = (_bytes_sent() - before) / reps

            weights, biases = trained
            predictions = np.array(
                await mpc.output(predict(weights, biases, activations, X)),
                dtype=np.float64)
            final_loss = float(await mpc.output(
                loss(weights, biases, activations, X, y)))

            indices = spec.witness_indices(n)
            witness = predictions[indices].reshape(-1).tolist()
            ref_pred, ref_loss, _ = ref.train(model_name, n)
            expected = ref_pred[indices].reshape(-1).tolist()

            records.append(common.make_record(
                FRAMEWORK, model_name, n, times, rank,
                bytes_sent=sent,
                final_loss=final_loss,
                ref_final_loss=ref_loss,
                got=witness, expected=expected,
                witness=witness,
                loc=loc_counts.get(model_name),
                note="network, backward pass and optimizer built from MPyC primitives"))

    await mpc.shutdown()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=spec.REPS)
    parser.add_argument("--models", type=str, default=",".join(spec.MODELS))
    parser.add_argument("--max-n", type=int, default=spec.MPYC_DEFAULT_MAX_N,
                        help="sizes above this are recorded as skipped")
    # MPyC consumes its own flags (-M, -I, ...) from sys.argv.
    args, _ = parser.parse_known_args()

    models = args.models.split(",")
    records = mpc.run(_run(args.reps, args.max_n, models))

    # Every party runs this file; only party 0 writes, so three concurrent
    # processes do not race on the same output path.
    if mpc.pid != 0:
        return

    meta = {
        "framework": FRAMEWORK,
        "parties": len(mpc.parties),
        "threshold": mpc.threshold,
        "sectype": f"SecFxp({SECFXP_BITS})",
        "note": "network built from MPyC primitives; MPyC ships no neural-network layer",
    }
    path = common.write_jsonl(FRAMEWORK, records, meta)

    print(f"\nMPyC ({meta['parties']} parties, Shamir t={meta['threshold']}, {meta['sectype']})")
    common.print_table(records)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
