"""MPyC column of the activation benchmark.

Three parties, Shamir secret sharing over a prime field, threshold 1, no
dealer -- MPyC's default configuration. MPyC ships secure fixed-point
arithmetic, comparison and division, but no function library at all, so every
entry is built here from MPyC primitives. Where CrypTen has an implementation
the algorithm is matched to it exactly:

    exp      limit approximation, (1 + x/2^k)^(2^k), k = spec-matched to
             CrypTen's exp_iterations
    sigmoid  1 / (1 + exp(-x)), division by MPyC's own secure reciprocal
    tanh     2 * sigmoid(2x) - 1
    relu     x * (1 - [x < 0]), comparison by MPyC's np_sgn
    sin/cos  Taylor series, Horner in x^2 (CrypTen uses a complex-exponential
             trick that has no MPyC equivalent, so this one differs)
    tan/cot  sin/cos, divided by MPyC's secure reciprocal
    sinh     (exp(x) - exp(-x)) / 2
    cosh     (exp(x) + exp(-x)) / 2
    poly     Horner over spec.POLY_COEFFS

Holding the algorithm fixed against CrypTen is deliberate: the CrypTen/MPyC gap
then isolates runtime cost (vectorised C++ kernels vs pure Python), and the
Sequre/Decor gap isolates the protocol, since Decor evaluates the function
exactly rather than approximating it.

Run (MPyC spawns the three parties itself):
    python bench_activations.py -M3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np
from mpyc.runtime import mpc

import common
import spec

FRAMEWORK = "mpyc"

# Matches crypten's cfg.functions.exp_iterations default, so the two runners
# evaluate the same approximation to the same order.
EXP_ITERATIONS = 8

# 64-bit fixed point with 32 fractional bits, matching Sequre's MPC_NBIT_F for
# a 128-bit integer size. CrypTen is fixed at 16 fractional bits and cannot be
# raised without changing its ring, which is why its accuracy column differs.
SECFXP_BITS = 64


def _exp(x):
    """(1 + x / 2^k)^(2^k) -- k squarings, no division."""
    y = 1 + x / (1 << EXP_ITERATIONS)
    for _ in range(EXP_ITERATIONS):
        y = y * y
    return y


def _sigmoid(x):
    return 1 / (1 + _exp(-x))


def _tanh(x):
    return 2 * _sigmoid(2 * x) - 1


def _relu(x):
    return x * (1 - mpc.np_sgn(x, LT=True))


# Degree of the Taylor series used for sin/cos. MPyC has no trigonometry, and
# neither does it have CrypTen's complex-exponential trick, so the textbook
# series is used. Terms through x^15 keep the error near 1e-5 at |x| = pi,
# which is the worst point of the benchmark's interval.
TRIG_TERMS = 8


def _sin(x):
    """x * P(x^2), Horner over the odd Taylor coefficients."""
    u = x * x
    acc = None
    for k in range(TRIG_TERMS - 1, -1, -1):
        c = (-1) ** k / math.factorial(2 * k + 1)
        acc = c if acc is None else acc * u + c
    return x * acc


def _cos(x):
    u = x * x
    acc = None
    for k in range(TRIG_TERMS - 1, -1, -1):
        c = (-1) ** k / math.factorial(2 * k)
        acc = c if acc is None else acc * u + c
    return acc


def _sinh(x):
    return (_exp(x) - _exp(-x)) / 2


def _cosh(x):
    return (_exp(x) + _exp(-x)) / 2


def _polynomial(x):
    """Horner over spec.POLY_COEFFS, highest order first."""
    acc = None
    for c in reversed(spec.POLY_COEFFS):
        acc = c if acc is None else acc * x + c
    return acc


def _apply(x, function: str):
    if function == "exp":
        return _exp(x)
    if function == "sigmoid":
        return _sigmoid(x)
    if function == "tanh":
        return _tanh(x)
    if function == "relu":
        return _relu(x)
    if function == "sin":
        return _sin(x)
    if function == "cos":
        return _cos(x)
    if function == "tan":
        return _sin(x) / _cos(x)
    if function == "cot":
        return _cos(x) / _sin(x)
    if function == "sinh":
        return _sinh(x)
    if function == "cosh":
        return _cosh(x)
    if function == "polynomial":
        return _polynomial(x)
    raise ValueError(f"unknown function: {function}")


def _bytes_sent() -> int:
    return sum(p.protocol.nbytes_sent for p in mpc.parties if p.pid != mpc.pid)


async def _run(sizes: list[int], reps: int, functions: list[str], max_n: int) -> list[common.Record]:
    await mpc.start()
    secfxp = mpc.SecFxp(SECFXP_BITS)
    rank = mpc.pid
    records: list[common.Record] = []

    for function in functions:
        interval = spec.FUNCTIONS[function]["interval"]
        for n in sizes:
            if n > max_n:
                records.append(common.skipped(
                    FRAMEWORK, function, n, rank,
                    f"n > --max-n={max_n}; MPyC is a pure-Python runtime"))
                continue

            values = spec.inputs(function, n)
            shared = mpc.input(secfxp.array(np.array(values)), senders=0)

            # MPyC is async: the timed region has to include awaiting the
            # result, otherwise it measures how fast coroutines are scheduled.
            async def body():
                return await mpc.output(_apply(shared, function))

            for _ in range(spec.WARMUP_REPS):
                await body()

            times: list[float] = []
            revealed = None
            before = _bytes_sent()
            for _ in range(reps):
                import time as _time
                start = _time.perf_counter()
                revealed = await body()
                times.append(_time.perf_counter() - start)
            sent = (_bytes_sent() - before) / reps

            expected = spec.ground_truth(function, values)
            records.append(common.make_record(
                FRAMEWORK, function, n, times, rank,
                bytes_sent=sent,
                got=[float(v) for v in revealed], expected=expected,
                note=f"interval={interval}"))

    await mpc.shutdown()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=spec.REPS)
    parser.add_argument("--sizes", type=str, default=",".join(str(s) for s in spec.SIZES))
    parser.add_argument("--functions", type=str, default=",".join(spec.FUNCTIONS))
    parser.add_argument("--max-n", type=int, default=spec.MPYC_DEFAULT_MAX_N,
                        help="sizes above this are recorded as skipped")
    # MPyC consumes its own flags (-M, -I, ...) from sys.argv.
    args, _ = parser.parse_known_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    functions = args.functions.split(",")

    records = mpc.run(_run(sizes, args.reps, functions, args.max_n))

    # Every party runs this file; only party 0 writes, so three concurrent
    # processes do not race on the same output path.
    if mpc.pid != 0:
        return

    meta = {
        "framework": FRAMEWORK,
        "parties": len(mpc.parties),
        "threshold": mpc.threshold,
        "sectype": f"SecFxp({SECFXP_BITS})",
        "exp_iterations": EXP_ITERATIONS,
        "note": "activations implemented from MPyC primitives; MPyC ships no activation library",
    }
    path = common.write_jsonl(FRAMEWORK, records, meta)

    print(f"\nMPyC ({meta['parties']} parties, Shamir t={meta['threshold']}, {meta['sectype']})")
    common.print_table(records)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
