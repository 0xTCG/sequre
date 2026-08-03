"""MPyC column of the core-operations benchmark.

Three parties, Shamir secret sharing over a prime field, threshold 1, no
dealer -- MPyC's default configuration.

Unlike the activation benchmark, where every entry had to be built out of MPyC
primitives because MPyC ships no function library, every operation here *is* an
MPyC primitive: `mpc.input`, `+`, `*`, `mpc.np_matmul`, `mpc.output`. Nothing
is reimplemented.

Run (MPyC spawns the three parties itself):
    python bench_core.py -M3
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
import spec

FRAMEWORK = "mpyc"

# 64-bit fixed point with 32 fractional bits, matching Sequre's MPC_NBIT_F for
# a 128-bit integer size. CrypTen is fixed at 16 fractional bits and cannot be
# raised without changing its ring, which is why its accuracy column differs.
SECFXP_BITS = 64


def _polynomial(x):
    """Horner over spec.POLY_COEFFS, highest order first."""
    acc = None
    for c in reversed(spec.POLY_COEFFS):
        acc = c if acc is None else acc * x + c
    return acc


def _bytes_sent() -> int:
    return sum(p.protocol.nbytes_sent for p in mpc.parties if p.pid != mpc.pid)


async def _run(sizes: list[int], reps: int, ops: list[str], max_n: int) -> list[common.Record]:
    await mpc.start()
    secfxp = mpc.SecFxp(SECFXP_BITS)
    rank = mpc.pid
    records: list[common.Record] = []

    for op in ops:
        for n in sizes:
            # The cap exists because MPyC's secret multiplications are slow
            # enough to dominate the suite's wall clock. The ops with no
            # secret multiplication in them are local and cheap at any size,
            # so they are measured at every size rather than capped.
            if n > max_n and spec.OPS[op]["secret_ops"] > 0:
                records.append(common.skipped(
                    FRAMEWORK, op, n, rank,
                    f"n > --max-n={max_n}; MPyC is a pure-Python runtime"))
                continue

            values_a = spec.inputs(op, n)
            values_b = spec.inputs_b(op, n)
            array_a = secfxp.array(np.array(values_a))
            array_b = secfxp.array(np.array(values_b))

            # MPyC is async: the timed region has to include awaiting the
            # result, otherwise it measures how fast coroutines are scheduled.
            # `share` and `open` therefore time mpc.input / mpc.output alone,
            # and the rest time the operation plus the output that forces it.
            if op == "share":
                async def body():
                    return await mpc.output(mpc.input(array_a, senders=0))
            else:
                shared_a = mpc.input(array_a, senders=0)
                shared_b = mpc.input(array_b, senders=0)

                if op == "open":
                    async def body(x=shared_a):
                        return await mpc.output(x)
                elif op == "add":
                    async def body(x=shared_a, y=shared_b):
                        return await mpc.output(x + y)
                elif op == "mul_public":
                    async def body(x=shared_a):
                        return await mpc.output(x * spec.PUBLIC_SCALAR)
                elif op == "mul":
                    async def body(x=shared_a, y=shared_b):
                        return await mpc.output(x * y)
                elif op == "dot":
                    async def body(x=shared_a, y=shared_b):
                        return await mpc.output(mpc.np_matmul(x, y))
                elif op == "polynomial":
                    async def body(x=shared_a):
                        return await mpc.output(_polynomial(x))
                else:
                    raise ValueError(f"unknown op: {op}")

            for _ in range(spec.WARMUP_REPS):
                await body()

            times: list[float] = []
            revealed = None
            before = _bytes_sent()
            for _ in range(reps):
                start = time.perf_counter()
                revealed = await body()
                times.append(time.perf_counter() - start)
            sent = (_bytes_sent() - before) / reps

            expected = spec.ground_truth(op, values_a, values_b)
            got = np.atleast_1d(np.asarray(revealed, dtype=float)).tolist()
            records.append(common.make_record(
                FRAMEWORK, op, n, times, rank,
                bytes_sent=sent, got=got, expected=expected,
                note=f"mpyc.{op}"))

    await mpc.shutdown()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=spec.REPS)
    parser.add_argument("--sizes", type=str, default=",".join(str(s) for s in spec.SIZES))
    parser.add_argument("--ops", type=str, default=",".join(spec.OPS))
    parser.add_argument("--max-n", type=int, default=spec.MPYC_DEFAULT_MAX_N,
                        help="sizes above this are recorded as skipped")
    # MPyC consumes its own flags (-M, -I, ...) from sys.argv.
    args, _ = parser.parse_known_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    ops = args.ops.split(",")

    records = mpc.run(_run(sizes, args.reps, ops, args.max_n))

    # Every party runs this file; only party 0 writes, so three concurrent
    # processes do not race on the same output path.
    if mpc.pid != 0:
        return

    meta = {
        "framework": FRAMEWORK,
        "parties": len(mpc.parties),
        "threshold": mpc.threshold,
        "sectype": f"SecFxp({SECFXP_BITS})",
        "note": "every op is an MPyC primitive; nothing reimplemented",
    }
    path = common.write_jsonl(FRAMEWORK, records, meta)

    print(f"\nMPyC ({meta['parties']} parties, Shamir t={meta['threshold']}, {meta['sectype']})")
    common.print_table(records)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
