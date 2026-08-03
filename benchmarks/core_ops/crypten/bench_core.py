"""CrypTen column of the core-operations benchmark.

Two online parties, Beaver protocol. Correlated randomness comes from one of
two providers, and the choice is a security decision, not a tuning knob:

    TFP  TrustedFirstParty, CrypTen's shipped default. Party 0 generates the
         Beaver triples (a, b, c=a*b) in the clear and shares them. With only
         two parties that is not a secure configuration: Beaver multiplication
         opens x-a and y-b, and party 0 knows a and b, so it reconstructs both
         private inputs outright. Useful as a speed ceiling, not as a
         security-matched comparison.

    TTP  TrustedThirdParty. A separate TTPServer process supplies the triples,
         so neither online party sees them. This is the configuration that is
         structurally comparable to Sequre's 3-party model.

Every operation here is a CrypTen tensor method -- `+`, `*`, `matmul`,
`polynomial`, `cryptensor`, `get_plain_text`. Nothing is reimplemented, which
is the point of this suite.

Run:
    python bench_core.py [--reps N] [--sizes 8,128,...] [--provider TFP|TTP]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common
import spec

FRAMEWORK = "crypten"
WORLD_SIZE = 2


def _framework_id(provider: str) -> str:
    """TFP keeps the bare name for continuity; TTP is a distinct column."""
    return FRAMEWORK if provider == "TFP" else f"{FRAMEWORK}-{provider.lower()}"


def _run(sizes: list[int], reps: int, ops: list[str], provider: str) -> list[dict]:
    import crypten
    import crypten.communicator as comm
    import torch
    from crypten.config import cfg

    # Byte/round accounting is off by default because it costs a little
    # synchronisation; the benchmark needs it, so it is on for every party.
    cfg.communicator.verbose = True
    # Set inside the worker too: run_multiprocess spawns fresh processes and
    # the parent's cfg does not necessarily follow.
    cfg.mpc.provider = provider

    crypten.init()
    rank = comm.get().get_rank()
    records: list[common.Record] = []

    for op in ops:
        for n in sizes:
            values_a = spec.inputs(op, n)
            values_b = spec.inputs_b(op, n)
            plain_a = torch.tensor(values_a, dtype=torch.float64)
            plain_b = torch.tensor(values_b, dtype=torch.float64)

            # `share` and `open` time the encoding and decoding themselves, so
            # for those two the operand is prepared but not pre-encrypted.
            if op == "share":
                def body(p=plain_a):
                    return crypten.cryptensor(p.float())
            elif op == "open":
                enc = crypten.cryptensor(plain_a.float())

                def body(e=enc):
                    return e.get_plain_text()
            else:
                enc_a = crypten.cryptensor(plain_a.float())
                enc_b = crypten.cryptensor(plain_b.float())

                if op == "add":
                    def body(x=enc_a, y=enc_b):
                        return x + y
                elif op == "mul_public":
                    def body(x=enc_a):
                        return x * spec.PUBLIC_SCALAR
                elif op == "mul":
                    def body(x=enc_a, y=enc_b):
                        return x * y
                elif op == "dot":
                    def body(x=enc_a, y=enc_b):
                        return x.matmul(y)
                elif op == "polynomial":
                    # CrypTen takes coefficients from the linear term up, with
                    # the constant excluded, so it is added back afterwards.
                    def body(x=enc_a):
                        return x.polynomial(spec.POLY_COEFFS[1:]) + spec.POLY_COEFFS[0]
                else:
                    raise ValueError(f"unknown op: {op}")

            comm.get().reset_communication_stats()
            times, out = common.time_reps(body, reps=reps)
            stats = comm.get().get_communication_stats()

            # Averaged over warmup+timed calls, so the per-call figure is not
            # inflated by the warmup allocation of triples.
            calls = reps + spec.WARMUP_REPS

            if op == "open":
                revealed = out.double().flatten().tolist()
            else:
                revealed = out.get_plain_text().double().flatten().tolist()
            expected = spec.ground_truth(op, values_a, values_b)

            records.append(common.make_record(
                _framework_id(provider), op, n, times, rank,
                bytes_sent=stats["bytes"] / calls,
                rounds=stats["rounds"] / calls,
                got=revealed, expected=expected,
                note=f"crypten.{op}"))

    return [r.__dict__ for r in records]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=spec.REPS)
    parser.add_argument("--sizes", type=str, default=",".join(str(s) for s in spec.SIZES))
    parser.add_argument("--ops", type=str, default=",".join(spec.OPS))
    parser.add_argument("--provider", type=str, default="TFP", choices=["TFP", "TTP"],
                        help="TFP is insecure with 2 parties; TTP adds a real third party")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    ops = args.ops.split(",")

    import crypten.mpc as mpc_module
    from crypten.config import cfg

    if args.provider == "TTP":
        # CrypTen launches the TTPServer as a plain multiprocessing.Process and
        # gives it no way to receive configuration. Its own code comments
        # ("This process will be forked") assume the fork start method, which
        # lets the child inherit cfg. macOS defaults to spawn, so the server
        # would come up with cfg.mpc.provider still at its default, decide no
        # TTP was required, and size the process group without itself --
        # failing in gloo with "rank < size. 2 vs 2".
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)

    # Must be set before run_multiprocess: it decides whether a TTPServer
    # process is spawned alongside the two online parties.
    cfg.mpc.provider = args.provider

    runner = mpc_module.run_multiprocess(world_size=WORLD_SIZE)(_run)
    per_party = runner(sizes, args.reps, ops, args.provider)
    if per_party is None:
        raise SystemExit("crypten run_multiprocess returned no results")

    # Party 0 is the trusted first party as well as an online party, so it does
    # strictly more work; party 1 is the honest online-cost figure. Both are
    # kept in the file and the report picks the non-dealer party.
    records = [common.Record(**row) for party in per_party for row in party]

    meta = {
        "framework": FRAMEWORK,
        "parties": WORLD_SIZE,
        "provider": str(cfg.mpc.provider),
        "protocol": str(cfg.mpc.protocol),
        "precision_bits": cfg.encoder.precision_bits,
        "ring_bits": 64,
        "note": "every op is a CrypTen tensor method; nothing reimplemented",
    }
    path = common.write_jsonl(_framework_id(args.provider), records, meta)

    print(f"\nCrypTen ({WORLD_SIZE} parties, provider={args.provider}, "
          f"{meta['precision_bits']} fractional bits)")
    common.print_table([r for r in records if r.party == WORLD_SIZE - 1])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
