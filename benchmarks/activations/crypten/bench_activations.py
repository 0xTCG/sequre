"""CrypTen column of the activation benchmark.

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
         structurally comparable to Sequre's 3-party model, where CP0 is a
         dealer holding no share of the data.

CrypTen's third provider, HomomorphicProvider ("HE"), raises
NotImplementedError in 0.4.1 and cannot be benchmarked.

Every activation is CrypTen's own library implementation, so this measures
CrypTen as a user would actually get it:

    exp      limit approximation, (1 + x/2^n)^(2^n), n = exp_iterations
    sigmoid  1 / (1 + exp(-x)), reciprocal by Newton-Raphson
    tanh     2 * sigmoid(2x) - 1
    relu     x * (x > 0), comparison by arithmetic-to-binary conversion

Run:
    python bench_activations.py [--reps N] [--sizes 8,128,...]
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


# CrypTen implements exp/sigmoid/tanh/relu/sin/cos/polynomial itself. It has no
# tan, cot, sinh or cosh, so those are composed from CrypTen's *own* primitives
# in the textbook way rather than approximated by hand -- tan = sin/cos, and the
# hyperbolics from CrypTen's exp. Each composed function is flagged in the
# result row's note so the report never implies a native implementation.
_COMPOSED = {"tan", "cot", "sinh", "cosh"}


def _apply(x, function: str):
    if function == "exp":
        return x.exp()
    if function == "sigmoid":
        return x.sigmoid()
    if function == "tanh":
        return x.tanh()
    if function == "relu":
        return x.relu()
    if function == "sin":
        return x.sin()
    if function == "cos":
        return x.cos()
    if function == "tan":
        return x.sin().div(x.cos())
    if function == "cot":
        return x.cos().div(x.sin())
    if function == "sinh":
        return (x.exp() - (-x).exp()) * 0.5
    if function == "cosh":
        return (x.exp() + (-x).exp()) * 0.5
    if function == "polynomial":
        # CrypTen takes coefficients from the linear term up, constant excluded.
        return x.polynomial(spec.POLY_COEFFS[1:]) + spec.POLY_COEFFS[0]
    raise ValueError(f"unknown function: {function}")


def _run(sizes: list[int], reps: int, functions: list[str], provider: str) -> list[dict]:
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

    for function in functions:
        interval = spec.FUNCTIONS[function]["interval"]
        for n in sizes:
            values = spec.inputs(function, n)
            plain = torch.tensor(values, dtype=torch.float64)
            enc = crypten.cryptensor(plain.float())

            def body():
                return _apply(enc, function)

            comm.get().reset_communication_stats()
            times, out = common.time_reps(body, reps=reps)
            stats = comm.get().get_communication_stats()

            # Averaged over warmup+timed calls, so the per-call figure is not
            # inflated by the warmup allocation of triples.
            calls = reps + spec.WARMUP_REPS
            revealed = out.get_plain_text().double().tolist()
            expected = spec.ground_truth(function, values)

            records.append(common.make_record(
                _framework_id(provider), function, n, times, rank,
                bytes_sent=stats["bytes"] / calls,
                rounds=stats["rounds"] / calls,
                got=revealed, expected=expected,
                note=("composed from CrypTen primitives; no native implementation"
                      if function in _COMPOSED else f"interval={interval}")))

    return [r.__dict__ for r in records]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=spec.REPS)
    parser.add_argument("--sizes", type=str, default=",".join(str(s) for s in spec.SIZES))
    parser.add_argument("--functions", type=str, default=",".join(spec.FUNCTIONS))
    parser.add_argument("--provider", type=str, default="TFP", choices=["TFP", "TTP"],
                        help="TFP is insecure with 2 parties; TTP adds a real third party")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    functions = args.functions.split(",")

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
    per_party = runner(sizes, args.reps, functions, args.provider)
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
        "exp_iterations": cfg.functions.exp_iterations,
        "reciprocal_method": str(cfg.functions.reciprocal_method),
        "reciprocal_nr_iters": cfg.functions.reciprocal_nr_iters,
        "sigmoid_tanh_method": str(cfg.functions.sigmoid_tanh_method),
    }
    path = common.write_jsonl(_framework_id(args.provider), records, meta)

    print(f"\nCrypTen ({WORLD_SIZE} parties, provider={args.provider}, "
          f"{meta['precision_bits']} fractional bits)")
    common.print_table([r for r in records if r.party == WORLD_SIZE - 1])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
