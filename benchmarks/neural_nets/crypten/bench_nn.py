"""CrypTen column of the neural-network benchmark.

CrypTen ships a real neural-network layer -- `crypten.nn` mirrors `torch.nn`,
with automatic differentiation over secret-shared tensors -- so this column uses
it, exactly as a user would. The MLP is a stock `nn.Sequential` of `Linear` and
`ReLU`; SIREN needs one custom module, because CrypTen has no sine layer, but
its backward pass still comes from CrypTen's autograd (`sin` is in the gradient
registry) rather than being written by hand.

Two deliberate departures from the most idiomatic CrypTen, both made so that
this column trains *the same network* as the Sequre column and their losses can
be compared:

  * The loss is written out as `((out - y)^2).sum() / (2 * rows)` instead of
    `nn.MSELoss()`. CrypTen's MSELoss averages over every element, Sequre's
    normalises by the row count and carries a factor of 1/2; the two give
    different gradients, and the difference would show up in the accuracy
    column as if it were a protocol effect.

  * The optimizer is written out instead of `crypten.optim.SGD(...,
    nesterov=True)`. PyTorch's Nesterov formulation and Sequre's are
    algebraically different variants, not two spellings of one update. Those
    lines are inside the LOC region and README.md says so, since they are an
    artifact of this benchmark rather than of CrypTen.

Both providers are available and the choice is a security decision:

    TFP  CrypTen's default. Party 0 makes the Beaver triples in the clear, so
         with two parties it can reconstruct party 1's inputs. A speed ceiling.
    TTP  A separate TTPServer supplies the triples -- structurally comparable to
         Sequre's 3-party model with an offline dealer.

Run:
    python bench_nn.py [--provider TFP|TTP] [--models siren,mlp]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crypten
import crypten.nn as cnn
import torch

import common
import loc
import ref
import spec

FRAMEWORK = "crypten"
WORLD_SIZE = 2


def _framework_id(provider: str) -> str:
    """TFP keeps the bare name for continuity; TTP is a distinct column."""
    return FRAMEWORK if provider == "TFP" else f"{FRAMEWORK}-{provider.lower()}"


# ---------------------------------------------------------------------------
# The models.
#
# Everything between the LOC-BEGIN/LOC-END markers is the code a user writes to
# define and train one network; ../loc.py counts it. Data, weight injection,
# timing and output are deliberately outside.
# ---------------------------------------------------------------------------

# LOC-BEGIN siren
class SineLayer(cnn.Module):
    """Linear, scaled by omega, through a sine. CrypTen's autograd covers sin."""

    def __init__(self, in_features, out_features, omega):
        super().__init__()
        self.omega = omega
        self.register_module("linear", cnn.Linear(in_features, out_features))

    def forward(self, x):
        return (self.linear(x) * self.omega).sin()


def siren_model():
    return cnn.Sequential(
        SineLayer(spec.SIREN_IN, spec.SIREN_HIDDEN, spec.SIREN_OMEGA),
        SineLayer(spec.SIREN_HIDDEN, spec.SIREN_HIDDEN, spec.SIREN_OMEGA),
        cnn.Linear(spec.SIREN_HIDDEN, spec.SIREN_OUT))
# LOC-END siren


# LOC-BEGIN mlp
def mlp_model():
    return cnn.Sequential(
        cnn.Linear(spec.MLP_IN, spec.MLP_HIDDEN), cnn.ReLU(),
        cnn.Linear(spec.MLP_HIDDEN, spec.MLP_HIDDEN), cnn.ReLU(),
        cnn.Linear(spec.MLP_HIDDEN, spec.MLP_OUT))
# LOC-END mlp


# LOC-BEGIN siren,mlp
def train(model, X, y, step, epochs, momentum):
    """Batch gradient descent with Nesterov momentum, in Sequre's formulation."""
    velocity = {name: param * 0.0 for name, param in model.named_parameters()}
    model.train()
    for _ in range(epochs):
        model.zero_grad()
        loss = (model(X) - y).square().sum() * (1.0 / (2 * X.size(0)))
        loss.backward()
        with crypten.no_grad():
            for name, param in model.named_parameters():
                previous = velocity[name]
                velocity[name] = velocity[name] * momentum - param.grad * step
                param.add_(velocity[name] * (momentum + 1) - previous * momentum)


def predict(model, X):
    model.eval()
    return model(X)
# LOC-END siren,mlp


# ---------------------------------------------------------------------------
# Benchmark scaffolding.
# ---------------------------------------------------------------------------

BUILDERS = {"siren": siren_model, "mlp": mlp_model}


def _inject(model, weights):
    """Loads the shared starting point over the modules' own initialization.

    CrypTen's Linear holds its kernel transposed relative to Sequre's -- it
    computes x @ W.T -- and its bias is 1-D rather than a 1-row matrix.
    """
    layers = [m for m in model.modules() if isinstance(m, cnn.Linear)]
    assert len(layers) == len(weights), (
        f"found {len(layers)} Linear modules, spec has {len(weights)} layers")
    for layer, (w, b) in zip(layers, weights):
        layer.set_parameter("weight", crypten.cryptensor(
            torch.tensor(w, dtype=torch.float32).t().contiguous()))
        layer.set_parameter("bias", crypten.cryptensor(
            torch.tensor(b[0], dtype=torch.float32)))
    return model


def _final_loss(model, X, y) -> float:
    """Sequre's loss: elementwise (y - out)^2 / 2, normalised by rows, summed."""
    with crypten.no_grad():
        out = predict(model, X)
        return float(((out - y).square().sum() * (1.0 / (2 * X.size(0)))).get_plain_text())


def _run(models: list[str], reps: int, provider: str,
         sizes: list[int] | None = None) -> list[dict]:
    import crypten.communicator as comm
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
    loc_counts = loc.counts().get(_framework_id(provider), {})

    for model_name in models:
        model_spec = spec.MODELS[model_name]
        weights = spec.initial_weights(model_name)
        step = model_spec["step"]

        for n in (sizes if sizes else model_spec["sizes"]):
            X_raw, y_raw = spec.data(model_name, n)
            X = crypten.cryptensor(torch.tensor(X_raw, dtype=torch.float32))
            y = crypten.cryptensor(torch.tensor(y_raw, dtype=torch.float32))

            # A fresh network per repetition: training is stateful, so reusing
            # one would time the second pair of epochs against the first.
            def body(model_name=model_name, weights=weights, X=X, y=y, step=step):
                model = _inject(BUILDERS[model_name]().encrypt(), weights)
                train(model, X, y, step, spec.EPOCHS, spec.MOMENTUM)
                return model

            comm.get().reset_communication_stats()
            times, model = common.time_reps(body, reps=reps)
            stats = comm.get().get_communication_stats()
            calls = reps + spec.WARMUP_REPS

            predictions = predict(model, X).get_plain_text().double()
            indices = spec.witness_indices(n)
            witness = predictions[indices].reshape(-1).tolist()

            ref_pred, ref_loss, _ = ref.train(model_name, n)
            expected = ref_pred[indices].reshape(-1).tolist()

            records.append(common.make_record(
                _framework_id(provider), model_name, n, times, rank,
                bytes_sent=stats["bytes"] / calls,
                rounds=stats["rounds"] / calls,
                final_loss=_final_loss(model, X, y),
                ref_final_loss=ref_loss,
                got=witness, expected=expected,
                witness=witness,
                loc=loc_counts.get(model_name),
                note=f"crypten.nn, {cfg.encoder.precision_bits} fractional bits"))

    # Written here, inside the worker, rather than returned to the parent.
    # CrypTen's run_multiprocess joins its children before draining the result
    # queue, so a child whose return value exceeds the pipe buffer can never
    # exit and the parent waits on it forever. The witness vectors are well
    # past that buffer; only the file path comes back.
    meta = {
        "framework": FRAMEWORK,
        "parties": WORLD_SIZE,
        "provider": str(cfg.mpc.provider),
        "protocol": str(cfg.mpc.protocol),
        "precision_bits": cfg.encoder.precision_bits,
    }
    path = common.write_jsonl(_framework_id(provider), records, meta, party=rank)

    if rank == WORLD_SIZE - 1:
        print(f"\nCrypTen ({WORLD_SIZE} parties, provider={provider}, "
              f"{cfg.encoder.precision_bits} fractional bits)")
        common.print_table(records)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=spec.REPS)
    parser.add_argument("--models", type=str, default=",".join(spec.MODELS))
    parser.add_argument("--provider", type=str, default="TFP", choices=["TFP", "TTP"],
                        help="TFP is insecure with 2 parties; TTP adds a real third party")
    parser.add_argument("--sizes", type=str, default=None,
                        help="override the spec's sizes, e.g. 8,32 -- for debugging only, "
                             "report.py will show the missing cells as absent")
    args = parser.parse_args()

    models = args.models.split(",")
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None

    import crypten.mpc as mpc_module
    from crypten.config import cfg

    if args.provider == "TTP":
        # CrypTen launches the TTPServer as a plain multiprocessing.Process and
        # gives it no way to receive configuration; its own code assumes the
        # fork start method so the child inherits cfg. macOS defaults to spawn,
        # where the server would come up thinking no TTP was required.
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)

    # Must be set before run_multiprocess: it decides whether a TTPServer
    # process is spawned alongside the two online parties.
    cfg.mpc.provider = args.provider

    # A stale single-file result from an older run would be read alongside the
    # per-party files and duplicate every row.
    legacy = os.path.join(common.RESULTS_DIR, f"{_framework_id(args.provider)}.jsonl")
    if os.path.exists(legacy):
        os.remove(legacy)

    runner = mpc_module.run_multiprocess(world_size=WORLD_SIZE)(_run)
    paths = runner(models, args.reps, args.provider, sizes)
    if not paths:
        raise SystemExit("crypten run_multiprocess returned no results")

    print("\nwrote " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
