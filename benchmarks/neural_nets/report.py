"""Builds the side-by-side comparison from whatever runners have written results.

    python report.py                # markdown to stdout
    python report.py --out REPORT.md

Reads results/*.jsonl, verifies every file was produced against the same spec,
and prints one table per metric with the frameworks as columns.

The accuracy columns are computed here rather than taken from the runners.
Every runner emits its trained network's predictions at `spec.witness_indices`,
and this module compares them against `ref.py`'s float64 run of the identical
training -- same architecture, same data, same initial weights, same update
rule. Doing it in one place is what makes the error column comparable: three
runners computing their own error against three copies of a reference would
also be measuring how well those copies agree.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import loc as loc_module
import ref
import spec

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Column order in every table. Sequre first, since it is the subject.
FRAMEWORK_ORDER = ["sequre-64", "sequre-128", "sequre-192",
                   "crypten", "crypten-ttp", "mpyc"]

# Share width is in the label because it is not a free parameter: it also fixes
# MPC_NBIT_V, Sequre's statistical security parameter (28 bits at 128, 64 at
# 192). CrypTen's ring is fixed at 64 bits and cannot be varied at all.
FRAMEWORK_LABELS = {
    # 64 is the width-matched control against CrypTen: same 16 fractional bits,
    # and only 10 bits of statistical security. Not a deployable configuration.
    "sequre-64": "Sequre (64b)",
    "sequre-128": "Sequre (128b)",
    "sequre-192": "Sequre (192b)",
    "crypten": "CrypTen (64b, TFP)",
    "crypten-ttp": "CrypTen (64b, TTP)",
    "mpyc": "MPyC (128b)",
}

# The Sequre build whose speedups are tabulated against the other frameworks.
PRIMARY_SEQURE = "sequre-192"

# Sequre forks one process per party; party 1 is an online compute party.
# CrypTen's party 0 doubles as the trusted first party and does strictly more
# work, so party 1 is the honest online figure there too. MPyC has no dealer.
REPORTING_PARTY = {"sequre": 1, "crypten": 1, "mpyc": 0}


class SpecMismatch(Exception):
    pass


def _check_meta(path: str, meta: dict) -> None:
    reference = spec.header()
    if meta.get("spec_version") != reference["spec_version"]:
        raise SpecMismatch(
            f"{path}: spec_version {meta.get('spec_version')!r} != {reference['spec_version']!r}")
    for key in ("epochs", "reps", "witness_points"):
        if meta.get(key) != reference[key]:
            raise SpecMismatch(f"{path}: {key} {meta.get(key)!r} != {reference[key]!r}")
    if meta.get("momentum") is not None and not math.isclose(
            float(meta["momentum"]), reference["momentum"], rel_tol=1e-9):
        raise SpecMismatch(f"{path}: momentum {meta['momentum']} != {reference['momentum']}")
    # The Codon runner writes its sizes as a flat {model: [...]} map rather
    # than the full model description, so only the sizes are cross-checked.
    sizes = meta.get("sizes")
    if sizes is not None:
        for model, expected in ((m, s["sizes"]) for m, s in reference["models"].items()):
            if model in sizes and sizes[model] != expected:
                raise SpecMismatch(f"{path}: sizes for {model} are {sizes[model]} != {expected}")


def load() -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    metas: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.jsonl"))):
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if not lines or "_meta" not in lines[0]:
            raise SpecMismatch(f"{path}: missing _meta header line")
        meta = lines[0]["_meta"]
        _check_meta(path, meta)
        metas[os.path.basename(path)] = meta
        for row in lines[1:]:
            family = row["framework"].split("-")[0]
            if row["party"] != REPORTING_PARTY.get(family, 0):
                continue
            rows.append(row)
    return rows, metas


# ---------------------------------------------------------------------------
# Accuracy, computed once here for every framework.
# ---------------------------------------------------------------------------

_REFERENCE_CACHE: dict[tuple[str, int], tuple[np.ndarray, float]] = {}


def _reference(model: str, n: int) -> tuple[np.ndarray, float]:
    """(predictions at the witness indices, final loss) from ref.py."""
    key = (model, n)
    if key not in _REFERENCE_CACHE:
        predictions, loss, _ = ref.train(model, n)
        witness = predictions[spec.witness_indices(n)].reshape(-1)
        _REFERENCE_CACHE[key] = (witness, loss)
    return _REFERENCE_CACHE[key]


def annotate(rows: list[dict]) -> None:
    """Fills in every row's error columns from its witness."""
    for row in rows:
        row["ref_final_loss"] = None
        if row.get("skipped") or not row.get("witness"):
            row["max_abs_pred_err"] = None
            row["mean_abs_pred_err"] = None
            row["loss_err"] = None
            continue

        expected, ref_loss = _reference(row["model"], row["n"])
        got = np.asarray(row["witness"], dtype=np.float64)
        if got.shape != expected.shape:
            raise SpecMismatch(
                f"{row['framework']} {row['model']} n={row['n']}: witness has "
                f"{got.shape[0]} values, the reference has {expected.shape[0]}")
        diffs = np.abs(got - expected)
        row["max_abs_pred_err"] = float(diffs.max())
        row["mean_abs_pred_err"] = float(diffs.mean())
        row["ref_final_loss"] = ref_loss
        row["loss_err"] = (None if row.get("final_loss") is None
                           else abs(float(row["final_loss"]) - ref_loss))


# ---------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------

def _index(rows: list[dict], mode: str = "end_to_end") -> dict[tuple[str, str, int], dict]:
    """Rows for one measurement mode.

    Sequre emits every cell twice -- `end_to_end` with the dealer inline and
    `online` with it detached. CrypTen and MPyC have only `end_to_end`, so they
    fall back to it and are never silently dropped from an online table.
    """
    out: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        key = (r["framework"], r["model"], r["n"])
        if r.get("mode", "end_to_end") == mode:
            out[key] = r
        elif key not in out and r.get("mode", "end_to_end") == "end_to_end":
            out.setdefault(key, r)
    return out


def _fmt(value, kind: str) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and math.isnan(value):
        return "--"
    if kind == "time":
        return f"{value:.4f}" if value < 10 else f"{value:,.1f}"
    if kind == "bytes":
        return f"{value / (1024 * 1024):,.1f}"
    if kind == "err":
        return f"{value:.1e}"
    if kind == "speedup":
        # Two digits below 1x, so a 60x deficit does not round to "0.0x".
        return f"{value:.1f}x" if value >= 1.0 else f"{value:.3g}x"
    if kind == "raw":
        return f"{value:,.0f}"
    if kind == "loss":
        return f"{value:.6f}"
    return str(value)


def _cells(rows_index, present, model: str, n: int, field: str, kind: str) -> list[str]:
    cells = []
    for framework in present:
        row = rows_index.get((framework, model, n))
        if row is None:
            cells.append("--")
        elif row.get("skipped"):
            cells.append("_skipped_")
        else:
            cells.append(_fmt(row.get(field), kind))
    return cells


def _table(rows_index, present, title: str, field: str, kind: str, unit: str) -> list[str]:
    out = [f"### {title}", "", f"_{unit}_", ""]
    out.append("| model | n | " + " | ".join(FRAMEWORK_LABELS[f] for f in present) + " |")
    out.append("|" + "---|" * (2 + len(present)))
    for model, model_spec in spec.MODELS.items():
        for n in model_spec["sizes"]:
            cells = _cells(rows_index, present, model, n, field, kind)
            out.append(f"| {model} | {n} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _loc_table(present) -> list[str]:
    counts = loc_module.counts()
    out = ["### Lines of model code", "",
           "_non-blank, non-comment lines a user must write to define and train "
           "the network in that framework; library code is not counted. See "
           "`loc.py` for exactly what is inside the markers._", ""]
    out.append("| model | " + " | ".join(FRAMEWORK_LABELS[f] for f in present) + " |")
    out.append("|" + "---|" * (1 + len(present)))
    for model in spec.MODELS:
        cells = [str(counts.get(f, {}).get(model, "--")) for f in present]
        out.append(f"| {model} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _speedup_table(rows_index, present, title_suffix: str = "", note: str = "") -> list[str]:
    baselines = [f for f in ("crypten", "crypten-ttp", "mpyc") if f in present]
    if PRIMARY_SEQURE not in present or not baselines:
        return []

    out = [f"### {FRAMEWORK_LABELS[PRIMARY_SEQURE]} speedup{title_suffix}", "",
           f"_median wall-clock of the baseline divided by {FRAMEWORK_LABELS[PRIMARY_SEQURE]}'s; "
           "&gt;1 means Sequre is faster. Note the differing share widths in the column labels._"
           + (f" {note}" if note else ""), ""]
    out.append("| model | n | " + " | ".join(f"vs {FRAMEWORK_LABELS[b]}" for b in baselines) + " |")
    out.append("|" + "---|" * (2 + len(baselines)))
    for model, model_spec in spec.MODELS.items():
        for n in model_spec["sizes"]:
            sequre = rows_index.get((PRIMARY_SEQURE, model, n))
            cells = []
            for baseline in baselines:
                other = rows_index.get((baseline, model, n))
                if (sequre is None or other is None
                        or other.get("skipped") or sequre.get("skipped")
                        or sequre["median_s"] <= 0):
                    cells.append("--")
                else:
                    cells.append(_fmt(other["median_s"] / sequre["median_s"], "speedup"))
            out.append(f"| {model} | {n} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _reference_table() -> list[str]:
    out = ["### float64 reference", "",
           "_`ref.py` trains the same networks in the clear, from the same "
           "initial weights. Every accuracy column above is measured against "
           "this._", "",
           "| model | n | final loss |", "|---|---|---|"]
    for model, model_spec in spec.MODELS.items():
        for n in model_spec["sizes"]:
            out.append(f"| {model} | {n} | {_reference(model, n)[1]:.6f} |")
    out.append("")
    return out


def build(rows: list[dict], metas: dict[str, dict]) -> str:
    annotate(rows)
    rows_index = _index(rows, "end_to_end")
    online_index = _index(rows, "online")
    seen = {r["framework"] for r in rows}
    present = [f for f in FRAMEWORK_ORDER if f in seen]
    if not present:
        raise SpecMismatch("no result rows found -- run the benchmarks first")

    out = [
        "# Neural networks: Sequre vs CrypTen vs MPyC",
        "",
        f"Frameworks present: {', '.join(FRAMEWORK_LABELS[f] for f in present)}.",
        "",
        f"Two sequential feed-forward networks, trained for {spec.EPOCHS} epochs of "
        f"batch gradient descent with Nesterov momentum, from initial weights shared "
        f"across every framework; {spec.REPS} timed repetitions per cell, median "
        f"reported. `n` is the number of training rows -- for SIREN that is the pixel "
        f"count of a square image, for the MLP the batch of a full-batch step.",
        "",
        "Missing frameworks show `--`. `_skipped_` means the runner declined that "
        "cell; its reason is in the row's `note` in `results/`.",
        "",
    ]

    out += _loc_table(present)
    out += _table(rows_index, present, "Training time", "epoch_s", "time",
                  "seconds per epoch, median of timed repetitions")
    out += _table(rows_index, present, "Communication", "bytes_sent", "bytes",
                  "MiB sent per training run by the reporting party")
    out += _table(rows_index, present, "Accuracy (max absolute prediction error)",
                  "max_abs_pred_err", "err",
                  "trained network's predictions vs the float64 reference's")
    out += _table(rows_index, present, "Accuracy (final training loss)", "final_loss", "loss",
                  "compare against the float64 reference below")
    out += _speedup_table(rows_index, present)

    # Online cost: Sequre with the dealer detached, so offline randomness
    # generation is off the critical path. CrypTen and MPyC have no separable
    # offline phase and appear here with their end-to-end numbers, which is
    # the comparison's one genuine asymmetry -- stated, not hidden.
    if any(r.get("mode") == "online" for r in rows):
        out += ["## Online cost (dealer detached)", "",
                "_Sequre rows have the dealer detached via `mpc.detach_dealer()`, so only "
                "the compute parties' work is timed. CrypTen and MPyC have no separable "
                "offline phase and repeat their end-to-end numbers here. Large cells are "
                "skipped: a detached dealer buffers every byte it would have sent until "
                "the block exits, and a full training run exceeds what the transport "
                "sustains._", ""]
        out += _table(online_index, present, "Training time, dealer detached", "epoch_s",
                      "time", "seconds per epoch, median of timed repetitions")
        out += _speedup_table(
            online_index, present, " (dealer detached)",
            "Sequre is detached; the baselines are not.")

    out += _reference_table()

    out += ["### Run metadata", ""]
    for name, meta in sorted(metas.items()):
        detail = ", ".join(f"{k}={v}" for k, v in meta.items()
                           if k not in ("sizes", "reps", "models", "spec_version",
                                        "epochs", "momentum", "witness_points"))
        out.append(f"- `{name}`: {detail}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None,
                        help="write markdown here instead of stdout")
    args = parser.parse_args()

    rows, metas = load()
    report = build(rows, metas)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
