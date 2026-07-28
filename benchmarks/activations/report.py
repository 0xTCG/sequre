"""Builds the side-by-side comparison from whatever runners have written results.

    python report.py                # markdown to stdout
    python report.py --out REPORT.md

Reads results/*.jsonl, verifies every file was produced against the same spec,
and prints one table per metric with the frameworks as columns.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spec

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Column order in every table. Sequre first, since it is the subject.
FRAMEWORK_ORDER = [
    "sequre-decor-64", "sequre-decor-128", "sequre-decor-192",
    "sequre-fourier-64", "sequre-fourier-128", "sequre-fourier-192",
    "crypten", "crypten-ttp", "mpyc",
]

# Share width is in the label because it is not a free parameter: it also fixes
# MPC_NBIT_V, Sequre's statistical security parameter (28 bits at 128, 64 at
# 192). CrypTen's ring is fixed at 64 bits and cannot be varied at all.
FRAMEWORK_LABELS = {
    "sequre-decor-64": "Sequre+Decor (64b)",
    "sequre-decor-128": "Sequre+Decor (128b)",
    "sequre-decor-192": "Sequre+Decor (192b)",
    "sequre-fourier-64": "Sequre/Fourier (64b)",
    "sequre-fourier-128": "Sequre/Fourier (128b)",
    "sequre-fourier-192": "Sequre/Fourier (192b)",
    "crypten": "CrypTen (64b, TFP)",
    "crypten-ttp": "CrypTen (64b, TTP)",
    "mpyc": "MPyC (128b)",
}

# The Sequre build whose speedups are tabulated against the other frameworks.
PRIMARY_SEQURE = "sequre-decor-192"

# Sequre forks one process per party; party 1 is an online compute party.
# CrypTen's party 0 doubles as the trusted first party and does strictly more
# work, so party 1 is the honest online figure there too. MPyC has no dealer.
REPORTING_PARTY = {"sequre": 1, "crypten": 1, "mpyc": 0}
# crypten-ttp splits on "-" to family "crypten", so it inherits party 1.


class SpecMismatch(Exception):
    pass


def _check_meta(path: str, meta: dict) -> None:
    reference = spec.header()
    if meta.get("spec_version") != reference["spec_version"]:
        raise SpecMismatch(
            f"{path}: spec_version {meta.get('spec_version')!r} != {reference['spec_version']!r}")
    for key in ("sizes", "reps"):
        if meta.get(key) != reference[key]:
            raise SpecMismatch(f"{path}: {key} {meta.get(key)!r} != {reference[key]!r}")
    for function, interval in reference["intervals"].items():
        got = meta.get("intervals", {}).get(function)
        if got is None:
            continue
        # Compared with a tolerance, not for equality: the Codon runner writes
        # these through an f-string that rounds to 6 significant digits, so an
        # interval of -pi arrives as -3.14159. The guard is there to catch a
        # runner benchmarking a different interval, not a formatting delta.
        if len(got) != len(interval) or not all(
            math.isclose(float(g), float(w), rel_tol=1e-5, abs_tol=1e-9)
            for g, w in zip(got, interval)
        ):
            raise SpecMismatch(f"{path}: interval for {function} is {got} != {interval}")


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


def _index(rows: list[dict], mode: str = "end_to_end") -> dict[tuple[str, str, int], dict]:
    """Rows for one measurement mode.

    Sequre emits every cell twice -- `end_to_end` with the dealer inline and
    `online` with it detached. CrypTen and MPyC have only `end_to_end`, so they
    fall back to it and are never silently dropped from an online table.
    """
    out: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        key = (r["framework"], r["function"], r["n"])
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
        return f"{value * 1000:.3f}" if value < 1 else f"{value * 1000:,.0f}"
    if kind == "bytes":
        return f"{value / 1024:,.1f}"
    if kind == "err":
        return f"{value:.1e}"
    if kind == "speedup":
        return f"{value:.1f}x"
    if kind == "raw":
        return f"{value:,.0f}"
    return str(value)


def _table(rows_index, present, title: str, field: str, kind: str, unit: str) -> list[str]:
    out = [f"### {title}", "", f"_{unit}_", ""]
    header = "| function | n | " + " | ".join(FRAMEWORK_LABELS[f] for f in present) + " |"
    out.append(header)
    out.append("|" + "---|" * (2 + len(present)))
    for function in spec.FUNCTIONS:
        for n in spec.SIZES:
            cells = []
            for framework in present:
                row = rows_index.get((framework, function, n))
                if row is None:
                    cells.append("--")
                elif row.get("skipped"):
                    cells.append("_skipped_")
                else:
                    cells.append(_fmt(row.get(field), kind))
            out.append(f"| {function} | {n} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _speedup_table(rows_index, present, title_suffix: str = "", note: str = "") -> list[str]:
    baselines = [f for f in ("crypten", "crypten-ttp", "mpyc") if f in present]
    if PRIMARY_SEQURE not in present or not baselines:
        return []

    out = [f"### {FRAMEWORK_LABELS[PRIMARY_SEQURE]} speedup{title_suffix}", "",
           f"_median wall-clock of the baseline divided by {FRAMEWORK_LABELS[PRIMARY_SEQURE]}'s; "
           "&gt;1 means Decor is faster. Note the differing share widths in the column labels._"
           + (f" {note}" if note else ""), ""]
    header = "| function | n | " + " | ".join(f"vs {FRAMEWORK_LABELS[b]}" for b in baselines) + " |"
    out.append(header)
    out.append("|" + "---|" * (2 + len(baselines)))
    for function in spec.FUNCTIONS:
        for n in spec.SIZES:
            decor = rows_index.get((PRIMARY_SEQURE, function, n))
            cells = []
            for baseline in baselines:
                other = rows_index.get((baseline, function, n))
                if decor is None or other is None or other.get("skipped") or decor.get("skipped"):
                    cells.append("--")
                elif decor["median_s"] <= 0:
                    cells.append("--")
                else:
                    cells.append(_fmt(other["median_s"] / decor["median_s"], "speedup"))
            out.append(f"| {function} | {n} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def build(rows: list[dict], metas: dict[str, dict]) -> str:
    rows_index = _index(rows, "end_to_end")
    online_index = _index(rows, "online")
    seen = {r["framework"] for r in rows}
    present = [f for f in FRAMEWORK_ORDER if f in seen]
    if not present:
        raise SpecMismatch("no result rows found -- run the benchmarks first")

    out = [
        "# Activation functions: Sequre+Decor vs CrypTen vs MPyC",
        "",
        f"Frameworks present: {', '.join(FRAMEWORK_LABELS[f] for f in present)}.",
        f"Inputs are `linspace(a, b, n)` over each function's interval; "
        f"{spec.REPS} timed repetitions per cell, median reported.",
        "",
        "Missing frameworks show `--`. `_skipped_` means the runner was told to "
        "skip that size (see its `--max-n`).",
        "",
    ]

    out += _table(rows_index, present, "Latency", "median_s", "time", "milliseconds, median of timed repetitions")
    out += _table(rows_index, present, "Throughput", "throughput_eps", "raw", "elements per second")
    out += _table(rows_index, present, "Communication", "bytes_sent", "bytes", "KiB sent per call by the reporting party")
    out += _table(rows_index, present, "Accuracy (max absolute error)", "max_abs_err", "err", "vs float64, over the whole interval")
    out += _table(rows_index, present, "Accuracy (max absolute error, interior 80%)", "max_abs_err_interior", "err",
                  "vs float64, endpoints trimmed -- separates a poor fit from a boundary artifact")
    out += _speedup_table(rows_index, present)

    # Online cost: Sequre with the dealer detached, so offline randomness
    # generation is off the critical path. CrypTen and MPyC have no separable
    # offline phase and appear here with their end-to-end numbers, which is
    # the comparison's one genuine asymmetry -- stated, not hidden.
    has_online = any(r.get("mode") == "online" for r in rows)
    if has_online:
        out += ["## Online cost (dealer detached)", "",
                "_Sequre rows have the dealer detached via `mpc.detach_dealer()`, so only "
                "the compute parties' work is timed. CrypTen and MPyC have no separable "
                "offline phase and repeat their end-to-end numbers here._", ""]
        out += _table(online_index, present, "Latency, dealer detached", "median_s", "time",
                      "milliseconds, median of timed repetitions")
        out += _table(online_index, present, "Communication, dealer detached", "bytes_sent",
                      "bytes", "KiB sent per call by the reporting party")
        out += _speedup_table(
            online_index, present, " (dealer detached)",
            "Sequre is detached; the baselines are not.")

    out += ["### Run metadata", ""]
    for name, meta in sorted(metas.items()):
        detail = ", ".join(f"{k}={v}" for k, v in meta.items()
                           if k not in ("sizes", "reps", "intervals", "spec_version"))
        out.append(f"- `{name}`: {detail}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="write markdown here instead of stdout")
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
