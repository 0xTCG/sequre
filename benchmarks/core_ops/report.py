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
    "sequre-64", "sequre-128", "sequre-192",
    "crypten", "crypten-ttp", "mpyc",
]

# Share width is in the label because it is not a free parameter: it also fixes
# MPC_NBIT_V, Sequre's statistical security parameter (28 bits at 128, 64 at
# 192). The other three cannot vary theirs at all.
FRAMEWORK_LABELS = {
    "sequre-64": "Sequre (64b)",
    "sequre-128": "Sequre (128b)",
    "sequre-192": "Sequre (192b)",
    "crypten": "CrypTen (TFP)",
    "crypten-ttp": "CrypTen (TTP)",
    "mpyc": "MPyC",
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
    for key in ("sizes", "reps"):
        if meta.get(key) != reference[key]:
            raise SpecMismatch(f"{path}: {key} {meta.get(key)!r} != {reference[key]!r}")

    scalar = meta.get("public_scalar")
    if scalar is not None and not math.isclose(float(scalar), spec.PUBLIC_SCALAR, rel_tol=1e-9):
        raise SpecMismatch(f"{path}: public_scalar {scalar} != {spec.PUBLIC_SCALAR}")

    for op, interval in reference["intervals"].items():
        got = meta.get("intervals", {}).get(op)
        if got is None:
            continue
        # Compared with a tolerance, not for equality: the non-Python runners
        # write these through formatted output that rounds. The guard is there
        # to catch a runner benchmarking a different interval, not a
        # formatting delta.
        if len(got) != len(interval) or not all(
            math.isclose(float(g), float(w), rel_tol=1e-5, abs_tol=1e-9)
            for g, w in zip(got, interval)
        ):
            raise SpecMismatch(f"{path}: interval for {op} is {got} != {interval}")


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
    `online` with it detached. The others have only `end_to_end`, so they fall
    back to it and are never silently dropped from an online table.
    """
    out: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        key = (r["framework"], r["op"], r["n"])
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
    out.append("| op | n | " + " | ".join(FRAMEWORK_LABELS[f] for f in present) + " |")
    out.append("|" + "---|" * (2 + len(present)))
    for op in spec.OPS:
        for n in spec.SIZES:
            cells = []
            for framework in present:
                row = rows_index.get((framework, op, n))
                if row is None:
                    cells.append("--")
                elif row.get("skipped"):
                    cells.append("_skipped_")
                else:
                    cells.append(_fmt(row.get(field), kind))
            out.append(f"| {op} | {n} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _speedup_table(rows_index, present, title_suffix: str = "", note: str = "") -> list[str]:
    baselines = [f for f in ("crypten", "crypten-ttp", "mpyc") if f in present]
    if PRIMARY_SEQURE not in present or not baselines:
        return []

    out = [f"### {FRAMEWORK_LABELS[PRIMARY_SEQURE]} speedup{title_suffix}", "",
           f"_median wall-clock of the baseline divided by {FRAMEWORK_LABELS[PRIMARY_SEQURE]}'s; "
           "&gt;1 means Sequre is faster. Note the differing party counts and share widths._"
           + (f" {note}" if note else ""), ""]
    out.append("| op | n | " + " | ".join(f"vs {FRAMEWORK_LABELS[b]}" for b in baselines) + " |")
    out.append("|" + "---|" * (2 + len(baselines)))
    for op in spec.OPS:
        for n in spec.SIZES:
            sequre = rows_index.get((PRIMARY_SEQURE, op, n))
            cells = []
            for baseline in baselines:
                other = rows_index.get((baseline, op, n))
                # A median of None means the cell finished inside one clock
                # tick and the runner declined to report a time; there is no
                # ratio to take.
                base = sequre.get("median_s") if sequre else None
                against = other.get("median_s") if other else None
                if (sequre is None or other is None
                        or other.get("skipped") or sequre.get("skipped")
                        or base is None or against is None or base <= 0):
                    cells.append("--")
                else:
                    cells.append(_fmt(against / base, "speedup"))
            out.append(f"| {op} | {n} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _skips(rows: list[dict], present: list[str]) -> list[str]:
    """Every skipped cell, with its reason.

    A framework that cannot express an operation is a result of this
    benchmark, not a gap in it, so the reasons are tabulated rather than left
    as `_skipped_` in a cell.
    """
    # Keyed by reason as well as by op: a single op can be skipped at one size
    # because it is unsupported and at another because the run failed, and
    # collapsing those into one row would hide the difference that matters.
    seen: dict[tuple[str, str, str], list[int]] = {}
    for r in rows:
        if r.get("skipped") and r["framework"] in present:
            key = (r["framework"], r["op"], r.get("note", ""))
            seen.setdefault(key, []).append(r["n"])
    if not seen:
        return []
    out = ["### Why cells are skipped", "",
           "_A cap named in the reason is a runner flag and can be raised; "
           "anything else is a property of the framework._", "",
           "| framework | op | n | reason |", "|---|---|---|---|"]
    for (framework, op, note), sizes in sorted(seen.items()):
        # A literal pipe would break the table.
        safe = note.replace("|", "\\|")
        out.append(f"| {FRAMEWORK_LABELS[framework]} | {op} | "
                   f"{', '.join(str(n) for n in sorted(sizes))} | {safe} |")
    out.append("")
    return out


def build(rows: list[dict], metas: dict[str, dict]) -> str:
    rows_index = _index(rows, "end_to_end")
    online_index = _index(rows, "online")
    seen = {r["framework"] for r in rows}
    present = [f for f in FRAMEWORK_ORDER if f in seen]
    if not present:
        raise SpecMismatch("no result rows found -- run the benchmarks first")

    parties = {}
    for r in rows:
        parties.setdefault(r["framework"], r.get("parties"))

    out = [
        "# Core operations: Sequre vs CrypTen vs MPyC",
        "",
        f"Frameworks present: "
        + ", ".join(f"{FRAMEWORK_LABELS[f]} ({parties.get(f)} parties)" for f in present)
        + ".",
        "",
        f"The operation set is the intersection of what all three implement in "
        f"their own library: no cell is a reimplementation. Inputs are "
        f"`linspace(a, b, n)` over each op's interval; {spec.REPS} timed "
        f"repetitions per cell, median reported.",
        "",
        "Missing frameworks show `--`. `_skipped_` cells are explained in the "
        "table at the end.",
        "",
    ]

    out += _table(rows_index, present, "Latency", "median_s", "time",
                  "milliseconds, median of timed repetitions")
    out += _table(rows_index, present, "Throughput", "throughput_eps", "raw",
                  "input elements per second")
    out += _table(rows_index, present, "Communication", "bytes_sent", "bytes",
                  "KiB sent per call by the reporting party")
    out += _table(rows_index, present, "Accuracy (max absolute error)", "max_abs_err", "err",
                  "vs float64 computed in the clear")
    out += _speedup_table(rows_index, present)

    # Online cost: Sequre with the dealer detached, so offline randomness
    # generation is off the critical path. The others have no separable offline
    # phase and appear here with their end-to-end numbers, which is the
    # comparison's one genuine asymmetry -- stated, not hidden.
    if any(r.get("mode") == "online" for r in rows):
        out += ["## Online cost (dealer detached)", "",
                "_Sequre rows have the dealer detached via `mpc.detach_dealer()`, so only "
                "the compute parties' work is timed. The other three have no separable "
                "offline phase and repeat their end-to-end numbers here._", ""]
        out += _table(online_index, present, "Latency, dealer detached", "median_s", "time",
                      "milliseconds, median of timed repetitions")
        out += _table(online_index, present, "Communication, dealer detached", "bytes_sent",
                      "bytes", "KiB sent per call by the reporting party")
        out += _speedup_table(online_index, present, " (dealer detached)",
                              "Sequre is detached; the baselines are not.")

    out += _skips(rows, present)

    out += ["### Run metadata", ""]
    for name, meta in sorted(metas.items()):
        detail = ", ".join(f"{k}={v}" for k, v in meta.items()
                           if k not in ("sizes", "reps", "intervals", "spec_version"))
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
