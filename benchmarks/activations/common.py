"""Result records and JSONL I/O shared by the Python runners."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import statistics
import time
from typing import Callable, Iterable

import spec


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@dataclasses.dataclass
class Record:
    framework: str
    function: str
    n: int
    reps: int
    median_s: float
    min_s: float
    mean_s: float
    throughput_eps: float          # elements per second, from median_s
    bytes_sent: float | None       # online bytes sent by this party
    rounds: float | None           # online communication rounds, if the framework counts them
    max_abs_err: float | None      # vs float64 ground truth
    mean_abs_err: float | None
    max_rel_err: float | None      # relative error, over entries with |truth| > REL_ERR_FLOOR
    max_abs_err_interior: float | None  # max abs error over the middle 80% of the interval
    party: int
    parties: int
    mode: str                      # "end_to_end": correlated randomness generated inside the timed call
    skipped: bool = False
    note: str = ""

    def as_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


def skipped(framework: str, function: str, n: int, party: int, note: str) -> Record:
    return Record(
        framework=framework, function=function, n=n, reps=0,
        median_s=float("nan"), min_s=float("nan"), mean_s=float("nan"),
        throughput_eps=float("nan"), bytes_sent=None, rounds=None,
        max_abs_err=None, mean_abs_err=None, max_rel_err=None,
        max_abs_err_interior=None, party=party,
        parties=spec.PARTIES[framework], mode="end_to_end", skipped=True, note=note)


# Relative error is meaningless where the true value is ~0 (relu at x<0, tanh
# near the origin), so those entries are excluded from the relative figure.
# Absolute error still covers them.
REL_ERR_FLOOR = 1e-6

# A Fourier series over [a, b] treats the function as periodic, so a
# non-periodic target has a jump at the interval boundary and the classic Gibbs
# overshoot next to it -- an O(1) max error that says nothing about the fit in
# the interior. Error is therefore reported twice for every framework: over the
# whole interval, and over the middle 80% of it. Must match INTERIOR_TRIM in
# sequre/bench_activations.codon.
INTERIOR_TRIM = 0.1


def errors(got: Iterable[float], expected: Iterable[float]) -> tuple[float, float, float, float]:
    """(max abs, mean abs, max relative, max abs over interior) vs the float64 reference."""
    pairs = [(float(g), float(e)) for g, e in zip(got, expected)]
    if not pairs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs = [abs(g - e) for g, e in pairs]
    rel = [abs(g - e) / abs(e) for g, e in pairs if abs(e) > REL_ERR_FLOOR]
    # Inputs are linspace, so trimming indices trims the interval.
    lo = int(len(diffs) * INTERIOR_TRIM)
    interior = diffs[lo:len(diffs) - lo] or diffs
    return (max(diffs), sum(diffs) / len(diffs),
            (max(rel) if rel else float("nan")), max(interior))


def time_reps(
    body: Callable[[], object],
    reps: int = spec.REPS,
    warmup: int = spec.WARMUP_REPS,
) -> tuple[list[float], object]:
    """Runs `body` warmup+reps times, returning the timed durations and the last result.

    The result of the final timed call is handed back so the caller can check it
    against ground truth -- a speedup that breaks the math has to be visible in
    the same row as the speedup.
    """
    for _ in range(warmup):
        body()
    times: list[float] = []
    out: object = None
    for _ in range(reps):
        start = time.perf_counter()
        out = body()
        times.append(time.perf_counter() - start)
    return times, out


def make_record(
    framework: str,
    function: str,
    n: int,
    times: list[float],
    party: int,
    *,
    bytes_sent: float | None = None,
    rounds: float | None = None,
    got: Iterable[float] | None = None,
    expected: Iterable[float] | None = None,
    mode: str = "end_to_end",
    note: str = "",
) -> Record:
    ordered = sorted(times)
    median = statistics.median(ordered)
    max_err, mean_err, rel_err, interior_err = (None, None, None, None)
    if got is not None and expected is not None:
        max_err, mean_err, rel_err, interior_err = errors(got, expected)
    return Record(
        framework=framework,
        function=function,
        n=n,
        reps=len(times),
        median_s=median,
        min_s=ordered[0],
        mean_s=statistics.fmean(ordered),
        throughput_eps=(n / median) if median > 0 else float("inf"),
        bytes_sent=bytes_sent,
        rounds=rounds,
        max_abs_err=max_err,
        mean_abs_err=mean_err,
        max_rel_err=rel_err,
        max_abs_err_interior=interior_err,
        party=party,
        parties=spec.PARTIES[framework],
        mode=mode,
        note=note,
    )


def write_jsonl(framework: str, records: list[Record], meta: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{framework}.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"_meta": {**spec.header(), **meta}}) + "\n")
        for record in records:
            f.write(record.as_json() + "\n")
    return path


def print_table(records: list[Record]) -> None:
    """Compact stdout echo so a single-framework run is readable on its own."""
    head = (f"{'function':<9}{'n':>7}{'median_s':>12}{'elem/s':>12}"
            f"{'bytes':>12}{'abs_err':>11}{'rel_err':>11}")
    print(head)
    print("-" * len(head))
    for r in records:
        if r.skipped:
            print(f"{r.function:<9}{r.n:>7}{'skipped':>12}{'':>12}{'':>12}"
                  f"{'':>11}{'':>11}  {r.note}")
            continue
        b = "-" if r.bytes_sent is None else f"{r.bytes_sent:,.0f}"
        a = "-" if r.max_abs_err is None else f"{r.max_abs_err:.2e}"
        e = "-" if r.max_rel_err is None or math.isnan(r.max_rel_err) else f"{r.max_rel_err:.2e}"
        print(f"{r.function:<9}{r.n:>7}{r.median_s:>12.6f}{r.throughput_eps:>12,.0f}"
              f"{b:>12}{a:>11}{e:>11}")
