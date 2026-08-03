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
    model: str
    n: int                         # training rows
    epochs: int
    reps: int
    median_s: float                # seconds for `epochs` epochs
    min_s: float
    mean_s: float
    epoch_s: float                 # median_s / epochs -- the headline figure
    bytes_sent: float | None       # bytes this party sent, per timed run
    rounds: float | None           # communication rounds, if the framework counts them
    final_loss: float | None       # after training, on the training set
    # Predictions at spec.witness_indices(n). report.py compares these against
    # ref.py's float64 run of the identical training, identically for every
    # framework -- the error columns in the report come from here and not from
    # any runner's own arithmetic.
    witness: list[float] | None
    # Convenience only: the two Python runners can import ref.py, so they fill
    # these for their own stdout echo. The Codon runner cannot and leaves them
    # null. report.py ignores them and recomputes from `witness`.
    ref_final_loss: float | None
    max_abs_pred_err: float | None
    mean_abs_pred_err: float | None
    loc: int | None                # lines of model code, see loc.py
    party: int
    parties: int
    mode: str                      # "end_to_end" or "online"
    skipped: bool = False
    note: str = ""

    def as_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


def skipped(framework: str, model: str, n: int, party: int, note: str) -> Record:
    return Record(
        framework=framework, model=model, n=n, epochs=spec.EPOCHS, reps=0,
        median_s=float("nan"), min_s=float("nan"), mean_s=float("nan"),
        epoch_s=float("nan"), bytes_sent=None, rounds=None,
        final_loss=None, witness=None, ref_final_loss=None,
        max_abs_pred_err=None, mean_abs_pred_err=None, loc=None, party=party,
        parties=spec.PARTIES[framework], mode="end_to_end", skipped=True, note=note)


def pred_errors(got: Iterable[float], expected: Iterable[float]) -> tuple[float, float]:
    """(max abs, mean abs) between secure and float64 predictions."""
    diffs = [abs(float(g) - float(e)) for g, e in zip(got, expected)]
    if not diffs:
        return float("nan"), float("nan")
    return max(diffs), sum(diffs) / len(diffs)


def time_reps(
    body: Callable[[], object],
    reps: int = spec.REPS,
    warmup: int = spec.WARMUP_REPS,
) -> tuple[list[float], object]:
    """Runs `body` warmup+reps times, returning the timed durations and the last result.

    `body` must train a *fresh* model: training is stateful, so reusing one
    across repetitions would time epoch 4 against epoch 1 and would leave the
    accuracy witness depending on the repetition count.
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
    model: str,
    n: int,
    times: list[float],
    party: int,
    *,
    bytes_sent: float | None = None,
    rounds: float | None = None,
    final_loss: float | None = None,
    ref_final_loss: float | None = None,
    got: Iterable[float] | None = None,
    expected: Iterable[float] | None = None,
    witness: Iterable[float] | None = None,
    loc: int | None = None,
    mode: str = "end_to_end",
    note: str = "",
) -> Record:
    ordered = sorted(times)
    median = statistics.median(ordered)
    max_err, mean_err = (None, None)
    if got is not None and expected is not None:
        max_err, mean_err = pred_errors(got, expected)
    return Record(
        framework=framework,
        model=model,
        n=n,
        epochs=spec.EPOCHS,
        reps=len(times),
        median_s=median,
        min_s=ordered[0],
        mean_s=statistics.fmean(ordered),
        epoch_s=median / spec.EPOCHS,
        bytes_sent=bytes_sent,
        rounds=rounds,
        final_loss=final_loss,
        witness=None if witness is None else [float(v) for v in witness],
        ref_final_loss=ref_final_loss,
        max_abs_pred_err=max_err,
        mean_abs_pred_err=mean_err,
        loc=loc,
        party=party,
        parties=spec.PARTIES[framework],
        mode=mode,
        note=note,
    )


def write_jsonl(framework: str, records: list[Record], meta: dict,
                party: int | None = None) -> str:
    """Writes one result file. `party` gives it a per-party name.

    A runner whose parties are separate processes has to write from inside each
    of them rather than collecting the records in a parent: the witness vectors
    are large enough that returning them through a multiprocessing queue
    deadlocks CrypTen's `run_multiprocess`, which joins its children before it
    drains the queue. report.py reads every `*.jsonl` in the directory and
    filters by party, so one file or several makes no difference to it.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = framework if party is None else f"{framework}_CP{party}"
    path = os.path.join(RESULTS_DIR, f"{name}.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"_meta": {**spec.header(), **meta}}) + "\n")
        for record in records:
            f.write(record.as_json() + "\n")
    return path


def print_table(records: list[Record]) -> None:
    """Compact stdout echo so a single-framework run is readable on its own."""
    head = (f"{'model':<7}{'n':>7}{'epoch_s':>11}{'bytes':>14}"
            f"{'loss':>11}{'ref_loss':>11}{'pred_err':>11}{'LOC':>5}")
    print(head)
    print("-" * len(head))
    for r in records:
        if r.skipped:
            print(f"{r.model:<7}{r.n:>7}{'skipped':>11}  {r.note}")
            continue
        b = "-" if r.bytes_sent is None else f"{r.bytes_sent:,.0f}"
        li = "-" if r.final_loss is None else f"{r.final_loss:.4e}"
        rl = "-" if r.ref_final_loss is None else f"{r.ref_final_loss:.4e}"
        pe = ("-" if r.max_abs_pred_err is None or math.isnan(r.max_abs_pred_err)
              else f"{r.max_abs_pred_err:.2e}")
        lo = "-" if r.loc is None else str(r.loc)
        print(f"{r.model:<7}{r.n:>7}{r.epoch_s:>11.6f}{b:>14}"
              f"{li:>11}{rl:>11}{pe:>11}{lo:>5}")
