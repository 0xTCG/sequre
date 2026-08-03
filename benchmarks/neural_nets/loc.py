"""Lines of model code per framework, counted mechanically from the runners.

The claim a secure-ML framework makes is not only "this is fast" but "you can
write your network in it". That second claim is usually made in prose. Here it
is a number, measured the same way for every column.

**What is counted.** Each runner brackets the code that defines and trains one
network with

    # LOC-BEGIN <model>[,<model>...]
    ...
    # LOC-END <model>[,<model>...]

and this module counts the non-blank, non-comment lines between the markers.
A region may name several models, in which case its lines count towards each
of them -- that is how a training loop shared by two networks is charged to
both rather than arbitrarily to one.
The region covers exactly: constructing the network, choosing loss and
optimizer, running the training loop, and producing predictions from it -- the
code a user of that framework would have to write.

**What is not counted**, and is therefore outside the markers in every runner:
loading or generating data, injecting the shared initial weights, timing,
communication accounting, JSONL output, and the float64 reference. These are
benchmark scaffolding, not model code, and they are roughly the same size
everywhere; counting them would dilute the difference the number exists to show.

Framework-provided machinery is not counted either -- that is the whole point.
Sequre's `Dense` and CrypTen's `nn.Linear` are library code and cost their users
nothing, while MPyC has no neural-network layer at all, so its forward pass,
its backward pass and its optimizer are all inside its markers. The number
answers "how much code must *you* write", and a framework that ships the layer
is credited for shipping it.

**These are physical lines, so they are sensitive to formatting.** There is no
way around that -- a token count would reward dense one-liners nobody writes --
so instead every runner is formatted to the same convention: an ~88 column
limit, one logical argument group per continuation line, no statement
compression. A layer whose constructor takes six keyword arguments therefore
costs several lines, and that is the intended reading: those arguments are
things the user has to supply.

The counts are *not* a quality measure on their own. A framework can be terse
because it is expressive or because it is inflexible, and this file cannot tell
the two apart; read the LOC column next to the accuracy and timing columns, not
instead of them.
"""

from __future__ import annotations

import argparse
import json
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# framework id -> the runner file whose markers it is counted from. The Sequre
# width variants and the two CrypTen providers are the same source code run
# with different settings, so they share a count.
SOURCES: dict[str, str] = {
    "sequre-64": os.path.join(HERE, "sequre", "bench_nn.codon"),
    "sequre-128": os.path.join(HERE, "sequre", "bench_nn.codon"),
    "sequre-192": os.path.join(HERE, "sequre", "bench_nn.codon"),
    "crypten": os.path.join(HERE, "crypten", "bench_nn.py"),
    "crypten-ttp": os.path.join(HERE, "crypten", "bench_nn.py"),
    "mpyc": os.path.join(HERE, "mpyc", "bench_nn.py"),
}

BEGIN = re.compile(r"^\s*#\s*LOC-BEGIN\s+(\S+)\s*$")
END = re.compile(r"^\s*#\s*LOC-END\s+(\S+)\s*$")


def _models(marker: str) -> list[str]:
    return [m for m in marker.split(",") if m]


def count_file(path: str) -> dict[str, int]:
    """model -> non-blank, non-comment lines inside that model's markers."""
    counts: dict[str, int] = {}
    open_region: list[str] | None = None

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            begin = BEGIN.match(line)
            if begin:
                if open_region is not None:
                    raise ValueError(
                        f"{path}:{lineno}: LOC-BEGIN {begin.group(1)} inside "
                        f"an open {','.join(open_region)} region")
                open_region = _models(begin.group(1))
                for model in open_region:
                    counts.setdefault(model, 0)
                continue

            end = END.match(line)
            if end:
                if open_region != _models(end.group(1)):
                    raise ValueError(
                        f"{path}:{lineno}: LOC-END {end.group(1)} does not "
                        f"close {open_region}")
                open_region = None
                continue

            if open_region is None:
                continue
            stripped = line.strip()
            # Blank lines and whole-line comments are formatting, not code.
            # An inline trailing comment stays with its line, as it should.
            if not stripped or stripped.startswith("#"):
                continue
            for model in open_region:
                counts[model] += 1

    if open_region is not None:
        raise ValueError(f"{path}: unterminated LOC-BEGIN {','.join(open_region)}")

    return counts


def counts() -> dict[str, dict[str, int]]:
    """framework -> model -> LOC, for every runner that exists on disk."""
    out: dict[str, dict[str, int]] = {}
    for framework, path in SOURCES.items():
        if os.path.exists(path):
            out[framework] = count_file(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "loc.json"))
    args = parser.parse_args()

    table = counts()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
        f.write("\n")

    width = max((len(k) for k in table), default=0)
    for framework in sorted(table):
        for model in sorted(table[framework]):
            print(f"{framework:<{width}}  {model:<7}{table[framework][model]:>5}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
