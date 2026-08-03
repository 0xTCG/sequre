# Core operations: Sequre vs CrypTen vs MPyC

The seven operations every arithmetic MPC framework implements in its own
library — sharing, addition, multiplication by a public constant,
multiplication of two secrets, an inner product, a fixed polynomial, and
reconstruction — measured for **latency**, **throughput**, **communication**
and **accuracy against a float64 reference**.

Generated output lives in [REPORT.md](REPORT.md); raw per-run rows are in
[results/](results/).

## Why this suite exists

The other two suites measure *functions*. [`activations/`](../activations/)
measures `exp`, `sigmoid`, `tanh`, `relu`; [`neural_nets/`](../neural_nets/)
measures a trained SIREN and the SecureML MLP. Those numbers mix two things
that move independently: the cost of the underlying protocol, and the quality
of the approximation each framework's function library happens to use. A
framework can lose the activation benchmark by having a slow multiply or by
having a wasteful sigmoid, and the table cannot tell you which.

This suite drops to the layer underneath and measures only that. It takes the
**intersection** of what every framework provides, not the union — nothing here
has to be emulated by anyone. What that costs is expressiveness: this benchmark
says nothing about nonlinear functions, which is exactly what the other two are
for. What it buys is a clean read on protocol cost, and cells that need no
caveat about who wrote them.

## The operations

| op | what it is | secret multiplications |
|---|---|---|
| `share` | secret-share a length-`n` vector of cleartext values | 0 |
| `add` | elementwise `a + b`, both secret | 0 (local in every scheme here) |
| `mul_public` | elementwise `a * 3`, public multiplier | 0 |
| `mul` | elementwise `a * b`, both secret | `n` |
| `dot` | inner product of two length-`n` secret vectors | `n`, one truncation |
| `polynomial` | `1 + 2x + 3x² + 4x³` by Horner | `2n` |
| `open` | reconstruct a length-`n` secret vector | 0 |

`add` and `mul_public` are local in all three frameworks, so those rows measure
constant factors — language, memory layout, encoding — rather than protocol.
`mul` and `dot` measure the protocol at fixed depth and growing width;
`polynomial` holds width fixed and adds depth. Horner's first product has a
public left operand and is local, which is why the polynomial costs two secret
multiplications and not three.

The public multiplier is `3`: an integer, so the product of two scale-`2^f`
encodings divides back exactly and no framework's rounding policy leaks into
the row, and not a power of two, so none of them can shortcut it into a shift.

## What is being compared

| Column | Framework | Parties | Fixed point | How the op is expressed |
|---|---|---|---|---|
| **Sequre** | Sequre (this repo), Codon-compiled native code, `--use-ring` | 3 (CP0 dealer, CP1/CP2 online) | 32 fractional bits | `Sharetensor.enc`, `+`, `*`, `.dot`, `.reveal`, through the `@sequre` IR pass. Reported at 128- and 192-bit shares plus a 64-bit control. |
| **CrypTen (TFP)** | Meta's CrypTen 0.4.1 on PyTorch | 2, one of which makes the triples — **not secure** | 16 fractional bits | `crypten.cryptensor`, `+`, `*`, `.matmul`, `.polynomial`, `.get_plain_text` |
| **CrypTen (TTP)** | Meta's CrypTen 0.4.1 | 2 online + a separate `TTPServer` | 16 fractional bits | Identical code; only the source of correlated randomness differs. The security-matched CrypTen column. |
| **MPyC** | MPyC 0.11, pure Python | 3, Shamir, threshold 1 | 32 fractional bits | `mpc.input`, `+`, `*`, `mpc.np_matmul`, `mpc.output` |

Unlike the activation benchmark, where MPyC's every entry had to be built from
primitives, here every cell in every column is a library call.

## Caveats

### Threat models are not identical

- **CrypTen TFP** is 2-party with party 0 generating the Beaver triples. Beaver
  multiplication opens `x-a` and `y-b`, and party 0 knows `a` and `b`, so it can
  reconstruct party 1's inputs outright. It is a speed ceiling, not a security
  baseline. **TTP** is the column to compare against.
- **Sequre and MPyC** are 3-party, semi-honest, threshold 1.

### The `share` row is the least comparable one

Its communication cell reads 0 KiB for Sequre and CrypTen and non-zero for
MPyC, and that is a difference in what the operation *is*, not in efficiency:

- **CrypTen** `cryptensor()` on a tensor both parties already hold splits it
  locally. Nothing is transmitted because nothing needs to be.
- **Sequre** `Sharetensor.enc` shares from CP0, and the reported party is CP1,
  which receives rather than sends.
- **MPyC** `mpc.input(..., senders=0)` genuinely transmits, and MPyC's reported
  party is the sender.

Compare the latency column across it if you like, but read the byte column as
three different questions rather than one.

### Fixed-point widths differ

CrypTen is fixed at 16 fractional bits and cannot be raised without changing
its ring. Sequre and MPyC both carry 32, and Sequre's 64-bit control carries 16
to match CrypTen. Read the accuracy table with the widths in mind: CrypTen is
not losing to Sequre-at-192 on protocol quality there, it is carrying half the
fractional bits — which is why the 64-bit Sequre control lands on CrypTen's
error and not on Sequre's.

### Timings below clock resolution are blank

`add` on a short vector finishes inside one tick of the Codon runner's
`time.time()`. Rather than report 0 ms and infinite throughput, those cells are
emitted as null and render as `--`; the accuracy fields for the same cell are
still reported.

### Sequre reports two modes

Every Sequre cell is measured twice: `end_to_end`, with the dealer's randomness
generation inline on the critical path, and `online`, with it detached via
`mpc.detach_dealer()`. CrypTen and MPyC have no separable offline phase and
appear in the online table with their end-to-end numbers. That asymmetry is
stated in the report rather than hidden.

## Frameworks considered and not included

**StoffelMPC** (`stoffelcrypto`, HoneyBadgerMPC) was evaluated for this suite
and left out as too immature to benchmark fairly. Recorded here so the next
person does not have to redo the investigation:

- It ships no comparison protocol, so it cannot appear in `activations/` or
  `neural_nets/` at all — no `relu`, no `argmax`, no nonlinear functions.
- Its `mul_fixed` is scalar-only; `add_fixed`/`sub_fixed` are vectorised but
  there is no batched multiply anywhere in the crate, so an `n`-element product
  is `n` sequential protocol invocations.
- It has no user-facing call for sharing a plain vector or opening one; I/O
  goes through an `InputClient`/`OutputClient` protocol against preprocessed
  masks.
- Three separate faults surfaced while driving it: PRandBit batches above 256
  never terminate (the `MAX_PRAND_SESSIONS = 256` guard is commented out, and
  the protocol runs over GF(2^8)); `mul_fixed` refills its PRandBit and
  PRandInt stores but never checks the Beaver triple store, so triples run dry
  and it returns `NotEnoughPreprocessing`; and `share_gen.rs:378` panics
  intermittently under sustained preprocessing with an index equal to its own
  batch length.

Worth revisiting once the multiply is batched and the preprocessing is stable.

## Running it

```bash
./setup.sh          # Python 3.12 venv (reused from ../activations if present)
./run_all.sh        # all three, then the report
./run_all.sh sequre crypten   # a subset

# Sequre at more than one share width, side by side in one report
sequre/run_widths.sh 64 128 192
```

Individual runners take `--sizes`, `--reps` and `--ops`, so a single cell can be
re-measured without rerunning the suite.
