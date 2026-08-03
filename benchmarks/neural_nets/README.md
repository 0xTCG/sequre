# Neural networks: Sequre vs CrypTen vs MPyC

Secure *training* of two sequential feed-forward networks — forward pass,
backward pass and weight update all under MPC — measured for **time per epoch**,
**communication**, **fidelity against a float64 reference**, and **lines of
model code**.

Generated output lives in [REPORT.md](REPORT.md); raw per-run rows are in
[results/](results/).

Only sequential feed-forward networks are benchmarked, because that is what
Sequre's `Sequential` supports. Convolutional and recurrent layers exist as open
pull requests and are not shipped code; benchmarking them would be benchmarking
a branch.

## The two networks

| | `siren` | `mlp` |
|---|---|---|
| Source | Sitzmann et al. 2020, *Implicit Neural Representations with Periodic Activation Functions* | Mohassel & Zhang 2017, *SecureML*, §5.1 |
| Shape | 2–64–64–1 | 784–128–128–10 |
| Activation | `sin`, with `omega = 30` on every pre-activation | `relu` |
| Init | uniform(±1/fan_in) first layer, uniform(±√(6/fan_in)/ω) after | uniform(±1/√fan_in) |
| `n` | pixels of a square image, 8×8 to 64×64 | rows in the full-batch gradient step |
| Why | its nonlinearity is Decor's **best** case | its nonlinearity is Decor's **worst** case |

The pair is chosen so the two results bracket rather than flatter. Decor
evaluates `sin` exactly in three rounds and its derivative `cos` in three more,
so SIREN is the architecture the protocol was waiting for. ReLU is a
comparison, which no addition theorem helps with, and the SecureML MLP is the
workload every competing framework has spent a decade optimising — it is the
reference network of SecureML, MiniONN, ABY3, Falcon and CrypTen alike, so the
numbers here sit next to a published literature.

Both train with mean squared error and batch gradient descent with Nesterov
momentum, for `EPOCHS = 2`. The benchmark measures per-epoch cost and
per-epoch fidelity, not convergence: nobody trains a network to completion
under MPC today, and a framework 20% faster over two epochs is 20% faster over
two hundred.

## What is being compared

| Column | Framework | Parties | How the network is expressed |
|---|---|---|---|
| **Sequre** | Sequre (this repo), Codon-compiled native code, `--use-ring` | 3 (CP0 dealer, CP1/CP2 online) | `Sequential` of `Dense` layers. Activations come from Decor: `sin` and `cos` are exact and constant-round, `relu` is a comparison. Reported at 128- and 192-bit shares, plus a 64-bit width-matched control. |
| **CrypTen (TFP)** | Meta's CrypTen 0.4.1 on PyTorch | 2, one of which makes the triples — **not secure**, see below | `crypten.nn`, CrypTen's own layer library with autograd over secret shares. The MLP is a stock `Sequential`; SIREN needs one custom module, since CrypTen has no sine layer, but its backward pass still comes from CrypTen's autograd. |
| **CrypTen (TTP)** | Meta's CrypTen 0.4.1 | 2 online + a separate `TTPServer` | Identical code; only the source of correlated randomness differs. This is the security-matched column. |
| **MPyC** | MPyC 0.11, pure Python | 3, Shamir, threshold 1 | MPyC ships no neural-network layer at all — no module, no autograd, no optimizer, no loss. The network, its backward pass and its update rule are written out from primitives. SIREN is not implemented; see below. |

### Why MPyC cannot run SIREN

MPyC has no secure trigonometric function, so a sine has to be a Taylor series,
and a Taylor series of `sin` is usable only near zero. SIREN's pre-activations
are scaled by `omega = 30`, and the degree-15 series — the one the activation
benchmark's MPyC runner uses, where it is accurate to ~1e-5 over [-π, π] —
evaluates `sin(30)` as **−8.8 × 10⁹**, against a true value of −0.988. That
result does not even fit: `SecFxp(64)` with 32 fractional bits tops out at
±2.1 × 10⁹, so it wraps. It is an overflow, not an approximation.

Making it work would need range reduction, `x mod 2π`, which under MPC means a
secure division and floor for every element, of every layer, of every epoch.
Those rows are recorded as `skipped` with that reason rather than filled with a
number that would be reporting overflow as inaccuracy.

That is the sharpest statement this benchmark makes about what Decor buys: not
that the sine is faster, but that a whole architecture is reachable that
otherwise is not.

## Results

Numbers below are from `REPORT.md`, on an Apple Silicon macOS host, all parties
on localhost. Read them with the caveats at the end.

**Secure training reproduces plaintext training to eight decimal places.**
Maximum absolute difference between the trained network's predictions and the
float64 reference's, after two epochs:

| model | n | Sequre (192b) | CrypTen TTP (64b) | MPyC (128b) |
|---|---|---|---|---|
| siren | 64 | **2.7e-08** | 1.2e-02 | no secure sine |
| siren | 4096 | **1.2e-07** | 1.4e-02 | no secure sine |
| mlp | 8 | **5.1e-09** | 2.2e-04 | 3.7e-09 |
| mlp | 512 | **6.6e-09** | 2.7e-04 | not run |

This is the result that does not depend on the machine or the batch size. It is
also the one that had to be earned: an error of 1e-08 after two epochs of
forward pass, backward pass and momentum update means every truncation in the
chain is behaving, and it is what caught the missing `omega` factor in the
backward pass described at the end of this file.

CrypTen's five-orders-of-magnitude gap is precision, not a broken protocol —
16 fractional bits, and on SIREN those bits pass through a sine of an `omega =
30` pre-activation twice before being backpropagated. MPyC, at the same 32
fractional bits as Sequre, lands where Sequre lands, which is the control that
shows the fractional bits are doing the explaining.

**SIREN is where the protocol shows.** Against the security-matched CrypTen TTP
column:

| n | Sequre 192b | Sequre 128b | CrypTen TTP | Sequre 192b MiB | CrypTen MiB |
|---|---|---|---|---|---|
| 64 | 0.058 s | **0.031 s** | 0.076 s | **7.5** | 11.5 |
| 256 | 0.142 s | 0.074 s | **0.096 s** | **24.5** | 45.3 |
| 1024 | 0.571 s | 0.296 s | **0.232 s** | **92.6** | 180.5 |
| 4096 | 2.715 s | 1.273 s | **0.630 s** | **364.9** | 721.4 |

Sequre leads on wall-clock at the smallest size and loses by ~4x at the
largest, the same crossover the activation benchmark found. But it **sends half
the bytes at every size** while carrying three times the share width and twice
the fractional bits — Decor's sine is three rounds and its
derivative `cos` is three more, where CrypTen pays for an approximation and its
backward pass. On a WAN, where the communication column dominates, that
inverts the ranking.

**The MLP is where it does not.** CrypTen is 60x faster at n = 512 (0.43 s
against 27.1 s per epoch) and sends a quarter of the bytes.

The reason is arithmetic throughput, not protocol. Going from n = 8 to n = 512
multiplies the data by 64; Sequre's time grows 61x while its traffic grows only
13x, so the cost is local and not on the wire. One epoch at n = 512 is roughly
1.8 × 10⁸ multiply-accumulates over 192-bit shares. At 27 s that is ~150 ns
each, against the ~3.5 ns per element measured for a bare local `mul_mod` in
[the activation benchmark](../activations/README.md#why-crypten-wins-at-large-n)
— so the matmul path costs about 40x its own arithmetic in Beaver masking,
truncation and allocation. CrypTen does the same work as one vectorised
`torch.int64` matmul with local truncation and no allocator in the loop.

SIREN does not hit this because its matrices are tiny (2×64, 64×64, 64×1); its
cost is the activation, which is exactly the term Decor makes cheap. Two
architectures, two regimes, and the benchmark reports both.

Matching CrypTen's word size takes the MLP deficit from 57x to 21x — see
[Matching CrypTen's 64-bit width](#matching-cryptens-64-bit-width) — so about
half of it is width and the rest is the matmul path.

**Narrowing the shares helps, but not linearly and not uniformly:**

| model | n | 192b | 128b | 64b |
|---|---|---|---|---|
| siren | 1024 | 0.571 s | 0.296 s (1.9x) | 0.232 s (2.5x) |
| siren | 4096 | 2.715 s | 1.273 s (2.1x) | 0.908 s (3.0x) |
| mlp | 128 | 3.791 s | 2.944 s (1.3x) | 1.724 s (2.2x) |
| mlp | 512 | 27.09 s | 26.68 s (1.0x) | 10.20 s (2.7x) |

Two different effects are visible here. SIREN improves smoothly with width
because its cost is truncation, and truncation scales with the word size.
The MLP does not: 192 → 128 buys it nothing at n = 512, then 128 → 64 buys
2.7x. That discontinuity is the machine word — at 128 and 192 bits every
coefficient is a multi-word integer, at 64 bits it fits in a register, and the
MLP is the workload dominated by raw multiply throughput rather than by
communication. Width is therefore not one knob but two: a linear one on
truncation volume and a step one at the register boundary.

Note that `MPC_INT_SIZE = 128` also drops `MPC_NBIT_V` from 64 to 28 bits,
below the κ = 40 conventional for statistical security, and 64 drops it to 10.
These are diagnostics, not deployable configurations — see below.

### Matching CrypTen's 64-bit width

Since only Sequre is tunable, the direct way to take width out of the
comparison is to bring Sequre down to CrypTen's ring. `MPC_INT_SIZE = 64` is a
2⁶¹ − 1 field with K = 32, **F = 16 — exactly CrypTen's 16 fractional bits** —
and V = 10. Run it with `./sequre/run_widths.sh 64 128 192`; it is not in the
default set.

**On the MLP it is the fairest comparison in the suite, and it lands where the
activation benchmark predicted:**

| n | Sequre 64b | CrypTen TTP 64b | Sequre 64b error | CrypTen error |
|---|---|---|---|---|
| 8 | 0.165 s | 0.054 s | 2.7e-04 | 2.2e-04 |
| 128 | 1.724 s | 0.154 s | 2.7e-04 | 3.0e-04 |
| 512 | 10.20 s | 0.478 s | 3.7e-04 | 2.7e-04 |

The accuracy columns become **the same number**. Both are precision-bound at
2⁻¹⁶, not algorithm-bound, and Sequre's 10⁵ advantage at 192 bits is revealed as
something bought with fractional bits rather than something the protocol
provides for free — the same conclusion the activation benchmark reached about
`relu`. Width also accounts for a real slice of the speed gap: at n = 512 the
deficit against CrypTen TTP narrows from 57x to 21x. The remaining 21x is the
matmul cost analysed above, and is not explained by width.

**On SIREN, 64 bits does not degrade the result — it destroys it.**

| n | 64b error | 64b final loss | float64 reference |
|---|---|---|---|
| 64 | 3.1e+01 | 219.4 | 0.0285 |
| 256 | 1.3e+00 | 0.388 | 0.0330 |
| 1024 | 1.3e+02 | 4235.9 | 0.0346 |
| 4096 | 1.9e+01 | **−32011.2** | 0.0354 |

A final loss of −32011 is the diagnosis on its own: the loss is a sum of
squares divided by a positive constant and **cannot be negative**, so the
fixed-point representation wrapped. This is overflow, not accumulated error,
and the numbers in those cells carry no information beyond "it broke".

The cause is the `omega`. SIREN's forward pass is `sin(30·z)` and its backward
pass carries that same factor of 30 per sine layer — 900 across the two of
them — on top of weights, momentum and two epochs of accumulation. At K = 32
with F = 16 there are only 16 bits of integer headroom, about ±32768, and the
amplified gradients run out of it. At 128 and 192 bits, where K = 64 gives 32
integer bits, the identical training is exact to 1e-08.

So the honest summary of the width axis is that it splits by architecture:

- **The MLP** is width-*tolerant*. 64 bits costs it nothing in accuracy that
  CrypTen does not also pay, and buys 2.7x the speed. If you only ever ran the
  MLP you would conclude Sequre's precision advantage is optional.
- **SIREN** is width-*bound*. The thing that makes it a good fit for Decor — a
  high-frequency periodic activation with a large `omega` — is exactly what
  makes it need the integer headroom that a 64-bit build does not have.

Neither the 64-bit nor the 128-bit build is deployable on its own terms (V = 10
and V = 28 respectively, against a conventional κ = 40). They are diagnostics.
The repository default of 192 is left untouched, and it is the only column in
which both networks are simultaneously fast enough to run and correct.

**MPyC is 344x slower and sends 23x more**, at the one size it was run:
153.1 s per epoch against Sequre's 0.446 at n = 8, and 1,391 MiB against
60 MiB. It is a pure-Python runtime executing a hand-written backward pass, so
this is a runtime result rather than a protocol one — and its accuracy column
matches Sequre's, so nothing is being traded for the time.

**Detaching the dealer makes training slower, not faster.** Unlike the
activation benchmark, where `mpc.detach_dealer()` took 15–46% off, here it
costs:

| model | n | attached | detached |
|---|---|---|---|
| siren | 64 | 0.058 s | 0.074 s |
| siren | 1024 | 0.571 s | 0.532 s |
| mlp | 8 | 0.446 s | 0.769 s |
| mlp | 32 | 1.032 s | 1.515 s |

A single activation call buffers kilobytes; a training run buffers hundreds of
megabytes, and managing that buffer costs more than removing the dealer from
the critical path saves. Past ~0.5 GB it stops being a trade-off at all and the
transport fails outright, which is why the large cells are skipped. The online
column is reported because it is the standard framing for Decor, but for
training workloads it is not currently a win.

## Lines of model code

The claim a secure-ML framework makes is not only "this is fast" but "you can
write your network in it". That second claim is usually made in prose. Here it
is a number, counted mechanically by [`loc.py`](loc.py) from markers in the
runners:

    # LOC-BEGIN mlp
    ...
    # LOC-END mlp

| model | Sequre | CrypTen | MPyC |
|---|---|---|---|
| siren | **21** | 29 | *cannot be written* |
| mlp | **18** | 21 | 50 |

Inside the markers: constructing the network, choosing loss and optimizer,
running the training loop, producing predictions. Outside, in every runner
equally: data, injection of the shared initial weights, timing, communication
accounting, output. Framework-provided machinery is not counted, which is the
point — Sequre's `Dense` and CrypTen's `nn.Linear` cost their users nothing,
while MPyC has no layer to provide, so its forward pass, backward pass and
optimizer are all inside its markers.

Two honest deductions from CrypTen's count. Its runner writes the loss out as
`((out - y)²).sum() / (2·rows)` instead of `nn.MSELoss()`, and the Nesterov
update out in full instead of `crypten.optim.SGD(..., nesterov=True)`. Neither
is a criticism of CrypTen: CrypTen's MSELoss averages over elements where
Sequre's normalises by rows, and PyTorch's Nesterov is a different algebraic
variant from Sequre's, so using the stock versions would train a *different*
network and the difference would show up in the accuracy column looking like a
protocol effect. Roughly seven of CrypTen's lines are that alignment, and would
be one line each in ordinary use.

The counts are physical lines and so are sensitive to formatting. Every runner
is therefore written to the same convention — ~88 columns, one logical argument
group per continuation line, no statement compression — which means a layer
whose constructor takes six keyword arguments costs several lines. That is the
intended reading: those arguments are things the user has to supply.

The count is not a quality measure on its own. A framework can be terse because
it is expressive or because it is inflexible, and `loc.py` cannot tell the two
apart. Read the LOC column next to the accuracy and timing columns, not instead
of them.

## Method

**Every framework trains the identical network from the identical starting
point.** Architecture, data, hyperparameters and initial weights all come from
[`spec.py`](spec.py); the runners load the weights over whatever their own
initializers produced. Reproducing an RNG stream across Codon, PyTorch and pure
Python is not possible in general, so the benchmark does not try: `spec.py`
defines a 63-bit LCG small enough to mirror in three languages in five lines,
and everything random is drawn from it.

**The accuracy column is computed in one place.** Each runner emits its trained
network's predictions at `spec.witness_indices(n)` — up to 256 evenly spaced
rows — and [`report.py`](report.py) compares them against
[`ref.py`](ref.py), a float64 numpy transcription of the same training. Three
runners each computing their own error against their own copy of a reference
would also be measuring how well those copies agree.

`ref.py` is transcribed from Sequre's `Sequential` rather than from a textbook,
because Sequre's semantics are what the other runners are being asked to
match: the loss and its derivative are pre-normalised by the row count, the
gradient step is Nesterov in Sequre's two-line form, and a layer computing
`f(s·(xW + b))` carries the public factor `s` into its backward pass.

**Data is synthetic and analytic**, not MNIST or a photograph. Three runners in
three languages have to agree bit-for-bit on the training set, and what is being
measured is protocol cost per unit of arithmetic, which a real dataset does not
change. The SIREN target is a genuine band-limited image, `0.5·sin(3x)·cos(4y)`
over a coordinate grid — exactly the kind of signal SIREN exists to fit — so
its fit quality still means something.

## Caveats

These matter for reading the numbers honestly.

- **Threat models are not identical.** Sequre runs 3 parties with an offline
  dealer; CrypTen TFP runs 2 online parties, one of which is trusted; CrypTen
  TTP runs 2 online plus a dealer; MPyC runs 3 parties with Shamir sharing and
  no dealer. Compare against the CrypTen **TTP** column for a security-matched
  reading.
- **CrypTen's default provider is not a secure configuration.** With
  `provider=TFP`, party 0 generates the Beaver triple `(a, b, c = a·b)` in the
  clear. Beaver multiplication opens `x − a` and `y − b`, and party 0 knows `a`
  and `b`, so it reconstructs both private inputs outright. With two parties
  there is no third place for that trust to live. It is a speed ceiling, not a
  threat model; the suite runs both providers and reports them separately.
- **Share width and fixed-point precision differ and are not freely tunable.**
  Sequre uses 192-bit shares with 32 fractional bits (128- and 64-bit also
  reported), MPyC a 128-bit field with 32, CrypTen a 64-bit ring with 16.
  CrypTen is doing materially less work per element than either. Only Sequre is
  tunable — CrypTen's ring is hardwired to `torch.int64` — so no single row
  equalises all three; the 64-bit column is the closest, and only for the MLP.
- **The Sequre 64-bit SIREN cells are a numerical failure, not a measurement.**
  A negative sum of squares is a wrapped fixed-point value. They are left in
  the report because a build that silently produces `-32011` where the answer
  is `0.035` is worth seeing, but they are not an accuracy figure and must not
  be read as one.
- **CrypTen's SIREN error is precision, not a broken protocol.** 16 fractional
  bits through a sine of an `omega = 30` pre-activation, twice, then
  backpropagated, is where its error comes from. It is the same trade CrypTen
  makes everywhere; it just costs more in an architecture built on a
  high-frequency nonlinearity.
- **Two measurement modes are reported.** In `end_to_end`, correlated
  randomness is generated inside the timed region for both Sequre and CrypTen.
  In `online`, Sequre's dealer is detached with `mpc.detach_dealer()`; CrypTen
  and MPyC have no separable offline phase and repeat their end-to-end numbers
  there, so that table is Sequre-favourable by construction and is labelled as
  such.
- **The online mode does not cover every cell.** Detaching buffers every byte
  the dealer would have sent until the block exits, and one training run of the
  MLP at n = 512 moves ~0.8 GB of randomness. Wrapping the whole pass, as the
  activation benchmark does, fails outright with `Socket connection broken`, so
  the online pass here detaches per cell and skips the large ones. That ceiling
  is a property of the implementation and is reported rather than worked
  around.
- **MPyC is capped at n = 8 by default.** A pure-Python runtime with no layer
  library needs minutes per epoch on a 784×128 first layer. Larger sizes are
  recorded as skipped rather than dropped; pass `--max-n` to fill them in and
  expect it to take hours.
- **Localhost only, single machine, no pinning.** Network latency is near zero,
  which favours protocols with more rounds. On a WAN the communication table
  becomes the dominant term and the ranking will shift.
- **CrypTen 0.4.1 is from 2022** and is the last release. It is pinned against
  torch 2.4.1 because newer torch removed an internal module it imports.
- **CrypTen's runner writes its results from inside each worker**, not by
  returning them to the parent. `run_multiprocess` calls `process.join()`
  before it drains its result queue, so a child whose return value exceeds the
  pipe buffer can never exit and the parent waits on it forever — the standard
  `multiprocessing.Queue` deadlock. The prediction witnesses cross that
  threshold once both models are run together, which is a silent hang rather
  than an error. If you extend this runner, keep bulk data out of the return
  value.

## Layout

```
spec.py               architectures, data, initial weights, hyperparameters -- the source of truth
ref.py                float64 numpy transcription of Sequre's training loop
common.py             result record, JSONL writer, timing helper
loc.py                counts lines of model code from the runners' markers
report.py             builds REPORT.md from results/*.jsonl, computes every accuracy column
sequre/bench_nn.codon Sequre runner
sequre/run_widths.sh  runs Sequre at 128 and 192 bits (64 on request), restoring settings.codon
crypten/bench_nn.py   CrypTen runner, both providers
mpyc/bench_nn.py      MPyC runner
setup.sh              links or creates the .venv the two Python runners need
run_all.sh            runs everything, writes REPORT.md
results/              one JSONL per framework and party, one JSON object per row
```

`spec.py` is authoritative. The Codon runner cannot import Python, so it mirrors
the constants in `sequre/bench_nn.codon` and echoes them into its output header;
`report.py` refuses to build a table if any runner's header disagrees.

## Running it

```bash
# one-time: Python environment for CrypTen and MPyC
# (links to benchmarks/activations/.venv if that already exists)
./benchmarks/neural_nets/setup.sh

# everything, then the report
./benchmarks/neural_nets/run_all.sh

# or a subset
./benchmarks/neural_nets/run_all.sh sequre crypten
```

Individually:

```bash
# from the repo root -- -release is essential, the default build carries backtraces
sequre run -release benchmarks/neural_nets/sequre/bench_nn.codon --local --use-ring

# both deployable share widths as separate report columns; restores
# settings.codon on any exit
./benchmarks/neural_nets/sequre/run_widths.sh

# add the 64-bit width-matched control (breaks SIREN -- see above)
./benchmarks/neural_nets/sequre/run_widths.sh 64 128 192

cd benchmarks/neural_nets/crypten && ../.venv/bin/python bench_nn.py --provider TFP
cd benchmarks/neural_nets/crypten && ../.venv/bin/python bench_nn.py --provider TTP
cd benchmarks/neural_nets/mpyc    && ../.venv/bin/python bench_nn.py -M3

cd benchmarks/neural_nets && .venv/bin/python report.py --out REPORT.md
cd benchmarks/neural_nets && .venv/bin/python loc.py       # just the LOC table
```

Budget several hours for a full run. Sequre spends ~40 s per epoch on the MLP
at n = 512, and MPyC needs ~150 s per epoch at n = 8.

## What this needed from the standard library

Three things were missing and were added, all of them visible in the API rather
than in the benchmark:

- **`"mse"` loss.** `stdlib/sequre/stdlib/learn/neural_net/loss.codon` shipped
  only hinge loss. A regression network cannot be trained with a
  classification loss.
- **`kernel_interval` and `kernel_scale` on `Dense`.** The initializers produce
  fixed distributions, so an initialization that depends on fan-in — which
  SIREN's does, and which most modern initializations do — could not be
  expressed.
- **`pre_act_scale` on `Dense`.** SIREN computes `sin(ω·(xW + b))`. The factor
  cannot be folded into the weights without rescaling the very gradient it
  exists to boost, so it is a layer parameter, and it reappears in the backward
  pass as a factor on `dhidden`.

One bug was fixed along the way: `MPCBoolean.normalizer` assumed a flat share
and produced a type error on a matrix, which made every Decor activation that
divides — `sigmoid`, `tanh`, `tan`, `cot` — unusable inside a layer, and since
the activation dispatch realises every branch, that made `Dense` unusable on a
2-D `Sharetensor` altogether. It now flattens and reshapes around the protocol,
the same dispatch `is_positive_ring` next to it already used.
