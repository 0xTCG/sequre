# Activation functions: Sequre+Decor vs CrypTen vs MPyC

A side-by-side benchmark of every function Decor implements — `exp`, `sigmoid`,
`tanh`, `relu`, `sin`, `cos`, `tan`, `cot`, `sinh`, `cosh` and a degree-3
`polynomial` — evaluated under secure computation, measuring **latency**,
**throughput**, **communication** and **accuracy** on identical inputs.

Generated output lives in [REPORT.md](REPORT.md); raw per-run rows are in
[results/](results/).

## What is being compared

| Column | Framework | Parties | How the activation is computed |
|---|---|---|---|
| **Sequre+Decor** | Sequre (this repo), Codon-compiled native code, run with `--use-ring` | 3 (CP0 dealer, CP1/CP2 online) | Decor's exact protocol: the dealer precomputes `f(r)` for a random mask `r`, the parties reveal `x - r`, evaluate `f` on it *in the clear*, and recombine with a cheap algebraic identity. ReLU uses Decor's reduced-ring comparison. |
| **Sequre/Fourier** | Sequre | 3 | Degree-15 Fourier series evaluated through Decor's constant-round trigonometric protocol. ReLU uses Sequre's ordinary full-width comparison. The approximation-based path, for contrast. |
| **CrypTen (TFP)** | Meta's CrypTen 0.4.1 on PyTorch | 2, one of which makes the triples — **not secure**, see below |  CrypTen's own library functions, defaults unmodified: `exp` by limit approximation (8 iterations), `sigmoid = 1/(1+exp(-x))` with Newton-Raphson reciprocal (10 iterations), `tanh = 2·sigmoid(2x) − 1`, `relu` by arithmetic-to-binary comparison. |
| **CrypTen (TTP)** | Meta's CrypTen 0.4.1 on PyTorch | 2 online + a separate `TTPServer` | Identical library functions; only the source of correlated randomness differs. This is the security-matched column. |
| **MPyC** | MPyC 0.11, pure Python | 3, Shamir, threshold 1 | MPyC ships no activation library, so the four functions are built from MPyC primitives using **exactly CrypTen's algorithms** (see below). |

### Why MPyC is the third framework

The three columns are chosen to separate two independent variables:

- **CrypTen vs MPyC** holds the *algorithm* fixed and varies the *runtime*.
  Both evaluate the same limit approximation for `exp` and the same
  `2·sigmoid(2x) − 1` for `tanh`; CrypTen dispatches into vectorised C++
  kernels, MPyC interprets Python. The gap is runtime cost, nothing else.
- **Sequre+Decor vs both** holds the runtime broadly fixed (compiled native
  code, localhost sockets) and varies the *protocol*. Decor does not
  approximate at all, so its accuracy column is a protocol property rather than
  a tuning parameter.

MPyC is also simply practical: it is actively maintained, pip-installable
everywhere with no native build, and is a genuine general-purpose MPC framework
with secure fixed-point arithmetic. SecretFlow's SPU was the other candidate
and would have been a closer architectural peer to Sequre, but it publishes no
macOS/arm64 wheel.

## Results

Numbers below are from `REPORT.md`, on an Apple Silicon macOS host, all parties
on localhost. Read them with the caveats in the next section.

**Accuracy — Decor wins by four to five orders of magnitude.** Max absolute
error against float64, at n = 1024:

| function | Sequre+Decor | CrypTen | MPyC | Sequre/Fourier |
|---|---|---|---|---|
| exp | 1.4e-07 | 4.2e-01 | 3.5e-01 | 1.0e+01 |
| sigmoid | 1.1e-07 | 2.4e-03 | 8.6e-04 | 5.0e-01 |
| tanh | 8.8e-07 | 3.6e-03 | 1.7e-03 | 1.0e+00 |
| relu | 2.3e-10 | 1.5e-05 | 1.2e-10 | 2.3e-10 |

This is the result that does not depend on the machine, the batch size, or the
network. Decor evaluates the function on plaintext, so its error is
fixed-point rounding; every other column carries approximation error that grows
towards the ends of the interval. The `interior 80%` table in `REPORT.md`
confirms this is not a boundary artifact: trimming the endpoints leaves
CrypTen's `exp` error at 1.4e-01 and Sequre/Fourier's at 6.5e-01, while
Decor stays at 7.2e-08.

### Round counts explain almost everything

Decor's per-call cost tracks its round count, and the round count varies by an
order of magnitude across its own function set (n = 1024, 192-bit):

| function | rounds | KiB | what the rounds are |
|---|---|---|---|
| `sin`, `cos` | **3** | 72 | the trigonometric protocol, and nothing else |
| `exp` | 13 | 256 | 3 for the protocol + **7 for a comparison** + truncations |
| `sigmoid` | 59 | 1,105 | the above plus a Newton division |

This is the answer to "surely `exp` should be fast, it has no division".
It has no division — but it *does* have a comparison. Non-periodic functions
need their input confined to an interval, and Decor handles the case where the
mask pushes `x − r` outside that interval by evaluating at both candidate
points and selecting between them with `decor_gt`. Measured on its own,
`decor_gt` is 7 rounds and 17.9 ms at n = 8192 — i.e. essentially the whole
cost of `exp` (13 rounds, 21.3 ms). Periodic functions need no confinement, so
`sin`/`cos` skip it entirely and land at 3 rounds.

That makes the trigonometric column the strongest result in the suite. Against
the security-matched CrypTen TTP at n = 1024:

| function | Decor | CrypTen TTP | speed | Decor error | CrypTen error |
|---|---|---|---|---|---|
| `sin` | 0.58 ms | 16.64 ms | **28.5x** | 3.1e-09 | 1.5e-02 |
| `cos` | 0.45 ms | 19.83 ms | **43.9x** | 2.5e-09 | 1.7e-02 |
| `tan` | 5.44 ms | 61.12 ms | **11.2x** | 3.6e-09 | 3.1e-02 |
| `cot` | 5.31 ms | 70.91 ms | **13.4x** | 6.0e-09 | 4.0e-02 |
| `sinh` | 2.77 ms | 9.37 ms | **3.4x** | 4.3e-09 | 3.9e-02 |
| `cosh` | 2.78 ms | 8.74 ms | **3.1x** | 5.0e-09 | 4.0e-02 |
| `polynomial` | 3.18 ms | 2.19 ms | 0.7x | 5.9e-08 | 9.6e-04 |

`sin` and `cos` are the only functions where Decor's lead *survives the batch
size*: at n = 8192 they are still **7.5x faster** than CrypTen TTP while sending
8x less (576 KiB vs 4,736 KiB), where `exp` has fallen to 0.3x. Constant-round
really does mean constant-round.

`polynomial` is CrypTen's one clear win — its exponentially-growing-tree
evaluation is genuinely good, 0.7x at n = 1024 falling to 0.1x at n = 8192 —
though still at 10^4 the error.

**Latency — Decor wins decisively below ~1000 elements per call, and loses
above it.** Speedup of Sequre+Decor over CrypTen, for the original four:

| n | exp | sigmoid | tanh | relu |
|---|---|---|---|---|
| 8 | 20.0x | 29.1x | 29.2x | 50.3x |
| 128 | 7.8x | 7.9x | 9.8x | 13.0x |
| 1024 | 0.9x | 1.6x | 1.7x | 3.0x |
| 8192 | 0.2x | 0.3x | 0.3x | 0.6x |

The n = 8 row is the noisiest in the suite — Sequre's calls there take under
0.5 ms, so the ratio moves by 10-20 points between runs. The order of magnitude
is stable; the exact figure is not.

The two frameworks have opposite cost structures. Fitting the `exp` row:

| | fixed cost per call | marginal cost per element |
|---|---|---|
| Sequre+Decor | ~0.05 ms | ~2.6 µs |
| CrypTen | ~2.9 ms | ~0.13 µs |

CrypTen's per-call cost is nearly flat from n = 8 to n = 8192 — 1024x the
elements for 1.4x the time — so below ~1000 elements it is paying almost
nothing but fixed overhead, and Sequre's compiled runtime simply has ~60x less
of it. Above that, Sequre's ~20x higher marginal cost takes over. At n = 8192
CrypTen is 3–5x faster in wall-clock while still being ~10^4x less accurate.

**The three frameworks do not use the same share width**, which is the first
thing to know before reading the marginal-cost gap as a protocol result:

| | share arithmetic | fixed-point fractional bits |
|---|---|---|
| Sequre | 192-bit ring/field | 32 |
| CrypTen | 64-bit ring (`torch.int64`) | 16 |
| MPyC | 128-bit prime field | 32 |

Sequre carries 3x CrypTen's width and 1.5x MPyC's. Part of CrypTen's low
marginal cost is therefore bought with narrower words and half the fractional
precision — it is a point on a precision/performance curve, not a free win.
MPyC, at 128 bits and the same 32 fractional bits as Sequre, is the closer
precision peer, and Sequre outruns it by 400–600x.

### CrypTen's default provider is not a secure configuration

CrypTen ships `provider=TFP` (`TrustedFirstParty`). Reading
`crypten/mpc/provider/tfp_provider.py`, it generates the Beaver triple
`(a, b, c = a·b)` **in the clear at party 0** and then shares it. Beaver
multiplication opens `x − a` and `y − b`; party 0 knows `a` and `b`, so it
recovers `x` and `y` outright. With only two parties there is no third place
for that trust to live — "one of two parties is trusted" means the other party
has no privacy at all. It is a speed ceiling, not a threat model.

CrypTen's other two providers: `HomomorphicProvider` raises
`NotImplementedError` in 0.4.1, so `TrustedThirdParty` (TTP) is the only usable
secure option. It runs a separate `TTPServer` process, giving 2 online parties
plus a dealer — structurally the same shape as Sequre's 3-party model, where
CP0 is a dealer holding no share of the data. The suite now runs both and
reports them as separate columns.

Security costs CrypTen 1.2–1.9x:

| function (n = 1024) | TFP (insecure) | TTP (secure) | |
|---|---|---|---|
| exp | 3.15 ms | 5.21 ms | 1.7x |
| sigmoid | 12.56 ms | 15.47 ms | 1.2x |
| tanh | 12.49 ms | 15.77 ms | 1.3x |
| relu | 4.90 ms | 5.54 ms | 1.1x |

Against the security-matched column, Sequre+Decor at its own 192-bit default
leads at n = 1024 on everything — `exp` 1.8x, `sigmoid` 1.8x, `tanh` 1.9x,
`relu` 3.4x end-to-end — while still trailing at n = 8192 (0.3–0.6x).

Matched on *both* axes — Sequre at 64-bit with the dealer detached against
CrypTen TTP, i.e. same ring width and both with a real third-party dealer:

| function | n = 1024 | n = 8192 |
|---|---|---|
| exp | **5.2x faster** | 1.0x (parity) |
| sigmoid | **4.5x faster** | 0.8x |
| tanh | **4.6x faster** | 1.0x (parity) |
| relu | **8.2x faster** | **2.1x faster** |

That is the fairest single comparison this benchmark can make on speed, and it
reverses the earlier picture: Sequre leads by 4.5–8.2x at n = 1024 and reaches
parity or better at n = 8192 on three of four functions. The remaining
`sigmoid` deficit and the accuracy caveats of the 64-bit build (above) still
stand.

### Is 64 bits enough?

Not for everything, though the answer is more specific than "too small".

*Confidentiality of the sharing itself is fine.* Additive shares over Z_2^64
with a uniformly random mask are perfectly hiding; width does not enter into
it. Nothing here suggests CrypTen leaks.

*The budget is where 64 bits bites — but only for some protocols.* Sequre
states the requirement explicitly — `K + F + V + 2 < modulus bits`, with `V`
the statistical security parameter — and spends 174 bits as 64 + 32 + 64.
CrypTen has 64 bits **in total**, of which 16 are fractional, leaving ~48 for
the integer part with nothing set aside for statistical slack. Any subprotocol
that masks a value and reveals it wants the mask ~2^κ larger than the value, and
at κ = 40 that would leave 8 bits of integer range. CrypTen instead leans on
binary sharing plus a trusted randomness provider, which is a coherent design,
just a different and weaker point than a 2^-64 statistical guarantee.

**Decor's comparison does not spend `V` at all.** In this repository `V` has
exactly two consumers, both in `mpc/fp.codon`: `MPCFP.trunc`, whose mask is
`k + V` bits wide, and `__normalizer_even_exp`, which backs division and sqrt.
`decor_gt` reaches neither. It goes through
`MPCBoolean._is_positive_ring`, which masks with a *full-width* uniform element
of the power-of-two ring, so revealing `x + r` is perfectly hiding rather than
statistically hiding. That is what lets it run on a reduced `MPC_NBIT_F + 1`
bit ring at all — and it is where the 7.6x communication saving on `relu`
comes from.

Measured truncation counts per call make the split concrete:

| operation | truncations (= `V` uses) |
|---|---|
| `decor_gt` alone | **0** |
| `decor` relu (rescale + gt + multiply) | 1 — entirely from the public rescale |
| Sequre baseline relu (field comparison) | **0** |
| `decor` exp | 3 |
| `decor` sigmoid | 14 |

So a narrow build degrades Decor's *fixed-point arithmetic*, not its
comparison. At `MPC_INT_SIZE = 64` the `relu` column is genuinely as secure as
it looks; the `exp`/`sigmoid`/`tanh` columns are not, because each call spends
`V` three to fourteen times over at 2^-10 apiece.

*The concrete consequence visible in this benchmark* is truncation. At
`world_size == 2` CrypTen truncates locally —
`share.div_(y, rounding_mode="trunc")` in
`crypten/mpc/primitives/arithmetic.py` — the SecureML-style probabilistic
truncation, correct except with probability growing as the truncated value
approaches the ring bound. A 64-bit ring therefore also caps usable dynamic
range, and 16 fractional bits is exactly what shows up as CrypTen's 1.5e-05
`relu` error against Sequre's and MPyC's 1e-10.

So: 64 bits is thin rather than broken, and the three frameworks are three
points in a (width, precision, statistical security, speed) space. No single
row of any table equalises them, which is why share width is now in every
column label.

**Where the rest of Sequre's marginal cost goes** is only partly identified.
Two hypotheses are ruled out by measurement:

- *Not PyTorch's thread parallelism.* Measuring CPU time against wall time
  inside CrypTen's timed region (`getrusage`, worker rank 1) gives
  `cores_busy` of 0.82–0.96 at n = 1024 and n = 8192 — under one core's worth,
  at the default 4 threads, before any pinning. Setting `OMP_NUM_THREADS=1`
  likewise moves n = 8192 by less than run-to-run noise. `taskset` would have
  been the direct check but is Linux-only and macOS has no working CPU-affinity
  equivalent; the CPU-time ratio answers the same question more strongly, since
  it shows the parallelism was never used rather than that the code survives
  without it.
- *Not the plaintext transcendental evaluation, and not bytes on the wire.*
  Sequre/Fourier evaluates ~30 `sin`/`cos` calls per element and sends 5x fewer
  bytes than Decor, yet runs at comparable or slower wall-clock. Neither term
  can be dominant.

### Why CrypTen wins at large n

**CrypTen's protocol.** Two-party additive secret sharing over Z_2^64, Beaver
triples supplied by a trusted first party (`provider=TFP`, `protocol=beaver`).
Its `exp` is the limit approximation `(1 + x/2^8)^(2^8)` — one public division
plus 8 squarings — which measures as exactly **8 rounds and 128 B per element**.
The decisive detail is that at `world_size == 2` CrypTen's fixed-point
truncation is *local*: `share.div_(y, rounding_mode="trunc")`, zero rounds,
zero bytes. All 8 of its truncations are free. It buys that with the
probabilistic-correctness and dynamic-range costs described above.

**Why Decor's exp is not faster, even though it needs no division.** The
exponential itself is not the cost. Splitting `via_decor exp` at n = 8192:

| piece | time | rounds | B/elem |
|---|---|---|---|
| beaver partition (reveal x − r) | 0.42 ms | 1 | 24 |
| `decor_partition` (mask + plaintext eval) | 3.48 ms | 2 | 48 |
| `fp.trunc` | 1.45 ms | 1 | 24 |
| **`decor_gt` (wraparound select)** | **17.91 ms** | **7** | **112** |
| **total** | **21.3 ms** | **13** | **256** |

Decor's actual exponential — partition plus combiner, no selection — runs in
**4.02 ms against CrypTen's 4.11 ms**, i.e. parity at the largest size. The
remaining ~83% is a comparison that has nothing to do with `exp`: the random
mask pushes `x − r` outside the interval about half the time, so `decor_eval`
evaluates both the in-range and the wrapped-around point and obliviously
selects between them. That select is a `decor_gt`, and on a 33-bit reduced ring
its prefix-carry costs 7 of the 13 rounds.

It is not optional — dropping it leaves **4094 of 8192 elements wrong**
(max error 7.5e+03 against 1.4e-07). But it is the whole gap: a cheaper
wraparound correction would put Decor's `exp` ahead of CrypTen at every size,
not behind it at large ones. Two directions, neither tried here: pick the mask
so wraparound cannot occur (trading statistical security for rounds), or
exploit that for `exp` the two branches differ only by the *public* factor
`e^(-period)`, so the shifted branch need not be recomputed from scratch.

The profile below explains the constant factor *inside* those rounds; the round
count above explains the structure.

Profiling a Sequre party in steady state (macOS `sample`, 6817 main-thread
samples while looping `via_decor exp` at n = 8192) gives the answer, and it is
not a cryptographic one:

| self time | where | what it is |
|---|---|---|
| ~27% | unsymbolised JIT code | the actual protocol arithmetic |
| **25%** | `__sendto` | blocking in the socket send syscall |
| **~21%** | Boehm GC + allocator | `GC_malloc_kind`, `GC_mark_from`, `GC_build_fl`, `GC_allochblk_nth`, `pthread_getspecific` (GC thread-local lookup), `_xzm_*` |
| ~6% | `_platform_memset` / `memmove` | serialisation buffers |
| ~5% | GMP `__gmpz_mod` | of which ~43% is `__gmpz_realloc` → `malloc` |
| ~2% | `aes_v8_ctr32_encrypt_blocks` | the secure PRG |

**Roughly half of Sequre's time at large n is memory management and synchronous
socket I/O, not cryptography.** The arithmetic itself is fast: a local
elementwise `mul_mod` over 8192 192-bit shares takes 0.029 ms — 3.5 ns per
element, about 0.1% of the 24 ms call. CrypTen, by contrast, works in place on
preallocated reference-counted `torch` tensors, batches its sends through gloo,
and has no garbage collector to run at all. That is the whole of its
per-element advantage.

The same profile independently re-confirms the threading result: every OpenMP
worker thread sits in `__kmp_fork_barrier` for the entire sample. Nothing is
parallel.

Truncation is the most expensive communicating primitive by a wide margin —
`fp.trunc` costs several times a `reveal` moving the *same* 192 KiB in the
*same* single round — and it is the term that scales with share width, which is
why the 64-bit build is ~4.6x cheaper there.

Two caveats on this analysis. Per-primitive micro-timings are noisy across runs
(`fp.trunc` measured anywhere from 1.08 to 2.63 ms), so the profile is the
reliable artifact and the primitive table is indicative only; the parties must
be re-synchronised with `sync_parties()` before each timed region or drift gets
charged to whichever operation a lagging party happens to be inside. And the
obvious first fix does not work: `GC_INITIAL_HEAP_SIZE=4G` made `via_decor exp`
*worse*, 52.6 ms against 22.3 ms, so the GC cost is not simply collection
frequency.

Share word size *is* a large, measured contributor, so the suite now runs
Sequre at both 128 and 192 bits and reports them as separate columns
(`benchmarks/activations/sequre/run_widths.sh`). At n = 8192:

| function | 192-bit | 128-bit | |
|---|---|---|---|
| exp | 21.7 ms | 10.4 ms | 2.1x |
| sigmoid | 56.8 ms | 40.8 ms | 1.4x |
| tanh | 56.2 ms | 43.4 ms | 1.3x |
| relu | 15.0 ms | 8.0 ms | 1.9x |

At 128 bits `relu` reaches parity with CrypTen at n = 8192 (8.0 ms vs 7.8 ms)
and the gap on the others narrows from 3–5x to roughly 2.3–2.5x. The remainder
is not accounted for here.

**The 128-bit build is not a free speedup.** `MPC_INT_SIZE` also fixes
`MPC_NBIT_V`, Sequre's statistical security parameter, via the budget asserted
in [`stdlib/sequre/constants.codon`](../../stdlib/sequre/constants.codon):

```
assert MPC_NBIT_K + MPC_NBIT_F + MPC_NBIT_V + 2 < MPC_MODULUS_BITS
```

| build | modulus | K | F | **V** |
|---|---|---|---|---|
| `MPC_INT_SIZE = 192` | 174 bits | 64 | 32 | **64** |
| `MPC_INT_SIZE = 128` | 127 bits | 64 | 32 | **28** |

Accuracy really is unchanged — `K` and `F` are identical, and the measured
errors match to three digits. But the masking slack drops from 2^-64 to 2^-28,
below the κ = 40 that is conventional for statistical security in MPC. The
128-bit column is the right one for isolating *why* Sequre's marginal cost is
what it is; the 192-bit column is the right one for a security-matched claim.
The repository default of 192 is left untouched.

### Matching CrypTen's 64-bit width

Since only Sequre is tunable, the direct way to remove width from the
comparison is to bring Sequre down to CrypTen's ring. `MPC_INT_SIZE = 64` was
added for exactly this (`./sequre/run_widths.sh 64 128 192`): a 2^61 − 1 field
with K = 32, F = 16 — the same 16 fractional bits CrypTen has — and V = 10.

At n = 1024, width-matched:

| function | Sequre+Decor (64b) | CrypTen (64b) | speed | accuracy |
|---|---|---|---|---|
| exp | 1.29 ms, 9.1e-03 | 3.15 ms, 4.2e-01 | **2.4x faster** | 46x better |
| sigmoid | 4.94 ms, 5.8e-03 | 12.56 ms, 2.4e-03 | **2.5x faster** | 2.4x *worse* |
| tanh | 8.99 ms, 7.3e-02 | 12.49 ms, 3.6e-03 | **1.4x faster** | 20x *worse* |
| relu | 0.96 ms, 1.5e-05 | 4.90 ms, 1.5e-05 | **5.1x faster** | identical |

At n = 8192, width-matched: `relu` 5.86 ms vs 7.80 ms (1.3x faster), but
`exp`/`sigmoid`/`tanh` at 0.5–0.6x — CrypTen still wins the large-batch
transcendental case. So width accounted for roughly half of the earlier
n = 8192 deficit (0.2–0.3x became 0.5–0.6x); the rest is still unidentified.

**The important result is the accuracy row.** ReLU lands on *exactly* CrypTen's
1.5e-05 — both are precision-bound at 2^-16, not algorithm-bound. And
`sigmoid`/`tanh` become *worse* than CrypTen, because Decor's Newton-iteration
division needs fractional headroom that 16 bits does not provide. Decor's
accuracy advantage is therefore not free-standing: it is purchased with
fixed-point precision, which requires modulus width, which costs speed. Strip
the width and much of the advantage goes with it.

The 64-bit build is a diagnostic, not a configuration anyone should ship —
though the reason is narrower than "V = 10". Comparison is unaffected (it
spends no `V`, see above), so the `relu` row is a fair like-for-like result.
What is not shippable is the fixed-point arithmetic underneath
`exp`/`sigmoid`/`tanh`: 3 to 14 truncations per call at 2^-10 each, `tanh`
error of 7.9e-02 at n = 8192, and the Fourier path numerically out of range
entirely (errors of 1e+01 to 9e+03 — those cells in `REPORT.md` are a range
failure, not a fit quality).

**So: does Sequre+Decor outperform CrypTen?**

- *At Sequre's own 192-bit configuration:* on accuracy, unambiguously and at
  every size — 10^4 to 10^6 better. On speed, decisively below ~1000 elements
  per call, and behind above it.
- *At matched 64-bit width:* faster than CrypTen on every function at
  n = 1024 and on `relu` at n = 8192, but no longer more accurate — identical
  on `relu`, worse on `sigmoid`/`tanh`.
- *The honest summary:* Decor buys exactness with precision, and precision with
  width, and width with throughput. Sequre wins the accuracy comparison at any
  width it can actually be deployed at, and wins the latency comparison at
  every width; CrypTen wins large-batch throughput at every width, by less than
  the raw numbers first suggested.

### Online cost, with the dealer detached

Wrapping the benchmark in `mpc.detach_dealer()` — the way the Decor paper's own
benchmarks do it, around the whole block rather than individual operations —
buffers the dealer's sends so offline randomness generation leaves the critical
path. This matters because the profile above found the dealer's contribution is
real: truncation, the term that dominates, is where dealer randomness is spent.

At n = 8192 the dealer accounts for a substantial slice:

| | attached | detached | |
|---|---|---|---|
| 192-bit exp | 21.7 ms | 18.5 ms | −15% |
| 192-bit relu | 15.0 ms | 9.9 ms | −34% |
| 64-bit exp | 10.4 ms | 6.0 ms | −42% |
| 64-bit relu | 8.0 ms | 4.3 ms | −46% |

Detached *and* width-matched against CrypTen — the closest thing to a
like-for-like online comparison this benchmark can construct:

| function | n = 1024 | n = 8192 |
|---|---|---|
| exp | **3.1x faster** | 0.7x |
| sigmoid | **3.7x faster** | 0.8x |
| tanh | **3.6x faster** | 0.9x |
| relu | **7.3x faster** | **1.8x faster** |

So with the dealer off the critical path and widths equalised, Sequre+Decor
leads everywhere at n = 1024, leads on `relu` at every size, and closes the
large-batch transcendental gap from 0.2–0.3x to 0.7–0.9x. What remains is the
allocator and socket overhead identified in the profile, not protocol cost.

Two methodology notes. Entering a detached block costs the receiving party a
one-time stall while the dealer runs ahead — ~119 ms against a ~17 ms steady
state — so the runner pre-warms inside the block to keep that out of every
timed region; without it the first cell would swallow the whole stall and every
later cell would look artificially fast. And results were verified identical
inside and outside the block (1.317e-07 vs 1.325e-07 max abs error on `exp`),
because a buffered dealer that silently starved the compute parties would
produce fast, wrong answers.

**Communication.** Decor sends more than CrypTen for `exp`/`sigmoid`/`tanh`
(2.0 MiB vs 1.0 MiB for `exp` at n = 8192; 8.6 MiB vs 5.9 MiB for `sigmoid`)
and considerably less for `relu` (1.4 MiB vs 3.6 MiB). Against Sequre's own
full-width comparison, Decor's reduced-ring `relu` sends 7.6x less and runs
~23x faster. MPyC sends one to two orders of magnitude more than either.

**Sequre/Fourier** is the cheapest column in bytes by a wide margin — its
communication is independent of the series degree, which is the point of
evaluating the series through Decor's trigonometric protocol — but at these
interval widths the approximation is not usable. That column is here to show
what the exact protocol buys, not as a serious competitor.

## Caveats

These matter for reading the numbers honestly.

- **Threat models are not identical.** Sequre runs 3 parties with an offline
  dealer; CrypTen runs 2 online parties plus a trusted first party; MPyC runs 3
  parties with Shamir sharing and no dealer at all. Decor's advantage depends on
  a dealer being available. This is a real assumption, not an implementation
  detail.
- **Share width and fixed-point precision differ and are not freely tunable.**
  Sequre uses 192-bit shares with 32 fractional bits, MPyC a 128-bit field with
  32, CrypTen a 64-bit ring with 16. CrypTen is doing materially less work per
  element than either, and `relu`'s 1.5e-05 error is that precision choice
  rather than its algorithm. Its `exp`/`sigmoid`/`tanh` error is far too large
  to explain that way.
- **Two measurement modes are reported.** In `end_to_end`, correlated
  randomness is generated inside the timed call for both Sequre and CrypTen, so
  neither gets credit for an idealised offline phase. In `online`, Sequre's
  dealer is detached with `mpc.detach_dealer()`; CrypTen and MPyC have no
  separable offline phase, so they repeat their end-to-end numbers there. That
  table is Sequre-favourable by construction and is labelled as such.
- **Localhost only.** Network latency is near zero, which favours protocols
  with more rounds and penalises none of them. On a WAN the communication table
  becomes the dominant term and the ranking will shift.
- **Single machine, no pinning.** Thread count was checked and does not explain
  the large-`n` gap (see above), but the three frameworks still share one host,
  so each party contends with the others for cores and memory bandwidth.
- **Share widths cannot be equalised.** CrypTen's ring is hardwired to
  `torch.int64` and cannot be widened; MPyC's field follows from its `SecFxp`
  length. Only Sequre is tunable, so the suite reports it at both 128 and 192
  bits rather than pretending one number is like-for-like. The 128-bit build is
  the speed-matched comparison and the 192-bit build the security-matched one;
  neither is "the" answer.
- **CrypTen 0.4.1 is from 2022** and is the last release. It is pinned against
  torch 2.4.1 because newer torch removed an internal module it imports. Newer
  hardware paths in PyTorch may therefore be unavailable to it.

## Layout

```
spec.py                        inputs, intervals, sizes, reps -- the single source of truth
common.py                      result record, error metrics, JSONL writer
report.py                      builds REPORT.md from results/*.jsonl
sequre/bench_activations.codon Sequre runner (Decor + Fourier paths)
sequre/run_widths.sh           runs Sequre at 128 and 192 bits, restoring settings.codon
crypten/bench_activations.py   CrypTen runner
mpyc/bench_activations.py      MPyC runner
setup.sh                       creates .venv for the two Python runners
run_all.sh                     runs everything, writes REPORT.md
results/                       one JSONL per framework, one JSON object per row
```

`spec.py` is authoritative. The Codon runner cannot import Python, so it
mirrors the constants and echoes them into its output header; `report.py`
refuses to build a table if any runner's header disagrees with `spec.py`.

## Running it

```bash
# one-time: Python environment for CrypTen and MPyC
./benchmarks/activations/setup.sh

# everything, then the report
./benchmarks/activations/run_all.sh

# or a subset
./benchmarks/activations/run_all.sh sequre crypten
```

Individually:

```bash
# from the repo root -- -release is essential, the default build carries backtraces
sequre run -release benchmarks/activations/sequre/bench_activations.codon --local

# both share widths as separate report columns; restores settings.codon on any exit
./benchmarks/activations/sequre/run_widths.sh

cd benchmarks/activations/crypten && ../.venv/bin/python bench_activations.py
cd benchmarks/activations/mpyc    && ../.venv/bin/python bench_activations.py -M3

cd benchmarks/activations && .venv/bin/python report.py --out REPORT.md
```

`--use-ring` is how Decor is meant to be run — its reduced-ring comparison
needs a power-of-two modulus. The runner also forces the ring internally with
`mpc.default_modulus`, so the flag does not change protocol timings, but it
puts input sharing on the ring too and saves ~9% of the bytes. Both scripts
pass it.

MPyC skips sizes above `--max-n` (default 1024) because a pure-Python runtime
at n = 8192 takes tens of seconds per call; skipped cells are recorded
explicitly rather than dropped. Pass `--max-n 8192` to fill them in.
