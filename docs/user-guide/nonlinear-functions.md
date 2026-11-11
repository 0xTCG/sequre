# Nonlinear Functions

Sequre evaluates nonlinear functions (exponentials, sigmoids, trigonometric functions, logarithms, divisions, softmax)
on secret data through one API, `sequre.stdlib.nonlinear`. The same call works on plaintext values and on every secure
type, and returns the same thing on each for the method you pick. By default it uses **Decor**, which computes the
function itself rather than a polynomial approximation of it. sigmoid and tanh default to `"auto"`: Decor on ring
shares, and on field shares, where Decor's division is slow, a Chebyshev interpolant within Decor's error bound.

---

## Decor in one paragraph

Beaver triples let parties multiply because a dealer precomputes randomness that fits an identity:
$xy = (x - a)(y - b) + (x - a)b + (y - b)a + ab$. Decor (*Delegated Computation on Randomness*) generalizes this.
Whenever a function decomposes as $f(u + v) = G(g(u), h(v))$ with $G$ cheap in MPC, the dealer secret-shares $h(r)$ for
a fresh random mask $r$. The parties open $x - r$, evaluate $g(x - r)$ in the clear, and combine. The nonlinear work
happens on plaintext, so the result carries only the errors of the fixed-point encodings, in a constant number of
rounds.

- $e^{u + v} = e^u e^v$ gives exp; sinh and cosh use their own addition theorems.
- $P(u + v) = \sum_j P^{(j)}(u)\, v^j / j!$ gives polynomials (and the Taylor method).
- $\sin(u + v) = \sin u \cos v + \cos u \sin v$ gives sin and cos for any $x$ (no interval); tan and cot add a division.
- sigmoid and tanh are ratios of exponential terms, followed by one division.

Decor is described in Smajlović et al., IEEE S&P 2026.

---

## Quick start

```python
from sequre.stdlib.nonlinear import sigmoid, exp, sin, log, softmax

y = sigmoid(mpc, x)                                   # auto on (-10, 10) at F = 32: Decor on ring shares, Chebyshev on field shares
y = sigmoid(mpc, x, interval=(-6.0, 6.0), clip=True)  # a narrower interval; inputs outside are clipped first
y = sin(mpc, x)                                       # Decor: any x, no interval
y = exp(mpc, x, method="chebyshev", degree=24)        # the degree-24 Chebyshev interpolant on (-8, 8)
y = log(mpc, x)                                       # Chebyshev by default (log has no Decor decomposition)
p = softmax(mpc, scores)                              # row-wise, max-shifted
```

`x` can be a `float`, a (nested) `List[float]`, an `ndarray`, a `Sharetensor`, an `MPP`, an `MPA`, an `MPU` or a
`Ciphertensor`. Every party, the dealer (CP0) included, makes the same call.

---

## Functions and methods

Every function has the signature `f(mpc, x, method="...", interval=(0, 0), degree=0, clip=False, tolerance=...)`.
`method` is a compile-time string; `interval=(0, 0)` and `degree=0` mean the method's defaults.

| Method | Computes | Functions | Default for |
|---|---|---|---|
| `"decor"` | the function itself, up to the errors of its fixed-point encodings | sin, cos, tan, cot, exp, sinh, cosh, sigmoid, tanh, inv, softmax, polynomial | all of these but sigmoid and tanh |
| `"auto"` | the function within Decor's error bound: Decor on ring shares; on field shares the Chebyshev interpolant of the smallest degree whose truncation (coefficients within double-precision noise count as zero) stays within half of Decor's error bound on the interval (`auto_degree`: 35 for sigmoid on (−10, 10) and 39 for tanh on (−5, 5) at F = 32, 99 and 107 on (−15, 15) and (−8, 8) at F = 64), plus Clenshaw's rounding; at F = 64 on narrow intervals, where Decor's bound falls below what double-precision coefficients reach, within $2^{-44}$ instead | sigmoid, tanh | sigmoid, tanh |
| `"chebyshev"` | the degree-`degree` Chebyshev interpolant on the interval (Clenshaw's recurrence) | the twelve named functions, softmax (its exp) and `approx` | log, erf |
| `"fourier"` | the degree-`degree` Fourier series on the interval (through Decor's trigonometric decomposition); a non-periodic target rings at the interval's ends (Gibbs: about 9% of the jump) | the twelve named functions and `approx` | |
| `"taylor"` | the degree-`degree` Taylor polynomial around the interval's centre (through Decor's polynomial decomposition); accurate only well inside the series' radius around the centre, so it takes no default interval | the twelve named functions | |

The twelve named functions: exp, sigmoid, tanh, sin, cos, tan, cot, sinh, cosh, log, inv, erf. Chebyshev and Fourier
share the default intervals below, which suit Chebyshev; for Fourier pass an interval that suits the target (its
series of inv on (0.5, 8) gives 0.97 at 0.5: 1/x is not periodic there). Taylor takes no default interval and raises
`ValueError` without one: a Taylor polynomial is accurate only near its centre (the degree-8 one of sigmoid on
(−10, 10) gives −1918 at 10; sigmoid's series converges only within π of the centre, tanh's within π/2). The
admissibility checks bound the encodings, not the distance from f.

Also in the module:

- `approx(mpc, x, op, interval, method="chebyshev", degree=0, clip=False)`: any plaintext function `op` of one float,
  by its Chebyshev interpolant or Fourier series.
- `polynomial(mpc, x, coeffs, interval)`: $\sum_k c_k x^k$ by Decor's polynomial decomposition, in constant rounds.
- `softmax(mpc, x, method="decor", shift="max", width=16.0, offset=0.0, degree=0)`: the softmax of each row (the whole
  vector for 1-D input). The exponent is shifted by the row maximum (`shift="max"`) or by a public offset
  (`shift="fixed"`, no comparisons), and evaluated on $(-\text{width}, 0]$.
- `default_interval(func, method)`: the interval used when you pass none (Decor's for `"auto"`; Taylor has none).
- `auto_degree(func, interval=(0, 0))`: the degree of the Chebyshev interpolant `"auto"` evaluates on field shares.

An unsupported (function, method) pair, such as Decor's log, is a compile error.

---

## Intervals, clipping and admissibility

**Intervals.** Decor's sin, cos, tan and cot take any input (tan and cot away from the poles), and Decor's inv any
admissible divisor ($2^{-(F-12)} \le \lvert x \rvert \le 2^{K-F-2}$). The other functions are evaluated on an interval
$[a, b]$:

| Function | Decor and auto default, F = 32 | Decor and auto default, F = 64 | Chebyshev and Fourier default |
|---|---|---|---|
| exp | (−8, 8) | (−20, 20) | (−8, 8) |
| sigmoid | (−10, 10) | (−15, 15) | (−10, 10) |
| tanh | (−5, 5) | (−8, 8) | (−5, 5) |
| sinh, cosh | (−5, 5) | (−8, 8) | (−5, 5) |
| sin, cos | any input | any input | (−π, π) |
| tan | any input (away from the poles) | the same | (−1.2, 1.2) |
| cot | any input (away from the poles) | the same | (0.4, 2.7) |
| inv | any admissible divisor | the same | (0.5, 8) |
| log | — | — | (0.5, 8) |
| erf | — | — | (−4, 4) |

**Inputs outside the interval** give different results by method:

- Decor's bounded functions and the Taylor method give $f(x - kp)$ on the secure types, for the decomposition's period
  $p \approx b - a$ (for example, sigmoid(10.5) on (−10, 10) gives $7.5 \times 10^{-5}$). Plaintext types and MPP
  partitions in the clear give $f(x)$.
- Chebyshev extrapolates its polynomial: its interpolant of erf on (−4, 4) gives −43 at 5.
- Fourier repeats with period $b - a$.
- `"auto"` behaves as the method it runs: as Decor on ring shares, as Chebyshev on field shares.

`clip=True` clips the input to the interval first, on every type, and makes the results agree. It costs one comparison
and one product: 9 rounds on the ring and 12 on the field.

**Admissibility.** Before any communication, every party runs the same public checks on the function, interval,
degree and tolerance. A failed check raises `ValueError`. The checks:

- Decor (and `"auto"`, on both moduli): every value the protocol encodes must fit the fixed-point range; sigmoid's and tanh's denominators must stay
  above the division's smallest divisor; and the worst-case error bound `decor_error_bound(func, interval)` must stay
  within `tolerance` times $\max(1, \max \lvert f \rvert)$ on the interval. The default tolerance is
  `DECOR_TOLERANCE` $= 2^{-(F/2 - 4)}$ ($2^{-12}$ at F = 32).
- Chebyshev: Clenshaw's intermediates, $2 \sum_k (k + 1)|c_k|$, must stay below the fixed-point range.
- Fourier: the sum of the coefficients must stay in range.

The widest admitted symmetric intervals at F = 32 are about $b < 14.3$ for exp, $b < 11.1$ for sinh and cosh,
$b \lesssim 12.3$ for sigmoid and $b \lesssim 6.5$ for tanh. See [Troubleshooting](troubleshooting.md) for the
messages.

---

## Precision

Decor's error comes from the fixed-point encodings (units of $2^{-F}$) and from the plaintext evaluations of its
terms in double precision ($2^{-52}$ of their size). It depends on the build's fractional bits F and on the interval's
width, not on a degree. At F = 32 the encodings dominate; at F = 64 the double-precision evaluations do, so the
256-bit build gains about 4 to 6 decimal digits, not 9.

| Build (`MPC_INT_SIZE`) | F | Field / ring bits |
|---|---|---|
| 128 | 32 | 127 |
| 192 (default) | 32 | 174 |
| 256 | 64 | 251 |

Errors at F = 32 (default intervals; measured: the largest absolute error against double precision over 400 to 1,000
random points, on the ring and on the field) and the worst-case bounds the admissibility check uses at both precisions
(`decor_error_bound`, `trig_error_bound`; F = 64 on its own default intervals):

| Function | F = 32: interval | measured | bound | F = 64: interval | bound |
|---|---|---|---|---|---|
| sin, cos | any $\lvert x \rvert \le 100$ | $2 \times 10^{-9}$ to $1.2 \times 10^{-8}$ | $3.9 \times 10^{-8}$ | any $\lvert x \rvert \le 100$ | $2.5 \times 10^{-14}$ |
| exp | (−8, 8) | $3.5 \times 10^{-5}$ | $4.9 \times 10^{-5}$ | (−20, 20) | $1.7 \times 10^{-5}$ |
| sigmoid | (−10, 10) | $5.3 \times 10^{-6}$ | $2.6 \times 10^{-5}$ | (−15, 15) | $1.5 \times 10^{-9}$ |
| tanh | (−5, 5) | $4.3 \times 10^{-6}$ | $1.3 \times 10^{-5}$ | (−8, 8) | $2.0 \times 10^{-9}$ |
| sinh, cosh | (−5, 5) | $1.6 \times 10^{-6}$ | $5.3 \times 10^{-6}$ | (−8, 8) | $2.9 \times 10^{-6}$ |

The errors are absolute and about the same over the whole interval: the dealer's terms reach the size of the
function's largest values whatever $x$ is. exp's error is therefore relative to $e^b$, the interval's maximum, not to
$e^x$: $e^{-7.9}$ on (−8, 8) carries the same $3.4 \times 10^{-5}$ as $e^{7.9}$, 9% of its value. Choose the
narrowest interval that holds the inputs. sinh and cosh lose more on wide intervals, because the addition theorem's two
terms grow to about $\cosh(2b)\cosh(b)$ before they cancel. At F = 64 the default intervals are wider; on the F = 32
intervals the 256-bit bounds are $4 \times 10^{-11}$ (exp), $10^{-11}$ (sigmoid), $5 \times 10^{-12}$ (tanh) and
$4 \times 10^{-10}$ (sinh, cosh). These F = 64 figures are bounds; the 256-bit build passes the test suite's Decor,
bitwise, activation and nonlinear groups, whose tolerances come from them.

A Chebyshev interpolant adds its interpolation error: degree 20 gives about $2 \times 10^{-4}$ for erf on (−4, 4), and
it cannot follow a function that oscillates many times on the interval. Taylor and Fourier add theirs, which depends
on the interval as described above.

---

## Cost

Rounds and time per call on 1,000 elements, 3 parties on one machine (release build, default 192-bit build; the
benchmark suite's `decor.fn.*`, `cheby.*` and `nl.*` cases):

| Function | Rounds, ring | Rounds, field | Time, ring | Time, field |
|---|---|---|---|---|
| sin, cos | 3 | 3 | 1.0–1.1 ms | 1.2–1.3 ms |
| exp (−8, 8) | 10 | 10 | 2.1 ms | 2.3 ms |
| sinh, cosh (−5, 5) | 10 | 10 | 2.6–2.9 ms | 2.6–2.9 ms |
| polynomial, degree 3 | 10 | 10 | 2.7 ms | 3.1 ms |
| sigmoid, tanh (Decor) | 51 | 121 | 12 ms | 427 ms |
| sigmoid, tanh (`"auto"`, the default: Decor on the ring, degrees 35 and 39 on the field) | 51 | 72, 80 | 12 ms | 14.5 ms, 16.3 ms |
| tan, cot | 54 | 113 | 13–19 ms | 418–421 ms |
| Chebyshev, degree $d$ (Clenshaw) | $2d + 2$ | $2d + 2$ | 3.9 ms ($d = 10$) | 4.7 ms ($d = 10$) |
| Chebyshev sigmoid matching Decor's measured error (degree 40 on (−10, 10)) | 82 | 82 | 13.7 ms | 16.7 ms |
| Chebyshev tanh matching Decor's measured error (degree 44 on (−5, 5)) | 90 | 90 | 15.2 ms | 18.1 ms |

On one machine, round counts barely show; across a network with latency, they dominate. Accuracy differs as well:
Decor's sigmoid on (−10, 10) is within $5.3 \times 10^{-6}$ of the function, the default degree-20 interpolant within
$1.8 \times 10^{-3}$, the degree-10 one on (−8, 8) within $1.7 \times 10^{-2}$. At equal accuracy (degree 40, $3.9 \times
10^{-6}$; tanh degree 44) Chebyshev costs about 4% of Decor's time on the field, and Decor 78–86% of Chebyshev's on the
ring. `"auto"`'s field interpolants match Decor's error bound rather than its measured error (degrees 35 and 39; measured
$8.8 \times 10^{-6}$ and $5.1 \times 10^{-6}$, within the bounds $2.6 \times 10^{-5}$ and $1.3 \times 10^{-5}$), at about 3.5% of
Decor's field time. The tan/cot and equal-accuracy rows come from one filtered run of the suite, the `"auto"` row from
the final full runs (the rest from the full suite).

---

## Ring or field

Sequre runs MPC on a prime field by default, or on the ring $\mathbb{Z}_{2^L}$ with `--use-ring`. Every nonlinear
function works on both and gives the same results within the errors above.

- exp, sinh, cosh, sin, cos and polynomials cost the same on both.
- The dividing functions (sigmoid, tanh, tan, cot, inv, softmax) use Decor's division on the ring and Sequre's field
  division on the field, which takes about 35 times longer. sigmoid's and tanh's default, `"auto"`, therefore runs a
  Chebyshev interpolant within Decor's error bound on field shares (72 rounds for sigmoid and 80 for tanh at F = 32,
  against Decor's 121).
- Comparisons (and so `clip=True`, the softmax maximum, and the comparison-based activations) are about 35 times faster
  on the ring.

---

## Multiparty types

| Type | How it is evaluated |
|---|---|
| `float`, lists, `ndarray` | in the clear, in double precision |
| `Sharetensor` | the method's protocol |
| public `Sharetensor` (after `publish`) | in the clear at every compute party, like an `ndarray`; the result stays public |
| `MPP` | locally, when every compute party holds its partition in the clear (one round to check); otherwise through MPC (`via_mpc`) |
| `MPA` | through MPC (additive shares: no local shortcut) |
| `MPU` | as its MPP or MPA |
| `Ciphertensor` | through MPC, once per compute party's ciphertexts |

Chebyshev without `clip` runs Clenshaw's recurrence on the input's own arithmetic for every type but plain lists and
public Sharetensors (for example, homomorphically on encrypted MPP partitions). `"auto"` evaluates MPP, MPA, MPU and
Ciphertensor through MPC (MPP partitions in the clear locally) on the default MPC modulus: Decor with `--use-ring`,
the interpolant on the field.

---

## In machine learning

The neural-network activations, `Dense` layers and `LogReg` take a `method` (`"decor"` or `"chebyshev"`) and pass it
to this API. The activations and `Dense` default to Chebyshev, Sequre's method before Decor (pass `method="decor"` for
activations that are exact up to the fixed-point encodings: sin and cos for any input). Binary `LogReg` picks by the
shares' modulus at each call (`LogReg.method_for`): Decor on ring Sharetensors (where Decor admits the interval),
Chebyshev on field Sharetensors, in the clear and on the multiparty types; the GWAS application does the same. Multinomial `LogReg` defaults to Decor,
which is faster for the softmax on both moduli. The cross-entropy losses take no method: their log
is a Chebyshev start refined through Decor's exp, and their derivatives divide with Decor's division on the ring and
the field's division on the field. See [Secure ML](../api/learn.md).

The choice matters most on the field, where Decor's sigmoid and tanh divide with the field's division. Training time
on Sharetensors, 3 parties on one machine (the benchmark suite's `e2e.*` cases; each row's four cells from the same
runs):

| Model | Decor, field | Chebyshev, field | Decor, ring | Chebyshev, ring |
|---|---|---|---|---|
| Neural network, 64 x 8 (tanh 16 + sigmoid, binary cross-entropy, 2 epochs) | 1.28 s | 0.40 s | 0.052 s | 0.045 s |
| GWAS logistic association, 64 individuals x 3 SNPs (a binary logistic regression per SNP, 10 epochs) | 1.82 s | 0.53 s | 0.14 s | 0.16 s |
| Multinomial logistic regression, 256 x 8, 3 classes (5 epochs) | 1.12 s | 1.16 s | 0.065 s | 0.135 s |

On the field Chebyshev is the cheaper method for the networks and the binary regression, by about 3.2 to 3.4 times. On
the ring the cheaper method depends on the model: Decor takes 15% more for the network, 11% less for the binary
regression and 52% less for the softmax regression, and it is the exact one. Hence the defaults: Chebyshev for the
network on both moduli, Decor for the softmax regression on both, and each modulus's cheaper method for the binary
regression and the GWAS application.
