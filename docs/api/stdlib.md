# Secure Standard Library

_Defined in `stdlib/sequre/stdlib/`_

Sequre provides a standard library of secure functions that operate on `Sharetensor`, `Ciphertensor`, and multiparty types (`MPU`, `MPP`, `MPA`). These functions are building blocks for secure applications.

---

## Nonlinear functions

_Defined in `stdlib/sequre/stdlib/nonlinear.codon`_

One API for nonlinear functions over plaintext and every secure type. The guide
[Nonlinear Functions](../user-guide/nonlinear-functions.md) explains the methods, intervals, precision and cost.

| Function | Signature | Default method |
|---|---|---|
| `exp`, `sinh`, `cosh` | `f(mpc, x, method="decor", interval=(0, 0), degree=0, clip=False, tolerance=DECOR_TOLERANCE)` | Decor on its default interval |
| `sigmoid`, `tanh` | the same, `method="auto"` | on Decor's default interval: Decor on ring shares, a Chebyshev interpolant within Decor's error bound (at least $2^{-44}$) on field shares |
| `sin`, `cos`, `tan`, `cot` | the same | Decor, any input (radians) |
| `inv` | the same | Decor's division (any sign; $2^{-(F-12)} \le \lvert x \rvert \le 2^{K-F-2}$) |
| `log`, `erf` | the same, `method="chebyshev"` | Chebyshev, degree 20 |
| `softmax` | `softmax(mpc, x, method="decor", shift="max", width=16.0, offset=0.0, degree=0)` | each row; Decor's exp |
| `polynomial` | `polynomial(mpc, x, coeffs, interval, method="decor", clip=False, tolerance=DECOR_TOLERANCE)` | Decor's polynomial decomposition |
| `approx` | `approx(mpc, x, op, interval, method="chebyshev", degree=0, clip=False)` | Chebyshev (or `"fourier"`) for any plaintext `op` |
| `default_interval` | `default_interval(func, method="decor")` | the interval used for `interval=(0, 0)` (Taylor has none) |
| `auto_degree` | `auto_degree(func, interval=(0, 0), tolerance=DECOR_TOLERANCE)` | the Chebyshev degree `"auto"` uses on field shares: the smallest whose truncation (coefficients within double-precision noise count as zero) stays within half of max(Decor's bound, `NONLINEAR_AUTO_FLOOR` $= 2^{-44}$); 35 for sigmoid and 39 for tanh on their default intervals at F = 32, 99 and 107 at F = 64 |

`method` is one of `"decor"`, `"auto"` (sigmoid and tanh), `"chebyshev"`, `"fourier"` and `"taylor"` (a compile-time
string). Each function checks its parameters at every party before any communication and raises `ValueError` when they
are not admissible. The checks bound the encodings, not the distance from f: Taylor is accurate only near the
interval's centre (it takes no default interval) and Fourier rings at the ends of a non-periodic target, so give them
an interval that suits the target (see the guide's method table).

```python
from sequre.stdlib.nonlinear import sigmoid, log

p = sigmoid(mpc, scores, clip=True)          # auto, (-10, 10) at F = 32, inputs clipped to it
l = log(mpc, x)                              # Chebyshev, degree 20 on (0.5, 8): an interpolant is poor near 0
```

### Decor package

_Defined in `stdlib/sequre/decor/`_

The protocols behind the `"decor"` method, for direct use on fixed-point `Sharetensor`s:

| Function | Description |
|---|---|
| `via_decor(mpc, x, func, interval=(0, 0), coeffs=[], clip=False, tolerance=DECOR_TOLERANCE)` | exp, sinh, cosh, sigmoid, tanh or a polynomial (`coeffs`, constant first) on an interval; sin, cos, tan, cot for any input |
| `via_decor_trig(mpc, x, func)` | sin, cos, tan or cot (radians, any input) |
| `via_decor_sincos(mpc, x, scale=1.0)`, `via_decor_sinhcosh(mpc, x, interval, tolerance=DECOR_TOLERANCE)` | both functions from one opening (`scale` multiplies the argument inside the protocol) |
| `decor_div(mpc, x, y, niter=DECOR_DIV_DEFAULT_NITER, sign_check=True)`, `decor_mul_inv(mpc, b, niter=DECOR_DIV_DEFAULT_NITER, sign_check=True)` | Decor's division on the ring (the sign test, the ring normalizer, Goldschmidt iterations, 4 at F = 32 and 5 at F = 64; `sign_check=False` for divisors known to be positive); the field's division on the field. Admissible divisors: $2^{-(F-12)} \le \lvert b \rvert \le 2^{K-F-2}$ |
| `decor_gt(mpc, x, y)`, `ring_gt(mpc, x, y, bits)` | $[x > y]$ from the sign of $x - y$ over the ring's low `bits` bits: valid while $\lvert x - y \rvert < 2^{\text{bits} - 1}$ raw units (`bits` = K + 1 covers any two fixed-point values). `decor_gt` uses F + 1 bits: only for $\lvert x - y \rvert < 1$ (Decor's internal select); beyond, it is wrong. On the field both are the field comparison (`bits` ignored) |
| `via_fourier(mpc, x, op, interval, degree, node_count=1000)` | the degree-`degree` Fourier series of a plaintext `op` through Decor's trigonometric decomposition |
| `via_taylor(mpc, x, func, interval, degree, evaluation="decor", tolerance=DECOR_TOLERANCE)` | the Taylor polynomial around the interval's centre, by Decor's polynomial decomposition (`"naive"`: secure powers, refused before any round when the powers leave the fixed-point range or the estimated error exceeds `tolerance`; it grows like $2^{-F} r^{\text{degree}}$ for half-widths $r > 1$) |
| `decor_error_bound(func, interval=(0, 0), coeffs=[])`, `trig_error_bound(x)` | the worst-case error the admissibility check uses |
| `modulo_float(mpc, x, period)` | `x` reduced modulo a public period share by share, up to a multiple of the period below $n_{cp}$ periods: for periodic functions of that period only (ring shares) |

The protocols refuse to run inside `mpc.delayed_truncation()`.

---

## Built-in functions

_Defined in `stdlib/sequre/stdlib/builtin.codon`_

### Element-wise operations

| Function | Signature | Description |
|---|---|---|
| `sign` | `sign(mpc, x)` | Secure sign: returns +1 or -1 per element |
| `abs` | `abs(mpc, x)` | Secure absolute value |
| `maximum` | `maximum(mpc, x, y)` | Secure element-wise maximum of two tensors |
| `minimum` | `minimum(mpc, x, y)` | Secure element-wise minimum of two tensors |
| `clip` | `clip(mpc, x, low, high)` | Clamp values to `[low, high]` range |
| `argmax` | `argmax(mpc, x)` | Returns `(index, value)` of the maximum element |
| `cov_max` | `cov_max(mpc, x)` | Covariance maximum: `maximum(x, x.T)` |

### Matrix operations

| Function | Signature | Description |
|---|---|---|
| `inv` | `inv(mpc, x)` | Secure matrix inverse (up to 3×3; uses closed-form determinant formulas) |

!!! note
    For multiparty types (`MPU`, `MPP`, `MPA`), `inv` automatically switches to MPC via `via_mpc`.

### Chebyshev approximations

Non-linear/transcendental functions (exp, log, sigmoid, …) cannot be computed directly on secret-shared or encrypted data. Sequre approximates them via **Chebyshev polynomial interpolation**: the input is scaled to $[-1, 1]$, a degree-$n$ Chebyshev expansion is evaluated using the [Clenshaw recurrence](https://en.wikipedia.org/wiki/Clenshaw_algorithm), and the result is returned as a secret-shared tensor.

Every Chebyshev function requires an **interval** — the expected range of the plaintext input.

| Function | Signature | Approximates |
|---|---|---|
| `chebyshev_exp` | `chebyshev_exp(mpc, x, interval)` | $e^x$ |
| `chebyshev_sigmoid` | `chebyshev_sigmoid(mpc, x, interval)` | $\frac{1}{1 + e^{-x}}$ |
| `chebyshev_log` | `chebyshev_log(mpc, x, interval)` | $\ln(x)$ |
| `chebyshev_mul_inv` | `chebyshev_mul_inv(mpc, x, interval)` | $\frac{1}{x}$ |
| `chebyshev_sqrt` | `chebyshev_sqrt(mpc, x, interval)` | $\sqrt{x}$ |
| `chebyshev_sqrt_inv` | `chebyshev_sqrt_inv(mpc, x, interval)` | $\frac{1}{\sqrt{x}}$ |

The polynomial degree is controlled by the `CHEBYSHEV_DEGREE` constant (default: **20**).

```python
from sequre.stdlib.builtin import chebyshev_sigmoid, clip

# Approximate sigmoid on encrypted data
shifted = clip(mpc, dot_product, -50.0, 10.0)
result = chebyshev_sigmoid(mpc, shifted, (-50.0, 10.0))
```

!!! note
    The `chebyshev_*` functions remain for compatibility; they evaluate the same interpolant as
    `nonlinear.approx` and the nonlinear functions' `"chebyshev"` method.

---

## Chebyshev interpolation engine

_Defined in `stdlib/sequre/stdlib/chebyshev.codon`_

The low-level Chebyshev machinery used by the built-in approximation functions above.

| Function | Signature | Description |
|---|---|---|
| `chebyshev_nodes` | `chebyshev_nodes(n, interval)` | Compute `n` Chebyshev nodes on the given interval |
| `chebyshev_coeffs` | `chebyshev_coeffs(op, nodes, interval)` | Compute Chebyshev polynomial coefficients for function `op` |
| `chebyshev_evaluate` | `chebyshev_evaluate(mpc, x, coeffs, interval)` | Evaluate a Chebyshev polynomial on encrypted data |
| `via_chebyshev` | `via_chebyshev(mpc, x, op, interval, degree, evaluation="clenshaw")` | End-to-end: compute nodes → coefficients → evaluate. `evaluation`: `"clenshaw"` (2 degree + 2 rounds), `"decor"` (the monomial form through Decor's polynomial decomposition: constant rounds, but its admissibility check limits the degree, to about 12 for sigmoid on (−8, 8) at F = 32) or `"naive"` (secure powers: refused when the estimated error exceeds `DECOR_TOLERANCE`, beyond degree 24 for sigmoid on (−8, 8) at F = 32). Both refuse an interval whose scaling to [−1, 1] alone errs beyond the tolerance (very wide intervals: the error grows with $\max(\lvert a \rvert, \lvert b \rvert) \, 2^{-F}$) |

```python
from sequre.stdlib.chebyshev import via_chebyshev
import math

# Custom function approximation
result = via_chebyshev(mpc, x, math.tanh, (-5.0, 5.0), degree=16)
```

---

## Linear algebra

_Defined in `stdlib/sequre/stdlib/lin_alg.codon`_

| Function | Signature | Description |
|---|---|---|
| `l2` | `l2(mpc, data)` | Pairwise L2 distance matrix |
| `householder` | `householder(mpc, x)` | Householder reflection vector |
| `qr_fact_square` | `qr_fact_square(mpc, A)` | QR factorization of a square matrix. Returns `(Q, R)` |
| `tridiag` | `tridiag(mpc, A)` | Tridiagonalization via Householder reflections. Returns `(T, Q)` |
| `eigen_decomp` | `eigen_decomp(mpc, A)` | Eigenvalue decomposition via QR iteration. Returns `(V, L)` where `V` is eigenvectors and `L` is eigenvalues |
| `orthonormalize` | `orthonormalize(mpc, A)` | Gram-Schmidt orthonormalization via Householder reflections |

All functions are decorated with `@sequre` and work on both `Sharetensor` and multiparty types.

```python
from sequre.stdlib.lin_alg import eigen_decomp, orthonormalize

V, eigenvalues = eigen_decomp(mpc, covariance_matrix)
Q = orthonormalize(mpc, sketch_matrix)
```

---

## Fixed-point arithmetic

_Defined in `stdlib/sequre/stdlib/fp.codon`_

Secure fixed-point division, inverse and square root by Goldschmidt and Newton iterations with normalizer-based
scaling, on either modulus: on the ring the normalizer comes from bitwise sharing (`mpc.bitwise.normalizer`), on the
field from the field protocol.

| Function | Signature | Description |
|---|---|---|
| `fp_div` | `fp_div(mpc, a, b)` | Secure fixed-point division $a / b$ |
| `fp_mul_inv` | `fp_mul_inv(mpc, b)` | Secure fixed-point inverse $1 / b$ |
| `fp_sqrt` | `fp_sqrt(mpc, a)` | Secure fixed-point square root. Returns `(sqrt, 1/sqrt)` |

The three functions use the `@flatten` decorator to transparently handle matrix inputs by flattening to 1-D, computing, and reshaping. They refuse to run inside `mpc.delayed_truncation()`.

!!! warning
    These are iterative methods whose convergence depends on the fixed-point precision settings (`MPC_NBIT_K`, `MPC_NBIT_F`).

---

## Bit-decomposition protocols

_Defined in `stdlib/sequre/stdlib/protocols.codon`_

Low-level protocols for secure bit-level operations on `Sharetensor` values. These are used internally by comparison operators and other operations.

| Function | Signature | Description |
|---|---|---|
| `prefix_carry` | `prefix_carry(mpc, s, p, k)` | Prefix carry computation (Damgård et al., 2006) |
| `carries` | `carries(mpc, a_bits, b_bits)` | Carry-overs when adding two secret-shared bit representations |
| `bit_add` | `bit_add(mpc, a_bits, b_bits)` | Bitwise addition of secret-shared bit decompositions |
| `bit_decomposition` | `bit_decomposition(mpc, a, bitlen, small_mod, mod)` | Decompose a secret-shared value into its bit representation |

!!! note
    These are internal building blocks. Most users interact with comparisons (`>`, `<`) through the operator overloads, which call these protocols automatically.
