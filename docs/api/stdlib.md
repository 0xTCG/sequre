# Secure Standard Library

_Defined in `stdlib/sequre/stdlib/`_

Sequre provides a standard library of secure functions that operate on `Sharetensor`, `Ciphertensor`, and multiparty types (`MPU`, `MPP`, `MPA`). These functions are building blocks for secure applications.

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

These functions approximate non-linear operations using Chebyshev polynomial interpolation. They require an `interval` parameter specifying the expected input range.

| Function | Signature | Approximates |
|---|---|---|
| `chebyshev_exp` | `chebyshev_exp(mpc, x, interval)` | $e^x$ |
| `chebyshev_sigmoid` | `chebyshev_sigmoid(mpc, x, interval)` | $\frac{1}{1 + e^{-x}}$ |
| `chebyshev_log` | `chebyshev_log(mpc, x, interval)` | $\ln(x)$ |
| `chebyshev_mul_inv` | `chebyshev_mul_inv(mpc, x, interval)` | $\frac{1}{x}$ |
| `chebyshev_sqrt` | `chebyshev_sqrt(mpc, x, interval)` | $\sqrt{x}$ |
| `chebyshev_sqrt_inv` | `chebyshev_sqrt_inv(mpc, x, interval)` | $\frac{1}{\sqrt{x}}$ |

The `interval` is a `tuple[float, float]` specifying the domain. The degree of the Chebyshev polynomial is set by the `CHEBYSHEV_DEGREE` constant.

```python
from sequre.stdlib.builtin import chebyshev_sigmoid, clip

# Approximate sigmoid on encrypted data
shifted = clip(mpc, dot_product, -50.0, 10.0)
result = chebyshev_sigmoid(mpc, shifted, (-50.0, 10.0))
```

---

## Chebyshev interpolation engine

_Defined in `stdlib/sequre/stdlib/chebyshev.codon`_

The low-level Chebyshev machinery used by the built-in approximation functions above.

| Function | Signature | Description |
|---|---|---|
| `chebyshev_nodes` | `chebyshev_nodes(n, interval)` | Compute `n` Chebyshev nodes on the given interval |
| `chebyshev_coeffs` | `chebyshev_coeffs(op, nodes, interval)` | Compute Chebyshev polynomial coefficients for function `op` |
| `chebyshev_evaluate` | `chebyshev_evaluate(mpc, x, coeffs, interval)` | Evaluate a Chebyshev polynomial on encrypted data |
| `via_chebyshev` | `via_chebyshev(mpc, x, op, interval, degree)` | End-to-end: compute nodes → coefficients → evaluate |

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

Secure fixed-point division and square root using iterative Newton-Raphson methods with normalizer-based scaling.

| Function | Signature | Description |
|---|---|---|
| `fp_div` | `fp_div(mpc, a, b)` | Secure fixed-point division $a / b$ |
| `fp_sqrt` | `fp_sqrt(mpc, a)` | Secure fixed-point square root. Returns `(sqrt, 1/sqrt)` |

Both functions use the `@flatten` decorator to transparently handle matrix inputs by flattening to 1-D, computing, and reshaping.

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
