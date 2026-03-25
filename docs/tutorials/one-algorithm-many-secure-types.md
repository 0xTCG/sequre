# One Algorithm, Many Secure Types

One of the nicest things about Sequre is that an algorithm can be implemented once and then run on plaintext ndarrays, on secret-shared Sharetensors, and on encrypted multiparty types (MPU/MPP) — without changing the algorithm code. This is extremely useful in practice: prototyping and debugging happens on plaintext first, then flipping the data type to a secure one reveals whether the outputs match.

This page walks through how that works, using linear regression and PCA as concrete examples.

## Linear regression: one class, any type

Open [stdlib/sequre/stdlib/learn/lin_reg.codon](../../stdlib/sequre/stdlib/learn/lin_reg.codon). The class is declared as `LinReg[T]` — it's generic over the tensor type `T`:

```python
class LinReg[T]:
    coef_: T
    optimizer: str

    def fit(self, mpc, X: T, y: T, step: float, epochs: int, ...) -> LinReg[T]:
        self.coef_ = LinReg._fit(mpc, X, y, self.coef_, ...)
        return self

    def predict(self, mpc, X: T, noise_scale: float = 0.0) -> T:
        return LinReg._predict(mpc, X, self.coef_, noise_scale)
```

The interesting part is what `_fit` actually does. Look at the batch gradient descent inner loop:

```python
@sequre
def _bgd(mpc, X_tilde: T, y: T, initial_w: T, step: float, epochs: int, ...) -> T:
    # Pre-compute invariants
    cov = X_tilde.T @ X_tilde  # n x n
    ref = X_tilde.T @ y        # n x 1

    w = initial_w
    for _ in range(epochs):
        w += (ref - cov @ w) * step

    return w
```

There's nothing type-specific in this code. `X_tilde.T @ X_tilde` is just matrix multiplication — but what happens underneath depends entirely on what `T` is. If `T` is `ndarray`, it's ordinary NumPy-style arithmetic. If `T` is `Sharetensor`, the `@sequre` decorator rewrites `@` into Beaver-triple secure multiplication. If `T` is `MPU`, the framework picks between HE-based matmul strategies or switches to MPC via `via_mpc`, depending on estimated cost.

The closed-form solver shows the same pattern with `inv`:

```python
@sequre
def _closed_form(mpc, X: T, y: T) -> T:
    return inv(mpc, X.T @ X) @ X.T @ y
```

`inv` also dispatches by type — for `Sharetensor` or `ndarray` it does direct matrix inversion formula; for encrypted types it calls `x.via_mpc(lambda stensor: inv(mpc, stensor))` to switch to MPC, compute the inverse there, and switch back. This can be seen in [stdlib/sequre/stdlib/builtin.codon](../../stdlib/sequre/stdlib/builtin.codon).

### Where this is used for real

The Multiple Imputation application ([applications/mi.codon](../../applications/mi.codon)) uses `LinReg[T]` and `LogReg[T]` over parameterized secure types — same algorithm, different backends depending on the deployment scenario.

## PCA: same algorithm on four data types

The PCA test in [tests/e2e_tests/test_pca.codon](../../tests/e2e_tests/test_pca.codon) is the clearest side-by-side comparison of running the same computation across representations. Here's what happens:

**Step 1: Run on plaintext.** The test calls `random_pca_with_norm(mpc, raw_data, ...)` where `raw_data` is just an ndarray. This gives a plaintext reference result.

**Step 2: Run on Sharetensor.** Same call, but now the data is secret-shared:

```python
mpc_data = Sharetensor.enc(mpc, raw_data, 0, modulus)
# ... (encode all inputs as Sharetensors)

mpc_pca_u, mpc_pca_z = random_pca_with_norm(
    mpc, mpc_data, mpc_miss, mpc_data_mean, mpc_data_std_inv, ...)
```

Then the test reveals the MPC result and asserts approximate equality with the plaintext version:

```python
assert_eq_approx("Sequre std PCA U (MPC)", mpc_pca_u.reveal(mpc), classic_pca_u)
```

**Step 3: Run on MPP and MPU.** The same data is loaded into partitioned/encrypted forms:

```python
mpp_data = MPP(mpc, ... raw_cent_data[(mpc.pid - 1) * rows_per_party:mpc.pid * rows_per_party])
mpu_data = MPU(mpc, mpp_data._local_data, "partition")
```

And PCA runs again, this time using `random_pca_without_projection` which internally does HE-backed matrix multiplications and calls `.via_mpc(...)` when it hits operations that need MPC (like orthonormalization and eigendecomposition). The result is again checked against the plaintext reference.

This is the recommended workflow for building a new protocol:

1. Get the math right on ndarray.
2. Switch to `Sharetensor` and check that the secure version matches.
3. Move to `MPU`/`MPP` for the distributed/encrypted-scale execution.
4. Fix any numerical drift (CKKS is approximate, so you may need to tune tolerances).

## How the dispatch works

There is no need to write separate implementations for each type. The `@sequre` decorator and Codon's operator overloading handle the dispatch:

- On `ndarray`: operators are plain arithmetic.
- On `Sharetensor`: `+` is local, `*` and `@` use Beaver triples (communication round).
- On `MPU`/`MPP`/`MPA`: operators route to the underlying `Ciphertensor` HE operations or switch to MPC via `via_mpc` when needed.

## Next steps

- [Transitioning to MHE](transition-mhe.md) — When and why to use Shechi's encrypted types instead of Sharetensor.
- [Dropping Down the Stack](dropping-down-the-stack.md) — What happens below `@sequre` and `via_mpc`.
- [Distributed Tensors (MPU)](../user-guide/distributed-tensors.md) — Detailed reference for MPU/MPP/MPA.
- [MPC ↔ MHE Protocol Switching](../user-guide/switching.md) — How `via_mpc` works under the hood.
