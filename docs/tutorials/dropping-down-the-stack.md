# Dropping Down the Stack

Sequre is usually written at the top level — `@sequre` functions and tensor expressions. However, it also allows for a finer control: maybe a particular subroutine is better in MPC than HE, specific matmul strategy needs to be picked, or even the raw CKKS procedures invoked.

This page traces two real examples from high-level code down through the layers, showing where each boundary is and when it makes sense to cross it.

For the overall architecture (Layers 1–4), see the [Home page](../index.md).

## Example A: PCA and `via_mpc`

### Starting point: high-level PCA

In [stdlib/sequre/stdlib/learn/pca.codon](../../stdlib/sequre/stdlib/learn/pca.codon), the randomized PCA routines are written as straightforward linear algebra — matrix products, orthonormalization, eigendecomposition. Most of the algorithm stays at the top level: `@` for matrix multiply, `.T` for transpose, slicing for submatrix extraction.

But two operations can't be done (or are very expensive) in pure HE: orthonormalization and eigendecomposition. So PCA drops into MPC for those, using `via_mpc`:

```python
# Inside random_pca_without_projection:
# Step 4 — orthonormalize needs comparisons, which HE can't do natively.
r_mpp = (p_mpa @ data_mpp.T).via_mpc(
    lambda stensor: orthonormalize(mpc, stensor))

# Step 6 — eigendecomposition also needs MPC.
u_mpa = z_cov_mpa.via_mpc(
    lambda stensor: eigen_decomp(mpc, stensor)[0][:top_components_count])
```

What happens when `.via_mpc(fn)` is called on an encrypted type like `MPA` or `MPP`:

1. The encrypted data is collectively decrypted into additive shares (E2S protocol).
2. The lambda runs on the resulting `Sharetensor` — using Beaver-triple MPC.
3. The result is re-encrypted back into the original form (S2E protocol).

So `via_mpc` is the boundary between "stay in HE" and "temporarily drop into MPC for this one operation." Everything around it — the `@` products, the slicing — stays in the HE world.

### One layer down: type conversions

The E2S and S2E steps are implemented in [stdlib/sequre/types/internal.codon](../../stdlib/sequre/types/internal.codon):

- `Ciphertensor.to_sharetensor(mpc, ...)` — decrypts via the E2S protocol.
- `Sharetensor.to_ciphertensor(mpc, ...)` — re-encrypts via S2E.
- `to_mpp`, `to_mpa`, `to_mpu` — handle the multiparty wrappers.

These methods delegate to `mpc.mhe.ciphervector_to_additive_share_vector` and `mpc.mhe.additive_share_vector_to_ciphervector` — the core MHE conversion routines in [stdlib/sequre/mpc/mhe.codon](../../stdlib/sequre/mpc/mhe.codon), documented in [Core MHE Module](../deep-dive-shechi/core-mhe.md).

### The deepest layer: Lattiseq protocols

Those MHE conversion methods ultimately call into the Lattiseq distributed CKKS protocols — `E2SProtocol`, `S2EProtocol`, `RefreshProtocol` — which operate on ring polynomials, NTT transforms, and secret key shards. For details, see [Lattiseq Overview](../deep-dive-lattiseq/overview.md) and [CKKS Operations](../deep-dive-lattiseq/ckks-operations.md).

This layer rarely needs to be touched directly. But when implementing a new collective protocol or debugging bootstrap failures, this is where things end up.

## Example B: how `@` picks a matmul strategy

### Starting point: linear regression

In [stdlib/sequre/stdlib/learn/lin_reg.codon](../../stdlib/sequre/stdlib/learn/lin_reg.codon), the gradient descent loop does:

```python
cov = X_tilde.T @ X_tilde  # n x n
ref = X_tilde.T @ y         # n x 1
```

When `X_tilde` is a `Sharetensor`, this is a standard Beaver-triple matmul — one implementation, done. But when `X_tilde` is backed by a `Ciphertensor` (HE), the `@` operator has to make a choice.

### One layer down: the cost selector

The key function is `_switch_matmul_by_cost` in [stdlib/sequre/types/ciphertensor.codon](../../stdlib/sequre/types/ciphertensor.codon). When a `Ciphertensor` is multiplied by a plaintext `ndarray`, it estimates the cost of four strategies:

```python
costs = (Ciphertensor._get_matmul_via_mpc_cost(self, other),   # decrypt, MPC matmul, re-encrypt
         Ciphertensor._get_matmul_v1_cost(self, other),          # M1: column-packed HE
         Ciphertensor._get_matmul_v2_cost(self, other),          # M2: row-packed HE
         Ciphertensor._get_matmul_v3_cost(self, other))          # M3: diagonal-packed HE

if not mpc.default_allow_mpc_switch:
    costs = (inf, *costs[1:])       # disable MPC path unless opted in
```

Then it picks the cheapest:

```python
match argmin(costs):
    case 0: return self.via_mpc(mpc, lambda stensor: secure_operator.matmul(mpc, stensor, other), ...)
    case 1: return self._matmul_v1(mpc, other_cipher, debug)
    case 2: return self._matmul_v2(mpc, ..., debug)
    case 3: return self._matmul_v3(mpc, ..., debug)
```

So the same `@` in the algorithm code can end up as a completely different computation path depending on tensor shapes and whether `mpc.allow_mpc_switch()` is active. `DEBUG` mode can be enabled to see the cost breakdown printed at runtime.

The three pure-HE strategies (M1, M2, M3) differ in how they pack matrix elements into CKKS ciphertext slots — column-wise, row-wise, or diagonal-wise. Each has different rotation and multiplication costs depending on the matrix dimensions. The "Via MPC" path does the full E2S → Beaver matmul → S2E round-trip.

### Going deeper

The M1/M2/M3 implementations call `mpc.mhe.iadd`, `mpc.mhe.imul`, `mpc.mhe.irotate` — working directly with ciphertext-level operations. These are the Layer 2 (MPCEnv/MHE) primitives documented in [Core MHE Module](../deep-dive-shechi/core-mhe.md). Below that, each of those methods calls into Lattiseq's `Evaluator` for the actual CKKS polynomial arithmetic.

## When to drop a layer

Most protocol code should stay at the top level. Here's a rough guide for when it makes sense to go deeper:

**Stay at `@sequre` / tensor level** for algorithm logic — this is where all the built-in secure functions (`inv`, `sqrt`, `maximum`, etc.) and type-generic patterns live. See [One Algorithm, Many Secure Types](one-algorithm-many-secure-types.md).

**Use `via_mpc` explicitly** when you have a non-linear or comparison-heavy step inside an otherwise HE-based pipeline. PCA's eigendecomposition is the canonical example. See [MPC ↔ MHE Switching](../user-guide/switching.md).

**Drop into Ciphertensor / MHE** when encoding strategy needs to be controlled, ciphertext levels manually managed, or the matmul cost model tuned. See [Ciphertensor Internals](../deep-dive-shechi/ciphertensor-internals.md) and [Core MHE Module](../deep-dive-shechi/core-mhe.md).

**Use Lattiseq directly** only when extending the cryptographic protocols themselves — new collective operations, custom refresh logic, or low-level debugging of noise/precision issues. See [Lattiseq Overview](../deep-dive-lattiseq/overview.md) and [Lattiseq API](../api/lattiseq.md).

## Next steps

- [One Algorithm, Many Secure Types](one-algorithm-many-secure-types.md) — How the same code runs on ndarray, Sharetensor, and MPU.
- [Transitioning to MHE](transition-mhe.md) — When to use Shechi's encrypted types.
- [Distributed Tensors (MPU)](../user-guide/distributed-tensors.md) — MPU/MPP/MPA reference.
