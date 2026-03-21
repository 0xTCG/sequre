# SMC ↔ MHE Protocol Switching

_Defined across `stdlib/sequre/mpc/env.codon`, `stdlib/sequre/types/internal.codon`, and the multiparty type modules_

One of Sequre/Shechi's key features is the ability to **switch between Secure Multiparty Computation (SMC) and Multiparty Homomorphic Encryption (MHE) mid-computation**. Some operations — comparisons, certain matrix multiplications, inverse, square root — are impractical or impossible in pure HE but straightforward in additive secret sharing. Conversely, many linear-algebra operations are more efficient in HE. Sequre lets you mix both within the same program.

## How it works

The switching mechanism is built on two low-level Lattiseq distributed protocols:

| Protocol | Direction | Description |
|---|---|---|
| **E2S** (Encryption-to-Shares) | MHE → SMC | Each party collectively decrypts a ciphertext into additive secret shares |
| **S2E** (Shares-to-Encryption) | SMC → MHE | Each party re-encrypts its additive share; the ciphertexts are summed to produce a fresh collective ciphertext |

A single round-trip (E2S → compute in SMC → S2E) is what `via_mpc` performs under the hood. The cost of one switch is estimated as:

$$
\text{switch\_cost} \approx \text{HE\_ENC\_COST} + \text{HE\_DEC\_COST} \;\;\text{per ciphertext}
$$

This is defined by the `MHE_MPC_SWITCH_COST_ESTIMATE` constant and used by the cost estimator to decide between pure-MHE and switching paths.

## The `via_mpc` method

Every multiparty type exposes `via_mpc` as the primary user-facing switching API:

```python
# On Ciphertensor
result_ct = ct.via_mpc(mpc, lambda stensor: some_smc_function(mpc, stensor))

# On MPP
result_mpp = mpp.via_mpc(lambda stensor: some_smc_function(mpc, stensor))

# On MPA
result_mpa = mpa.via_mpc(lambda stensor: some_smc_function(mpc, stensor))

# On MPU
result_mpu = mpu.via_mpc(lambda stensor: some_smc_function(mpc, stensor))
```

The flow inside `via_mpc` is:

1. **Collect** — gather encrypted data from all parties (if needed)
2. **E2S** — convert each party's `Ciphertensor` to a `Sharetensor` (additive share)
3. **Compute** — run the user-supplied lambda on the `Sharetensor` using Beaver-triple MPC
4. **S2E** — convert the result `Sharetensor` back to a `Ciphertensor`

The Ciphertensor variant additionally supports `mirrored` mode (all parties hold the same ciphertext) and `exclude_parties` to skip specific parties.

## Underlying conversion methods

The actual type conversions are implemented in `stdlib/sequre/types/internal.codon`:

| Method | Direction | Description |
|---|---|---|
| `Ciphertensor.to_sharetensor(mpc, ...)` | MHE → SMC | Decrypt via E2S into additive shares |
| `Sharetensor.to_ciphertensor(mpc)` | SMC → MHE | Re-encrypt via S2E into a collective ciphertext |
| `MPP.to_sharetensor(mpc)` | MHE → SMC | Convert partitioned encryption to shares |
| `Sharetensor.to_mpp(mpc, ratios)` | SMC → MHE | Convert shares to partitioned encryption |
| `MPA.to_sharetensor(mpc)` | MHE → SMC | Convert aggregated encryption to shares |
| `Sharetensor.to_mpa(mpc)` | SMC → MHE | Convert shares to aggregated encryption |

## Automatic switching with `AllowMPCSwitch`

For certain operations — notably `Ciphertensor.matmul` with a plaintext operand — Sequre includes a **cost estimator** that automatically chooses between a pure-HE path and a switching path. This is gated by the `AllowMPCSwitch` context manager:

```python
with mpc.allow_mpc_switch():
    result = ct_a @ plain_b  # cost estimator picks MHE or via_mpc
```

Inside the context, `mpc.default_allow_mpc_switch` is set to `True`. The cost estimator compares four strategies for `Ciphertensor @ ndarray`:

| Strategy | Label | Description |
|---|---|---|
| Via SMC | `Via SMC` | E2S → Beaver matmul → S2E |
| M1 | `M1` | Column-packed HE matmul |
| M2 | `M2` | Row-packed HE matmul |
| M3 | `M3` | Diagonal-packed HE matmul |

The cheapest strategy wins. When `AllowMPCSwitch` is not active, the "Via SMC" cost is set to infinity and only pure-HE paths are considered.

!!! tip
    Enable `DEBUG` mode to see the cost breakdown printed at each matmul:
    ```
    CP1:  Matmul costs:
          Via SMC: 12.34
          M1: 45.67
          M2: 23.45
          M3: 89.01
    ```

## Methods that switch automatically

Several built-in operations use `via_mpc` under the hood without requiring `AllowMPCSwitch`:

### Comparisons

MPP and MPA comparisons (`>`, `<`) always switch to SMC because HE does not natively support comparison:

```python
mask = mpp > 0.0   # internally: mpp.via_mpc(lambda s: secure_operator.gt(mpc, s, 0.0))
mask = mpa < 1.0   # internally: mpa.via_mpc(lambda s: secure_operator.lt(mpc, s, 1.0))
```

### Layout transformations

These Ciphertensor methods use `via_mpc` to materialize pending layout changes:

- `actual_transpose(mpc)` — when a non-diagonal ciphertensor needs an actual (non-lazy) transpose
- `diagonal_contig(mpc)` — converting to diagonal-contiguous encoding from ciphertext form
- `get_actual(mpc)` / `iget_actual(mpc)` — resolving pending transposition or diagonality

### MPP matrix multiplication

Certain MPP matmul cases use switching:

- `_matmul_case_ntnt` — non-transposed × non-transposed with a switching path
- `_matmul_case_tt` — transposed × transposed with a switching path

### Standard-library functions

Higher-level Sequre stdlib routines that rely on switching:

| Function | Location | Uses switching for |
|---|---|---|
| `inv(mpc, x)` | `stdlib/sequre/stdlib/builtin.codon` | Matrix inverse (iterative Newton method in SMC) |
| `sqrt(mpc, x)` | `stdlib/sequre/stdlib/builtin.codon` | Square root (iterative approximation in SMC) |
| `orthonormalize` | `applications/gwas.codon` | Gram-Schmidt orthonormalization |
| `eigen_decomp` | `stdlib/sequre/stdlib/pca.codon` | Eigenvalue decomposition |

## Monitoring switches

The number of E2S/S2E round-trips is tracked by the statistics module:

```python
print(mpc.stats.secure_mhe_mpc_switch_count)  # number of switches so far
```

Use `mpc.stats` or the `StatsLog` context manager to profile switching overhead in your application.

## When to use switching

| Scenario | Recommendation |
|---|---|
| Comparison operators (`>`, `<`, `==`) | Automatic — always switches |
| Matrix multiply (ciphertext × plaintext) | Use `AllowMPCSwitch` and let the cost estimator decide |
| Matrix inverse, square root | Use `via_mpc` explicitly or call the stdlib `inv`/`sqrt` |
| Custom non-linear function | Call `via_mpc` directly with a lambda |
| All operations are linear (add, mul, rotate) | Stay in pure MHE — no switching needed |

!!! warning
    Each switch incurs E2S + S2E communication rounds. For small tensors the overhead may outweigh the benefit. The cost estimator accounts for this, but when calling `via_mpc` manually, consider whether the operation truly requires SMC.
