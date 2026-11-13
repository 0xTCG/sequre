# MPC ↔ MHE Protocol Switching

_Defined across `stdlib/sequre/mpc/env.codon`, `stdlib/sequre/types/internal.codon`, and the multiparty type modules_

One of Sequre/Shechi's key features is the ability to **switch between Secure Multiparty Computation (MPC) and Multiparty Homomorphic Encryption (MHE) mid-computation**. Some operations — comparisons, certain matrix multiplications, inverse, square root — are impractical or impossible in pure HE but straightforward in additive secret sharing. Conversely, many linear-algebra operations are more efficient in HE.

## How it works

The switching mechanism is built on two low-level Lattiseq distributed protocols:

| Protocol | Direction | Description |
|---|---|---|
| **E2S** (Encryption-to-Shares) | MHE → MPC | The parties decrypt a ciphertext collectively under random masks, so that each ends up with an additive share of its values and none sees them |
| **S2E** (Shares-to-Encryption) | MPC → MHE | Each party encodes its additive share and masks it with its secret-key share; the sum is a fresh collective ciphertext |

Both directions hide the values statistically from every party; [Security Assumptions](../crypto-corner/security-assumptions.md#noise-and-masks-in-the-mhe-protocols) gives the guarantees.

A single round-trip (E2S → compute in MPC → S2E) is what `via_mpc` performs under the hood. The cost of one switch is estimated as:

$$\mathrm{switch\text{-}cost} \approx \mathrm{HE\text{-}ENC\text{-}COST} + \mathrm{HE\text{-}DEC\text{-}COST} \quad \text{per ciphertext}$$

This is defined by the `MHE_MPC_SWITCH_COST_ESTIMATE` constant and used by the cost estimator to decide between pure-MHE and switching paths.

## The `via_mpc` method

Every multiparty type exposes `via_mpc` as the primary user-facing switching API:

```python
# On Ciphertensor
result_ct = ct.via_mpc(mpc, lambda stensor: some_mpc_function(mpc, stensor))

# On MPP
result_mpp = mpp.via_mpc(lambda stensor: some_mpc_function(mpc, stensor))

# On MPA
result_mpa = mpa.via_mpc(lambda stensor: some_mpc_function(mpc, stensor))

# On MPU
result_mpu = mpu.via_mpc(lambda stensor: some_mpc_function(mpc, stensor))
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
| `Ciphertensor.to_sharetensor(mpc, ...)` | MHE → MPC | Decrypt via E2S into additive shares |
| `Sharetensor.to_ciphertensor(mpc)` | MPC → MHE | Re-encrypt via S2E into a collective ciphertext |
| `MPP.to_sharetensor(mpc)` | MHE → MPC | Convert partitioned encryption to shares |
| `Sharetensor.to_mpp(mpc, ratios)` | MPC → MHE | Convert shares to partitioned encryption |
| `MPA.to_sharetensor(mpc)` | MHE → MPC | Convert to shares: shares every compute party holds in the clear directly (each party shares its own); otherwise the aggregate, which every party first forms (a share held in the clear encrypted for it), so an MPA held in the clear at some parties and encrypted at others converts too |
| `Sharetensor.to_mpa(mpc)` | MPC → MHE | Convert shares to aggregated encryption |

The shares that `MPP.to_sharetensor` and `MPA.to_sharetensor` return are fixed point whatever the dtype: partitions switched from MHE come out fixed point, and integer partitions held in the clear are scaled to match, so that every party (the trusted dealer included) takes the same fixed-point steps. Integer values must therefore fit the fixed-point range, $|x| < 2^{k-f-1}$ (2^31 by default).

## Automatic switching with `AllowMPCSwitch`

For certain operations — notably `Ciphertensor.matmul` with a plaintext operand — Sequre includes a **cost estimator** that automatically chooses between a pure-HE path and a switching path. This is gated by the `AllowMPCSwitch` context manager:

```python
with mpc.allow_mpc_switch():
    result = ct_a @ plain_b  # cost estimator picks MHE or via_mpc
```

Every party must run the product, the trusted dealer (CP0) included: the via-MPC path needs it for the Beaver products and truncations. The cost estimator works from public data only (the shapes and layouts of the operands, and `mpc.default_allow_mpc_switch`), so every party picks the same strategy; CP0 joins the via-MPC path when it is chosen and otherwise gets a placeholder with the result's shape and layout.

Inside the context, `mpc.default_allow_mpc_switch` is set to `True`. For a row-wise `Ciphertensor @ ndarray` the cost estimator compares five strategies:

| Strategy | Label | Description |
|---|---|---|
| Via MPC | `Via MPC` | E2S → Beaver matmul → S2E |
| M1 | `M1` | Column-packed HE matmul |
| M2 | `M2` | Row-packed HE matmul |
| M3 | `M3` | Diagonal-packed HE matmul |
| BSGS | `BSGS` | Baby-step giant-step diagonal matmul (square plaintext of at most half the slots) |

The cheapest strategy wins, among those whose shape constraints hold (see the matrix products section of the [Ciphertensor API](../api/ciphertensor.md#matrix-products)). When `AllowMPCSwitch` is not active, the "Via MPC" cost is set to infinity and only pure-HE paths are considered. Column-wise (lazily transposed) and diagonal-encoded ciphertensors have no via-MPC path, nor does `Ciphertensor @ Ciphertensor`.

MPP and MPA products with a plaintext matrix take the via-MPC path under `AllowMPCSwitch` too, for the whole tensor at once:

- **MPP @ ndarray:** switching converts every partition's rows and multiplies them in MPC, while the HE path runs each compute party's product on its own partition at the same time (the partitions padded to the most rows). So the estimator compares the via-MPC cost of all the rows with the cheapest HE kernel on the largest partition, from the public ratios and shapes. When the via-MPC path wins, every party, CP0 included, takes one more round to find out whether the compute parties hold their partitions encrypted: only then does the MPP switch, at every party and with empty partitions joining in. Partitions held in the clear multiply locally.
- **MPA @ ndarray:** an MPA held as an aggregate, or as encrypted shares, converts once (the aggregate's common ciphertexts, or the sum of the shares) and comes back as an aggregate. It is costed like a row-wise `Ciphertensor` of the MPA's shape, with the same extra round. An MPA that keeps its shares in the clear multiplies them locally.

Each partition's own product never switches on its own: that would need the trusted dealer, which takes no part in it.

!!! tip
    Enable `DEBUG` mode to see the cost breakdown printed at each matmul:
    ```
    CP1:  Matmul costs:
          Via MPC: 12.34
          M1: 45.67
          M2: 23.45
          M3: 89.01
          BSGS: 34.56
    ```

## Methods that switch automatically

Several built-in operations use `via_mpc` under the hood without requiring `AllowMPCSwitch`:

### Comparisons

HE does not support comparisons, so MPP and MPA comparisons (`>`, `<`) with a public value run the secure comparison in MPC:

```python
mask = mpp > 0.0   # internally: mpp.via_mpc(lambda s: secure_operator.gt(mpc, s, 0.0))
mask = mpa < 1.0   # internally: mpa.via_mpc(lambda s: secure_operator.lt(mpc, s, 1.0))
```

The exception is an MPP (or a partition-mode MPU) that every compute party holds in the clear. There each compute party compares its own partition locally, the result stays in the clear at its owner, and the trusted dealer takes no part. The trusted dealer cannot see how the compute parties hold an MPP, so every comparison starts with one round among all parties, CP0 included, to agree on the path. Call comparisons at every party. An MPP that some compute parties hold encrypted and others in the clear goes through MPC as a whole.

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
| Division | _(implicit)_ | Dividing two encrypted operands (iterative Newton method in MPC) |
| `sqrt(mpc, x)` | `stdlib/sequre/stdlib/builtin.codon` | Square root (iterative Newton method in MPC) |
| `orthonormalize` | `applications/gwas.codon` | Gram-Schmidt orthonormalization |
| `eigen_decomp` | `stdlib/sequre/stdlib/pca.codon` | Eigenvalue decomposition |

## Monitoring switches

The number of E2S/S2E round-trips is tracked by the statistics module:

```python
print(mpc.stats.secure_mhe_mpc_switch_count)  # number of switches so far
```

Use `mpc.stats` or the `mpc.stats_log()` context manager to profile and log switching overhead.

## When to use switching

| Scenario | Recommendation |
|---|---|
| Comparison operators (`>`, `<`) | Automatic — switches, except on an MPP every compute party holds in the clear |
| Matrix multiply (ciphertext × plaintext) | Use `AllowMPCSwitch` and let the cost estimator decide; run the product at every party, the trusted dealer included |
| Matrix inverse, square root | Use `via_mpc` explicitly or call the stdlib `inv`/`sqrt` |
| Custom non-linear function | Call `via_mpc` directly with a lambda |
| All operations are linear (add, mul, rotate) | Stay in pure MHE — no switching needed |

!!! warning
    Each switch incurs E2S + S2E communication rounds. For small tensors the overhead may outweigh the benefit. The cost estimator accounts for this, but when calling `via_mpc` manually, consider whether the operation truly requires MPC.
