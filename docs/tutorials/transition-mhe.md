# Transitioning to MHE (Shechi)

This tutorial explains when and how to switch from Sequre's additive secret sharing to **Shechi**, the multiparty homomorphic encryption (MHE) sub-system.

## Why MHE?

Additive secret sharing excels at linear operations on smaller data scale (addition is free, multiplication costs one communication round). But for workloads heavy in:

- **Consecutive multiplications** — each requires a full network round
- **Matrix multiplications** — communication scales with matrix dimensions
- **Floating-point batched arithmetic** — fixed-point truncation adds overhead
- **Large scale data** — MHE scales better than secret-sharing MPC

MHE can be significantly faster because computations happen **locally on data-parallel encrypted data** with communication only for:

1. Collective key generation (one-time setup)
2. Key-switching / relinearization (after multiplications)
3. Ciphertext refresh / bootstrapping (when noise budget runs low)

## The CKKS advantage

Shechi uses the CKKS homomorphic encryption scheme, which natively supports:

- **Approximate arithmetic** on real/complex numbers (no fixed-point encoding needed)
- **SIMD batching** — a single ciphertext holds thousands of values in "slots"
- **Efficient rotations** — reorder data within a ciphertext without decrypting

## Using Shechi types

### `Ciphertensor` — encrypted local tensor

```python
from sequre.lattiseq.ckks import Ciphertext
from sequre.types.ciphertensor import Ciphertensor

# Encrypt a vector into CKKS ciphertexts
ct_list = mpc.mhe.enc_vector[Ciphertext](my_values)
ct = Ciphertensor(ct_list, shape=[len(my_values)], slots=params.slots())

# Arithmetic works via operator overloading
result = ct_a + ct_b      # HE addition
result = ct_a * ct_b      # HE multiplication + relinearization
```

### `MPU` — multiparty union

`MPU` is the highest-level abstraction. It allows writing code that looks like ordinary tensor arithmetic while the framework handles encryption, distribution, and collective operations:

```python
from sequre.types.multiparty_union import MPU

# Each party provides its local rows
mpu = MPU(mpc, my_local_data, "partition")

# Standard operations dispatch to HE under the hood
result = mpu @ weights + bias
```

### `MPP` vs `MPA`

`MPU` is internally backed by one of:

| Type | Data distribution | Use case |
|---|---|---|
| `MPP` (partition) | Each party holds a block of rows | Horizontal data partitioning (e.g., each hospital has patients) |
| `MPA` (aggregate) | Each party holds an additive share | Additive sharing with optional encryption |

Choose via the second argument to `MPU(mpc, data, "partition")` or `MPU(mpc, data, "additive")`.

## MHE setup

MHE requires a one-time collective key generation phase:

```python
mpc = mpc()  # Automatically calls mpc.mhe.default_setup()
```

This runs:

1. **Secret key shard generation** — each party generates a local secret key shard
2. **Collective public key generation (CKG)** — interactive protocol produces a joint public key
3. **Collective relinearization key generation (RKG)** — joint key for post-multiplication cleanup
4. **Collective rotation key generation** — joint Galois keys for slot rotations

Adding `--skip-mhe-setup` flag skips MHE setup. Useful when only MPC is needed.

<!-- ## Compiler optimizations for MHE

Shechi includes compiler-level IR passes that optimize HE code automatically:

### `@mhe_cipher_opt` — expression reordering

The compiler builds a Binary Expression Tree (BET) of the function's arithmetic, then:

- Factorizes common subexpressions: `a*b + a*c` → `a*(b + c)` (saves one expensive HE multiply)
- Reorders operations to minimize ciphertext-ciphertext multiplications

### `@mhe_enc_opt` — encoding optimization

For matrix multiplications, the compiler selects the optimal CKKS encoding strategy (row-wise, column-wise, or diagonal) via brute-force search over encoding candidates.

Both passes are applied automatically to `@sequre`-annotated functions when the operands are Ciphertensor or MPU types. -->

## When to use which

| Criterion | MPC (Sharetensor) | MHE (Shechi) |
|---|---|---|
| Setup cost | None | Key generation (seconds) |
| Per-multiply cost | 1 network round | Local HE op (~34ms, no network) |
| Batching | No | Yes (thousands of values per ciphertext) |
| Precision | Exact (integer/fixed-point) | Approximate (CKKS) |
| Best for | Exact results, smaller scale data | Dense linear algebra, large scale ML training |

## Next steps

- **[Distributed Tensors (MPU)](../user-guide/distributed-tensors.md)** — Deep dive into MPU semantics.
- **[Ciphertensor API](../api/ciphertensor.md)** — Complete reference.
- **[Core MHE module](../deep-dive-shechi/core-mhe.md)** — Internals of the collective protocols.
