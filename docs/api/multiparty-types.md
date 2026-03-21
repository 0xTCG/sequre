# Multiparty Types: MPU, MPP, MPA

_Defined in `stdlib/sequre/types/multiparty_union.codon`, `multiparty_partition.codon`, `multiparty_aggregate.codon`_

These three types model how data is distributed across parties in a multiparty computation. They are the primary abstractions for Shechi's high-level layer.

---

## MPU — Multiparty Union

`MPU[S, dtype]` is the highest-level distributed type. It is a *union* of `MPP` and `MPA`, selecting the appropriate one internally.

### Construction

```python
from sequre.types.multiparty_union import MPU

# Horizontal partition: each party owns a subset of rows
mpu = MPU(mpc, my_local_rows, "partition")

# Additive sharing: each party holds an additive share
mpu = MPU(mpc, my_share, "additive")
```

### Type parameters

| Parameter | Description |
|---|---|
| `S` | Shape type (e.g., `Tuple[int]` for 1-D, `Tuple[int, int]` for 2-D) |
| `dtype` | Element type (typically `float`) |

### Operators

All standard arithmetic and comparison operators are supported. They dispatch to either MPP or MPA operations depending on how the MPU was constructed:

- `+`, `-`, `*`, `@` — Arithmetic
- `>`, `<`, `==` — Comparisons (vs scalars)
- `[]`, `[]=` — Indexing and slicing

### Key methods

| Method | Description |
|---|---|
| `.is_mpp()` / `.is_mpa()` | Check which internal representation is active |
| `.shape` | Logical shape of the distributed data |
| `.T` | Transpose |
| `.sum(axis)` | Distributed reduction |

---

## MPP — Multiparty Partition

`MPP[S, dtype]` represents data that is **horizontally partitioned** across parties. Each party holds a contiguous block of rows of the global matrix.

### Fields

| Field | Description |
|---|---|
| `_mpc` | Reference to the `MPCEnv` |
| `_ratios` | List of per-party row counts |
| `_local_data` | This party's local plaintext rows |
| `_encryption_unified` | Encrypted version of the local data (`Ciphertensor[Ciphertext]`) |

### Construction

```python
from sequre.types.multiparty_partition import MPP

# From local data (each party provides its own rows)
mpp = MPP(mpc, local_rows)

# From a Ciphertensor (each party provides encrypted local data)
mpp = MPP(mpc, local_ciphertensor)
```

!!! note
    Party 0 (trusted dealer) holds a zero-filled placeholder. Actual data resides on parties 1..N.

---

## MPA — Multiparty Aggregate

`MPA[S, dtype]` represents data where each party holds an **additive share** of the global value, optionally encrypted.

### Fields

| Field | Description |
|---|---|
| `_mpc` | Reference to the `MPCEnv` |
| `_plain` | Plaintext share (ndarray) |
| `_encryption` | Per-party encrypted share (`Ciphertensor`) |
| `_aggregate` | Aggregated (summed) encrypted ciphertext |

### Key methods

| Method | Description |
|---|---|
| `.via_mpc(fn)` | Apply a function through the MPC layer (converts to Sharetensor, runs fn, converts back) |
| `.enc()` | Encrypt the plaintext share |
| `.has_plain()` / `.has_encryption()` / `.has_aggregate()` | State queries |
