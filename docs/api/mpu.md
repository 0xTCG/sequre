# MPU — Multiparty Union

_Defined in `stdlib/sequre/types/multiparty_union.codon`_

`MPU[S, dtype]` is the highest-level distributed type in Sequre. It is a *union* of [MPP](mpp.md) and [MPA](mpa.md), selecting the appropriate one internally based on how data is distributed.

---

## Construction

```python
from sequre.types.multiparty_union import MPU

# Horizontal partition: each party owns a subset of rows
mpu = MPU(mpc, my_local_rows, "partition")

# Additive sharing: each party holds an additive share
mpu = MPU(mpc, my_share, "additive")
```

## Type parameters

| Parameter | Description |
|---|---|
| `S` | Shape type (e.g., `Tuple[int]` for 1-D, `Tuple[int, int]` for 2-D) |
| `dtype` | Element type (typically `float`) |

## Factory methods

| Method | Description |
|---|---|
| `MPU.partition(mpc, data)` | Create a horizontally partitioned MPU |
| `MPU.additive(mpc, data)` | Create an additively shared MPU |
| `MPU.rand(shape, distribution, mpc, collective_type)` | Random MPU with specified collective type |
| `MPU.enc(mpc, data)` | Encrypt data with specified ratios |

## Properties and state queries

| Property / Method | Type | Description |
|---|---|---|
| `.shape` | `S` | Logical shape across all parties |
| `.cohort_shape` | `S` | Alias for distributed shape |
| `.partition_shape` | `S` | Shape of local partition |
| `.shape_local` | `S` | Local shape (MPA) or partition shape (MPP) |
| `.ndim` | `int` | Number of dimensions |
| `.mpc` | `MPCEnv` | Associated MPC environment |
| `.T` | `MPU` | Transpose |
| `.I` | `MPU` | Identity matrix |
| `.is_mpp()` | `bool` | Check if internal representation is MPP |
| `.is_mpa()` | `bool` | Check if internal representation is MPA |
| `.is_valid()` | `bool` | Verify exactly one of MPP/MPA is set |
| `.is_empty()` | `bool` | Check if empty |

## Operators

All standard arithmetic and comparison operators are supported. They dispatch to either MPP or MPA operations depending on how the MPU was constructed:

| Operator | Description |
|---|---|
| `a + b` | Addition (MPU, scalar, or ndarray) |
| `a - b` | Subtraction |
| `a * b` | Element-wise multiplication |
| `a @ b` | Matrix multiplication |
| `a / b` | Division by scalar |
| `a ** n` | Integer exponentiation |
| `-a` | Negation |
| `a > b`, `a < b` | Comparisons (vs. scalars) |
| `a[i]`, `a[i] = v` | Indexing and slicing |

## Methods

| Method | Description |
|---|---|
| `.reveal()` | Reveal all data with secure aggregation |
| `.reveal_local()` | Reveal local share without aggregation |
| `.encrypt()` | Encrypt plaintext data |
| `.level()` | Current HE noise-growth level |
| `.sum(axis)` | Distributed sum reduction |
| `.dot(axis)` | Distributed dot product |
| `.via_mpc(fn, *args)` | Execute via [SMC layer](../user-guide/switching.md) |
| `.sign(*args)` | Secure sign function |
| `.expand_dims(axis=0)` | Insert dimension |
| `.extend(other)` | Concatenate with compatible MPU |
| `.pad_with_value(val, size, axis)` | Pad with repeated value |
| `.erase_element(index)` | Remove element at index |
| `.filter(mask)` | Boolean mask filtering |
| `.hstack(other)` | Horizontal concatenation |
| `.astype(T)` | Cast to different dtype |
| `.to_fp()` | Convert to floating point |
| `.copy()` | Deep copy |
| `.zeros(...)` | Zero tensor (same or specified shape) |
| `.ones(...)` | Ones tensor |
| `.rand(distribution, ...)` | Random values |
| `.get_matmul_cost(other)` | Estimate matmul cost |
| `.getitem_local(index)` | Local indexing without communication |
| `.slice_local(i, ...)` | Local slicing |
| `.rotate_local(i, ...)` | Local rotation |

## See also

- [MPP](mpp.md) — The partitioned representation used when `MPU` is constructed with `"partition"`
- [MPA](mpa.md) — The aggregated representation used when `MPU` is constructed with `"additive"`
- [SMC ↔ MHE Switching](../user-guide/switching.md) — How `via_mpc` works under the hood
