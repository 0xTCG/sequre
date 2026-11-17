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
| `a + b` | Addition (MPU, scalar, or ndarray). An additive-mode MPU plus a partition-mode one: each compute party adds its partition into its own share, in the clear if every share and every partition is in the clear, otherwise with every share encrypted, in one layout: that of shares held only encrypted, else that of the encrypted partitions. An additive MPU that holds only its aggregate gets every partition added to the aggregate (still only an aggregate: every party adds the same partitions to its copy). A partition in another layout than such fixed shares or the aggregate is converted as it is added, at one masked move per value; diagonal-encoded storage raises `NotImplementedError` at every compute party. The result holds only what the addition updated: an aggregate, or an encryption kept beside shares in the clear, is not carried over (`aggregate()` rebuilds it when needed) |
| `a - b` | Subtraction (likewise for additive minus partition-mode) |
| `a * b` | Element-wise multiplication |
| `a @ b` | Matrix multiplication. Partition mode: `A.T @ B` gives additive shares (each compute party's product of its own rows of both); `A.T @ B.T`, which is `(B @ A).T`, and the other cases are the [MPP](mpp.md) products (through MPC where they cross the partitions) |
| `a / b` | Division by scalar |
| `a ** n` | Integer exponentiation: additive mode as [`MPA ** n`](mpa.md) (the power of the value), partition mode each partition's power |
| `-a` | Negation |
| `a > b`, `a < b` | Comparisons (vs. scalars): local at each owner for a partition-mode MPU every compute party holds in the clear, otherwise the secure comparison in MPC (see [switching](../user-guide/switching.md#comparisons)) |
| `a[i]`, `a[i] = v` | Indexing and slicing (as for [MPP](mpp.md) or [MPA](mpa.md)) |

## Methods

| Method | Description |
|---|---|
| `.reveal()` | Reveal all data with secure aggregation |
| `.reveal_local()` | This compute party's own data: its partition (partition mode, as `MPP.reveal_local`) or its share (additive mode, as `MPA.reveal_local`), decrypted to it alone; call it at every compute party |
| `.encrypt()` | Encrypt plaintext data |
| `.level()` | Current HE noise-growth level |
| `.sum(axis)` | Additive mode: each share summed (`MPA.sum`). Partition mode: within the partitions (`X.sum(1)`, `X.T.sum(0)`, a held row) the MPP's sum, a column split with the operand's ratios (a held row's sum stays with its holder); across them (`X.sum(0)`, `X.T.sum(1)`, a column) an additive-mode MPU whose shares are the compute parties' partial sums: in the clear if every partition is in the clear (an empty partition's share is zeros), otherwise encrypted at every party, so its shares are never in the clear at some parties and encrypted at others. One agreement round among the compute parties; call it at every party |
| `.dot(axis)` | Additive mode: `MPA.dot`. Partition mode: `MPP.dot`, `(x * x)` summed as `MPP.sum` sums (across the partitions a row held by the hub, not additive shares) |
| `.via_mpc(fn, *args)` | Execute via [MPC layer](../user-guide/switching.md). Additive mode: shares in the clear are shared as they are; otherwise the aggregate is switched (one built in the call from shares in one layout from each party's own copy, the same at every party; one held already from the hub's copy), so an aggregate whose copies came apart (each party's refreshed in a round of its own, as `clip` of a tall additive MPU leaves it) still switches to its value; `to_sharetensor()` likewise |
| `.sign(*args)` | Secure sign function |
| `.expand_dims(axis=0)` | Insert dimension |
| `.extend(other)` | Concatenate with compatible MPU |
| `.pad_with_value(val, size, axis)` | Append `size` columns of `val`, as the MPP or MPA beneath does |
| `.erase_element(index)` | Remove element at index |
| `.filter(mask)` | Boolean mask filtering |
| `.hstack(other)` | Horizontal concatenation |
| `.astype(T)` | Cast to different dtype |
| `.to_fp()` | Convert to floating point |
| `.copy()` | Deep copy |
| `.zeros(...)` | Zero tensor (same or specified shape) |
| `.ones(...)` | Ones tensor: additive mode as `MPA.ones()` (the hub's share), partition mode every partition ones (a transposed MPU's too) |
| `.rand(distribution, ...)` | Random values |
| `.get_matmul_cost(other)` | Estimate matmul cost |
| `.getitem_local(index)` | Local indexing without communication |
| `.slice_local(i, ...)` | Local slicing |
| `.rotate_local(i, ...)` | Local rotation |

## See also

- [MPP](mpp.md) — The partitioned representation used when `MPU` is constructed with `"partition"`
- [MPA](mpa.md) — The aggregated representation used when `MPU` is constructed with `"additive"`
- [MPC ↔ MHE Switching](../user-guide/switching.md) — How `via_mpc` works under the hood
