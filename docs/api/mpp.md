# MPP — Multiparty Partition

_Defined in `stdlib/sequre/types/multiparty_partition.codon`_

`MPP[S, dtype]` represents data that is **horizontally partitioned** across parties. Each party holds a contiguous block of rows of the global matrix.

---

## Fields

| Field | Description |
|---|---|
| `_mpc` | Reference to the `MPCEnv` |
| `_ratios` | List of per-party row counts |
| `_local_data` | This party's local plaintext rows |
| `_encryption_unified` | Encrypted version of the local data (`Ciphertensor[Ciphertext]`) |

## Construction

```python
from sequre.types.multiparty_partition import MPP

# From local data (each party provides its own rows)
mpp = MPP(mpc, local_rows)

# From a Ciphertensor (each party provides encrypted local data)
mpp = MPP(mpc, local_ciphertensor)
```

!!! note
    Party 0 (trusted dealer) holds a zero-filled placeholder. Actual data resides on parties 1..N.

## Factory methods

| Method | Description |
|---|---|
| `MPP.enc(mpc, data, ratios=None)` | Create encrypted partition with optional ratios |
| `MPP.collective_load(mpc, path, rows, cols, binary)` | Load partitioned matrix from files across parties |
| `MPP.rand(shape, distribution, mpc)` | Random partition |
| `MPP.like(other, arr)` | Create MPP matching another's structure |
| `MPP.uniform_ratios(size, n_parties)` | Generate evenly distributed partition ratios |
| `MPP.apply_ratios(ratios, target_size, approximate)` | Scale ratios proportionally |

## Properties and state queries

| Property / Method | Type | Description |
|---|---|---|
| `.shape` | `S` | Current shape (transposition-aware) |
| `.cohort_shape` | `S` | Collective shape across all parties |
| `.shape_local` | `S` | Alias for shape |
| `.partition_shape` | `S` / `list[int]` | Raw partition shape |
| `.ndim` | `int` | Number of dimensions |
| `.T` | `MPP` | Lazy transpose |
| `.I` | `MPP` | Identity matrix partition |
| `.local` | `ndarray` | Local plaintext data (requires `is_local`) |
| `.modulus` | `mpc_uint` | MPC modulus |
| `.ratios` | `list[int]` | Partition size ratios |
| `.has_partial_encryption()` | `bool` | Left/right encrypted portions exist? |
| `.has_unified_encryption()` | `bool` | Full partition encrypted? |
| `.has_local_data()` | `bool` | Plaintext data present? |
| `.is_semi_encrypted()` | `bool` | Mixed plaintext/ciphertext? |
| `.is_full_encrypted()` | `bool` | Fully encrypted? |
| `.is_local()` | `bool` | Plaintext only? |
| `.is_empty()` | `bool` | No data present? |
| `.cohort_is_aligned()` | `bool` | All parties have equal partition size? |

## Operators

| Operator | Description |
|---|---|
| `a + b` | Secure addition |
| `a - b` | Secure subtraction |
| `a * b` | Secure multiplication |
| `a @ b` | Secure matrix multiplication |
| `a / b` | Division by scalar |
| `a ** n` | Integer exponentiation |
| `-a` | Negation |
| `a > b` | Secure greater-than (uses [MPC switching](../user-guide/switching.md) internally) |
| `a < b` | Secure less-than (uses [MPC switching](../user-guide/switching.md) internally) |
| `a[i]`, `a[i] = v` | Indexing and slicing |
| `bool(a)` | Non-empty check |
| `len(a)` | Length of first dimension |

## Reveal and encryption

| Method | Description |
|---|---|
| `.encrypt()` | Encrypt local plaintext to ciphertext |
| `.reveal()` | Reveal with secure aggregation to all parties |
| `.reveal_local()` | Reveal local partition only |
| `.level()` | Minimum encryption level |
| `.unify_encryption()` | Consolidate left/right encryptions to unified |

## Distributed operations

| Method | Description |
|---|---|
| `.collect()` | Collect encryptions from all parties |
| `.collect_and_execute_at(target_pid, op)` | Collective operation executed at target party |
| `.aggregate()` | Aggregate to aligned uniform partition |
| `.aggregate_at(target_pid)` | Aggregate all shares at specific party |
| `.join()` | Join encrypted partitions into unified encryption |
| `.join_at(target_pid)` | Join at specific target party |
| `.broadcast_from(source_pid)` | Broadcast partition from source party |
| `.broadcast(target)` | Broadcast single-row MPP to match target ratios |
| `.align_ratios(ratios, force)` | Realign partition ratios |
| `.sum(axis)` | Sum along axis with cross-party reduction |
| `.dot(axis)` | Dot product along axis |

## Shape manipulation

| Method | Description |
|---|---|
| `.expand_dims(axis=0)` | Insert dimension |
| `.extend(other)` | Concatenate partitions |
| `.pad_with_value(val, size, axis, ...)` | Pad axis with value |
| `.erase_element(index)` | Remove element by index |
| `.filter(mask)` | Boolean mask filtering |
| `.cohort_filter(mask)` | Filter across cohort with adjusted ratios |
| `.hstack(other)` | Horizontal concatenation |
| `.local_broadcast(target_shape)` | Broadcast single element to shape |
| `.replicate(new_size)` | Copy with size expansion |
| `.actual_transpose()` | Materialize actual transpose (vs. lazy `.T`) |
| `.actual_itranspose()` | In-place actual transpose |
| `.getitem_local(index)` | Local indexing on plaintext data |
| `.slice_local(idx, ...)` | Local slicing on transposed encrypted data |
| `.rotate_local(i, ...)` | Local rotation on transposed data |

## Protocol switching

| Method | Description |
|---|---|
| `.via_mpc(fn, *args)` | Execute function through the [MPC layer](../user-guide/switching.md) (converts to Sharetensor, runs fn, converts back) |
| `.sign(*args)` | Secure sign function |

!!! info
    Comparisons (`>`, `<`) and certain `matmul` cases automatically use `via_mpc` under the hood. See the [MPC ↔ MHE Switching](../user-guide/switching.md) page.

## Conversion and creation

| Method | Description |
|---|---|
| `.astype(T)` | Cast to different dtype |
| `.to_fp()` | Convert to floating point |
| `.copy()` | Deep copy all components |
| `.set(other)` | Copy state from another MPP |
| `.zeros(shape)` / `.zeros()` | Zero tensor |
| `.ones(...)` | Ones tensor |
| `.rand(distribution, ...)` | Random partition |
| `.get_matmul_cost(other)` | Estimate matmul cost |
| `.get_relative_indices()` | Local partition start/end indices |
