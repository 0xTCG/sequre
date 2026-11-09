# MPA — Multiparty Aggregate

_Defined in `stdlib/sequre/types/multiparty_aggregate.codon`_

`MPA[S, dtype]` represents data where each party holds an **additive share** of the global value, optionally encrypted.

---

## Fields

| Field | Description |
|---|---|
| `_mpc` | Reference to the MPC instance |
| `_plain` | Plaintext share (ndarray) |
| `_encryption` | Per-party encrypted share (`Ciphertensor`) |
| `_aggregate` | Aggregated (summed) encrypted ciphertext |

## Construction

```python
from sequre.types.multiparty_aggregate import MPA

# From plaintext: every party's argument is its own share, and the MPA's value is their sum (CP0's is ignored)
mpa = MPA(mpc, my_plaintext_share)

# From encrypted data
mpa = MPA(mpc, my_ciphertensor)
```

## Factory methods

| Method | Description |
|---|---|
| `MPA.enc(mpc, data)` | MPA whose value is `data`: CP1's share is `data`, the others' zeros, all in the clear (`.encrypt()` encrypts them) |
| `MPA.rand(shape, distribution, mpc, plain=False)` | Random MPA, plaintext or encrypted |

## Properties and state queries

| Property / Method | Type | Description |
|---|---|---|
| `.shape` | `S` | Current tensor shape |
| `.partition_shape` | `list[int]` | Raw partition shape |
| `.ndim` | `int` | Number of dimensions |
| `.T` | `MPA` | Transpose |
| `.I` | `MPA` | Identity matrix |
| `.modulus` | `mpc_uint` | MPC modulus |
| `.has_plain()` | `bool` | Plaintext component present? |
| `.has_encryption()` | `bool` | Encrypted component present? |
| `.has_distributed()` | `bool` | Plaintext or encrypted present? |
| `.has_aggregate()` | `bool` | Aggregated component present? |
| `.is_plain()` | `bool` | Only plaintext? |
| `.is_distributed()` | `bool` | Plaintext/encrypted without aggregate? |
| `.is_aggregate()` | `bool` | Only aggregated? |
| `.is_empty()` | `bool` | No components? |

## Operators

| Operator | Description |
|---|---|
| `a + b` | Addition with smart component selection |
| `a - b` | Subtraction |
| `a * b` | Multiplication with lazy aggregation |
| `a @ b` | Matrix multiplication |
| `a / b` | Division by scalar |
| `a ** n` | Integer exponentiation |
| `-a` | Negation |
| `a > b` | Secure greater-than (uses [MPC switching](../user-guide/switching.md) internally) |
| `a < b` | Secure less-than (uses [MPC switching](../user-guide/switching.md) internally) |
| `a[i]`, `a[i] = v` | Indexing and slicing |
| `bool(a)` | Boolean conversion |
| `len(a)` | Length of first dimension |

## Reveal and encryption

| Method | Description |
|---|---|
| `.encrypt()` | Convert plaintext to encrypted component |
| `.aggregate()` | Aggregate plaintext/encrypted into aggregated component |
| `.reveal()` | The sum of the shares, to every compute party, and nothing else about them. Shares every compute party holds in the clear are sent under pairwise masks that cancel in the sum (fixed point with twice the MPC fraction bits for floats, 64 by default; ints exact); encrypted ones go through the aggregate |
| `.reveal_local()` | This compute party's own share: in the clear as it is, or an encrypted share decrypted to it alone (call it at every compute party). An MPA that holds only its aggregate, the same ciphertexts at every party, has no share of its own: revealing it reveals the sum, to every compute party |
| `.level()` | Minimum encryption level |

## Distributed operations

| Method | Description |
|---|---|
| `.sum(axis)` | Sum along axis |
| `.dot(axis)` | Dot product along axis |

## Shape manipulation

| Method | Description |
|---|---|
| `.expand_dims(axis=0)` | Insert dimension (1-D only) |
| `.pad_with_value(val, size, axis, ...)` | Pad with value |
| `.erase_element(index)` | Remove element by index |
| `.filter(mask)` | Boolean mask filtering |
| `.hstack(other)` | Horizontal concatenation |

## Protocol switching

| Method | Description |
|---|---|
| `.via_mpc(fn, *args)` | Execute function through the [MPC layer](../user-guide/switching.md) (converts to Sharetensor, runs fn, converts back) |
| `.sign(*args)` | Secure sign function |

!!! info
    Comparisons (`>`, `<`) automatically use `via_mpc` under the hood. See the [MPC ↔ MHE Switching](../user-guide/switching.md) page.

## Conversion and creation

| Method | Description |
|---|---|
| `.astype(T)` | Cast to different dtype |
| `.to_fp()` | Convert to floating point |
| `.copy()` | Deep copy all components |
| `.zeros(shape)` / `.zeros()` | Zero tensor |
| `.ones(...)` | Ones tensor |
| `.rand(distribution, ...)` | Random values |
| `.get_matmul_cost(other)` | Estimate matmul cost |
