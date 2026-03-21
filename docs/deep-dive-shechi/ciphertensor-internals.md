# Ciphertensor Internals

`Ciphertensor[ctype]` is the typed container for CKKS ciphertexts (or plaintexts) in Sequre. It tracks shape, slot packing, and layout metadata so that higher-level types (`MPP`, `MPA`, `MPU`) can perform tensor algebra on encrypted data without manual bookkeeping.

## Data layout

```python
class Ciphertensor[ctype]:
    _data: list[ctype]              # Flat list of ciphertext/plaintext chunks
    shape: list[int]                # Logical tensor dimensions, e.g. [100, 50]
    slots: int                      # CKKS slots per chunk (= N/2 for ring degree N)
    _chunk_size: int                # Computed: prod(shape[:-1]) * ceil(shape[-1]/slots)
    _transposed: bool               # Logical transpose flag (no data movement)
    _diagonal_contiguous: bool      # Whether data is in diagonal packing order
    _skinny: bool                   # Whether the "short" dimension was moved first
    _is_broadcast: bool             # Whether all parties hold the same data
```

### Chunk mapping

A matrix of shape `[m, n]` with `S` slots per ciphertext is stored as:

```
Row-wise: m rows × ceil(n/S) ciphertexts = m * ceil(n/S) total chunks
```

The `cipher_shape` property computes `[m, ceil(n/S)]`, reflecting the ciphertext-level grid.

### Transposition

`ct.T` returns a new `Ciphertensor` with `_transposed` flipped. **No data is moved** — the flag tells downstream operations to swap dimension interpretation. This matches NumPy's lazy-transpose semantics.

## Encoding modes

The `enc` static method selects between three strategies:

| Mode | Constant | When selected | Packing |
|---|---|---|---|
| Row-wise | `ENC_ROW` | `rows <= cols` (default) or 1-D | One row per ciphertext (or multi-cipher for wide rows) |
| Column-wise | `ENC_COL` | `rows > cols` | One column per ciphertext |
| Diagonal | `ENC_DIAG` | Set by `@mhe_enc_opt` pass | Diagonal `d_k` of the matrix per ciphertext |

```python
@staticmethod
def enc(mpc, data, padding=0, encoding=""):
    if not encoding:
        encoding = mpc.default_ciphertensor_encoding
    if not encoding:
        encoding = ENC_ROW if data.shape[0] <= data.shape[-1] else ENC_COL
    ...
```

### Diagonal packing

For a matrix $A \in \mathbb{R}^{m \times n}$, the $k$-th diagonal is:

$$d_k[i] = A[i, (i + k) \bmod n], \quad i = 0, \ldots, m-1$$

This layout enables matrix-vector multiplication via rotate-and-sum:

$$A \cdot v = \sum_{k=0}^{n-1} d_k \odot \text{rot}(v, k)$$

which costs $n$ rotations + $n$ plaintext multiplies (or ciphertext multiplies), compared to $m \times n$ multiplies in naive row packing.

## Key operations

### Indexing

`__getitem__` and `__setitem__` raise `NotImplementedError` at the library level — they require the **IR pass** to rewrite into actual ciphertext-level operations (masking, rotation, selection). This is by design: the compiler knows the access pattern at compile time and can emit optimal code.

### Arithmetic stubs

All arithmetic operators (`__add__`, `__mul__`, `__matmul__`, etc.) also raise `NotImplementedError`. The IR expressiveness pass rewrites these into calls to `mpc.mhe.iadd`, `mpc.mhe.imul`, etc. This separation keeps the library type-checkable while letting the compiler insert refresh/bootstrap/protocol-switch logic.

### Serialization

`Ciphertensor` supports Codon's `pickle`/`unpickle` protocol for network communication:

```
pickle order: _skinny → _diagonal_contiguous → _transposed → slots → shape → _data
```

This is used by `MPCComms` when sending encrypted tensors between parties.

### Helper methods

| Method | Description |
|---|---|
| `copy(shallow=False)` | Deep or shallow copy |
| `actual_shape` | Shape after applying transpose and skinny flags |
| `cipher_shape` | Shape in ciphertext units |
| `size` / `ndim` | Total elements / number of dimensions |
| `T` | Lazy transpose |
| `I(mpc)` | Encrypted identity matrix |
| `diagonal(idx)` | Extract a diagonal (requires `_diagonal_contiguous`) |
| `mask(mpc, i)` | One-hot mask on a 1-D ciphertensor |
| `getitemdup(mpc, i, new_size)` | Extract element `i` and duplicate across `new_size` slots |
| `aggregate(mpc)` | Collect and sum across all parties |

## Cost model

The compiler's `@mhe_cipher_opt` pass uses constant cost estimates to decide operation ordering:

| Operation | Cost constant |
|---|---|
| CT × CT multiply | `HE_MUL_COST_ESTIMATE` |
| Rotation | `HE_ROT_COST_ESTIMATE` |
| Encryption | `HE_ENC_COST_ESTIMATE` |
| MHE ↔ MPC switch | `MHE_MPC_SWITCH_COST_ESTIMATE` |
| MPC big-int multiply | `MPC_BIG_INT_MUL_COST` |

These live in `stdlib/sequre/constants.codon`.

## Next steps

- **[Core MHE Module](core-mhe.md)** — The collective protocols that Ciphertensor relies on.
- **[Ciphertensor API](../api/ciphertensor.md)** — Public interface reference.
- **[CKKS Operations](../deep-dive-lattiseq/ckks-operations.md)** — What happens inside each evaluator call.
