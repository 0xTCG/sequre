# Ciphertensor

_Defined in `stdlib/sequre/types/ciphertensor.codon`_

`Ciphertensor[ctype]` wraps a list of CKKS ciphertexts (or plaintexts) into a tensor abstraction with shape tracking, operator overloading, and automatic encoding-mode awareness.

## Type parameter

| Parameter | Typical value | Description |
|---|---|---|
| `ctype` | `Ciphertext` or `Plaintext` | The underlying CKKS data type from Lattiseq |

## Fields

| Field | Type | Description |
|---|---|---|
| `_data` | `list[ctype]` | The underlying list of ciphertext/plaintext chunks |
| `shape` | `list[int]` | Logical tensor shape |
| `slots` | `int` | Number of CKKS slots per ciphertext (determined by ring degree) |
| `_chunk_size` | `int` | Elements per chunk |
| `_transposed` | `bool` | Whether the tensor is in transposed layout |
| `_diagonal_contiguous` | `bool` | Whether data is stored in diagonal order |
| `_skinny` | `bool` | Whether the matrix is "skinny" (rows >> cols) |
| `_is_broadcast` | `bool` | Whether every computing party holds the same ciphertexts (aggregates, trivially encrypted public data). Refreshing such a tensor bootstraps each ciphertext once, collectively, and keeps the copies identical |

## Construction

`Ciphertensor` objects are created via the `.enc` family of factory methods:

```python
# Encrypt a 2-D ndarray (default encoding chosen automatically)
ct = Ciphertensor.enc(mpc, my_array)

# Encrypt with explicit encoding
ct = Ciphertensor.enc(mpc, my_array, encoding=ENC_DIAG)

# Encrypt in row-wise encoding with padding
ct = Ciphertensor.enc_row_wise(mpc, my_array, padding=16)
```

### Factory methods

| Method | Description |
|---|---|
| `Ciphertensor.enc(mpc, data, padding=0, encoding="")` | Encrypt and encode plaintext data into a ciphertensor |
| `Ciphertensor.enc_public(mpc, data, padding=0, encoding="")` | Encode public data (the same at every party) in `enc`'s layout as trivial encryptions `(Encode(v), 0)`: identical at every party, noiseless, no public-key encryption. Private data needs `enc` |
| `Ciphertensor.enc_row_wise(mpc, data, padding=0)` | Encrypt in row-wise encoding |
| `Ciphertensor.enc_diag_wise(mpc, data, padding=0)` | Encrypt 2-D data in diagonal-contiguous encoding (a tall matrix needs its column count to divide its row count) |
| `Ciphertensor.enc_replicate(mpc, value, shape)` | Create ciphertensor with a scalar replicated across shape |
| `Ciphertensor.enc_alpern(mpc, data)` | Encode 3-D arrays in Alpern order |
| `Ciphertensor.zeros(mpc, shape)` | Zero-initialized ciphertensor with given shape |
| `Ciphertensor.placeholder(shape, slots)` | Placeholder with uninitialized data |
| `Ciphertensor.like(other)` | Empty ciphertensor with same properties as `other` |

## Properties

| Property | Type | Description |
|---|---|---|
| `.shape` | `list[int]` | Logical tensor shape |
| `.slots` | `int` | CKKS slots per ciphertext |
| `.actual_shape` | `list[int]` | Shape accounting for transposition and diagonal layout |
| `.cipher_shape` | `list[int]` | Shape in ciphertext counts (last dim packed by slots) |
| `.size` | `int` | Total element count |
| `.ndim` | `int` | Number of dimensions |
| `.T` | `Ciphertensor` | Lazy transpose (metadata flip only) |
| `.level()` | `int` | Current noise-growth level |

## Arithmetic

All operators dispatch to the CKKS evaluator:

| Operation | HE operation | Cost estimate |
|---|---|---|
| `a + b` | Ciphertext-ciphertext add | ~0.3 ms |
| `a * b` | Ciphertext-ciphertext multiply + relin | ~34.4 ms |
| `a + plain` | Ciphertext-plaintext add | ~0.3 ms |
| `a * plain` | Ciphertext-plaintext multiply | ~1.4 ms |
| Rotation | Galois automorphism | ~32.9 ms |

### Operators

| Operator | Description |
|---|---|
| `a + b` | Element-wise addition |
| `a - b` | Element-wise subtraction |
| `a * b` | Element-wise multiplication (+ relin) |
| `a @ b` | Matrix multiplication (adaptive algorithm selection) |
| `a ** n` | Exponentiation |
| `-a` | Negation |
| `a > b`, `a < b` | Comparisons (via compiler IR pass) |

### Functional arithmetic

These methods take `mpc` as the first argument and return a new ciphertensor:

| Method | Description |
|---|---|
| `.neg(mpc)` | Element-wise negation |
| `.add(mpc, other)` | Element-wise addition |
| `.sub(mpc, other)` | Element-wise subtraction |
| `.mul(mpc, other, ...)` | Element-wise multiplication |
| `.pow(mpc, p)` | Element-wise exponentiation |

### In-place arithmetic

| Method | Description |
|---|---|
| `.iadd(mpc, other)` | In-place addition |
| `.isub(mpc, other)` | In-place subtraction |
| `.imul(mpc, other, no_refresh=False)` | In-place multiplication (optional refresh suppression) |
| `.ipow(mpc, p)` | In-place exponentiation |
| `.ineg(mpc)` | In-place negation |
| `.irotate(mpc, step)` | In-place homomorphic rotation |

## Matrix and vector operations

| Method | Description |
|---|---|
| `.matmul(mpc, other)` | Matrix multiplication with adaptive MHE/MPC algorithm selection |
| `.dot(mpc, other, axis)` | Dot product along axis |
| `.dot(mpc, axis)` | Self dot product along axis |
| `.aggregate(mpc)` | Aggregate distributed shares to party 0 |
| `.sum(mpc, axis)` | Sum reduction along axis |
| `.reduce_add(mpc, keep_dims=True)` | Sum all elements of a 1-D tensor |
| `.reduce_add_tiled(mpc, tile_size)` | Tiled sum reduction for 1-D/2-D tensors |

!!! note
    `matmul` automatically selects between pure-MHE and [MPC ↔ MHE switching](../user-guide/switching.md) paths based on a cost estimator when `AllowMPCSwitch` is active.

### Matrix products

`a @ b` (or `a.matmul(mpc, b)`) picks a best matrix multiplication algorithm (kernel) based on the operands' layouts and shapes. With A of shape p x m, B of shape m x q, and S slots per ciphertext:

| A | B | Kernel | Shapes |
|---|---|---|---|
| row-wise | row-wise | M2 | any |
| row-wise | column-wise | M1 | any |
| column-wise | column-wise | M2 on the transposes | any |
| column-wise | row-wise (ciphertensor or plaintext) | TNT, diagonal-encoded result | max(p, q) + min(p, q) - 1 <= S, and q divides p when q < p |
| column-wise | row-wise ciphertensor, outside TNT's shapes | M2 reading A by columns | any |
| row-wise | diagonal-encoded | M3 | max(m, q) <= S; max(m, q) + min(m, q) - 1 <= S unless m divides S; q divides m when q < m, and m divides q when m < q for a lazily transposed B |
| diagonal-encoded | diagonal-encoded | M3 | each of p, m, q divides or is divisible by the others; max(p, m, q) <= S; no tall A (p > m > 1) times wide B with q < p; 2p <= S when A is wide and q < p |
| diagonal-encoded | not diagonal-encoded | none | not implemented |
| column-wise | diagonal-encoded | none | not implemented |

With a plaintext `ndarray` B, the dispatcher estimates the cost of each kernel whose shape constraints hold and runs the cheapest: M1, M2, M3 (B encoded diagonal-wise) and BSGS (a square B of at most S/2 rows) for a row-wise A, plus via-MPC under `mpc.allow_mpc_switch()`; TNT and M2 on the transposes for a column-wise A; M3 for a diagonal-encoded A. `ndarray @ Ciphertensor` runs as `(b.T @ a.T).T`. The choice depends on public data only (shapes, layouts, slots and `mpc.default_allow_mpc_switch`), so every party takes the same one. Run the product at every party, the trusted dealer included: it takes part in the via-MPC path, and otherwise gets a placeholder with the result's shape and layout.

## Encoding modes

The `@mhe_enc_opt` compiler pass selects the optimal encoding strategy per matrix multiplication:

| Mode | Constant | Description |
|---|---|---|
| Row-wise | `ENC_ROW` | Each ciphertext holds one row |
| Column-wise | `ENC_COL` | Each ciphertext holds one column |
| Diagonal | `ENC_DIAG` | Diagonal packing for matrix-vector products |

## Shape manipulation

| Method | Description |
|---|---|
| `.expand_dims(axis=0)` | Insert a dimension of size 1 |
| `.pad(mpc, new_size)` | Pad 1-D tensor with zeros |
| `.pad_with_value(val, size, axis, mpc)` | Pad with a specific value |
| `.resize(mpc, shape)` | Resize 1-D tensor to new shape |
| `.concat(mpc, other, axis)` | Concatenate along axis (returns new tensor) |
| `.iconcat(mpc, other, axis)` | In-place concatenation |
| `.extend(mpc, other)` | Extend tensor with another (1-D or 2-D) |
| `.append(other)` | Append rows to 2-D tensor |
| `.pop()` | Remove last row |
| `.filter(mask)` | Boolean mask filtering |
| `.replicate(mpc, new_size)` | Extend 1-D tensor by recursive patch copying |
| `.local_broadcast(mpc, target_shape)` | Broadcast to target shape locally |

## Encoding and layout

| Method | Description |
|---|---|
| `.rotate(mpc, step)` | Homomorphic rotation (returns new tensor) |
| `.shift(mpc, step)` | Shift 1-D tensor with zero fill and shape expansion |
| `.diagonal(idx)` | Extract diagonal from diagonal-contiguous 2-D tensor |
| `.diagonal_contig(mpc)` | Convert to diagonal-contiguous encoding |
| `.diagonal_transpose(mpc)` | Transpose diagonal-contiguous layout |
| `.mask(mpc, i)` | One-hot mask for element `i` of a 1-D tensor |
| `.getitemdup(mpc, i, new_size)` | Duplicate a single element to fill a new size |
| `.actual_transpose(mpc)` | Materialize actual transpose (vs. lazy `.T`) |
| `.get_actual(mpc)` | Resolve pending transposition/diagonality |
| `.iget_actual(mpc)` | In-place `get_actual` |
| `.I(mpc)` | Identity matrix ciphertensor (requires 2-D) |

## Conversion and decryption

| Method | Description |
|---|---|
| `.copy(shallow=False)` | Deep or shallow copy |
| `.astype(T)` | Type conversion (currently identity) |
| `.decrypt(mpc, source_pid=-2)` | Collective decryption to plaintext. `-2` (default): every compute party gets its own tensor decrypted and learns nothing of the others' (call it at every compute party, even if it's tensor is empty); `-1`: a tensor that is the same at every compute party. See [`MPCMHE.decrypt`](mpc-instance.md) |
| `.decode(mpc)` | Decode plaintext tensor to `ndarray` |
| `.reveal(mpc, source_pid=-2)` | Decrypt and decode in one step (same `source_pid` modes as `.decrypt`) |
| `.via_mpc(mpc, fn, *args)` | Execute a function through the [MPC layer](../user-guide/switching.md) (E2S → fn → S2E) |

## Indexing and iteration

| Method | Description |
|---|---|
| `ct[i]` | Index / slice (requires compiler IR pass) |
| `ct[i] = v` | Set element (requires compiler IR pass) |
| `for row in ct` | Iterate over rows (asserts not transposed, ndim > 1) |
| `bool(ct)` | `True` if the tensor contains data |

## Utility

| Method | Description |
|---|---|
| `.set(other)` | Replace all internal state from another tensor |
| `.validate()` | Assert internal consistency |
| `.get_matmul_cost(other)` | Cost estimate for algorithm selection |
| `Ciphertensor.requires_collective(a, b)` | Check if a collective op is needed (transposition mismatch) |
