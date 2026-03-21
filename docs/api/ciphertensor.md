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
| `_is_broadcast` | `bool` | Whether this tensor was produced by broadcasting |

## Construction

`Ciphertensor` objects are typically created by the MHE layer rather than directly:

```python
# Via MHE encryption
ct = mpc.mhe.enc_vector[Ciphertext](values)

# Or constructed from a list of ciphertexts
ct = Ciphertensor(data_list, shape=[100, 50], slots=4096)
```

## Arithmetic

All operators dispatch to the CKKS evaluator:

| Operation | HE operation | Cost estimate |
|---|---|---|
| `a + b` | Ciphertext-ciphertext add | ~0.3 ms |
| `a * b` | Ciphertext-ciphertext multiply + relin | ~34.4 ms |
| `a + plain` | Ciphertext-plaintext add | ~0.3 ms |
| `a * plain` | Ciphertext-plaintext multiply | ~1.4 ms |
| Rotation | Galois automorphism | ~32.9 ms |

## Encoding modes

The `@mhe_enc_opt` compiler pass selects the optimal encoding strategy per matrix multiplication:

| Mode | Constant | Description |
|---|---|---|
| Row-wise | `ENC_ROW` | Each ciphertext holds one row |
| Column-wise | `ENC_COL` | Each ciphertext holds one column |
| Diagonal | `ENC_DIAG` | Diagonal packing for matrix-vector products |

## Key methods

| Method | Description |
|---|---|
| `.copy()` | Deep copy the tensor |
| `.astype(T)` | Convert between `Ciphertext` and `Plaintext` |
| `.T` | Transpose (metadata flip) |
| `.__repr__()` | String representation with shape/slot info |
