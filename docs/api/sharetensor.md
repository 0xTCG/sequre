# Sharetensor

_Defined in `stdlib/sequre/types/sharetensor.codon`_

`Sharetensor[TP]` is the fundamental secure data type in Sequre's additive secret sharing layer. It represents a value (scalar, vector, or matrix) that has been split into additive shares distributed across computing parties.

## Type parameter

| Parameter | Description |
|---|---|
| `TP` | The underlying unsigned integer type, typically `mpc_uint` (`UInt[192]` by default) |

## Fields

| Field | Type | Description |
|---|---|---|
| `share` | `TP` | The party's additive share of the secret value |
| `x_r` | `TP` | Beaver triple reconstruction component |
| `r` | `TP` | Random mask component |
| `modulus` | `mpc_uint` | The modulus for arithmetic (field or ring) |
| `sqrt` | `TP` | Cached square root (when computed) |
| `sqrt_inv` | `TP` | Cached inverse square root |
| `fp` | `bool` | Whether this tensor is in fixed-point representation |
| `public` | `bool` | Whether this value is public (known to all parties) |
| `diagonal` | `bool` | Whether this is a diagonal matrix |

## Construction

### `Sharetensor.enc(mpc, value, ...)`

Secret-share a plaintext value. The trusted dealer (party 0) distributes shares to all parties. Overloads accept an optional `source_pid` and `modulus`.

```python
from sequre import Sharetensor as Stensor

a = Stensor.enc(mpc, 42)        # Share an integer
v = Stensor.enc(mpc, [1, 2, 3]) # Share a vector
```

### Factory methods

| Method | Description |
|---|---|
| `Sharetensor.enc(mpc, data)` | Secret-share plaintext data from the dealer |
| `Sharetensor.enc(mpc, data, source_pid, modulus)` | Secret-share from a specific party under a given modulus |
| `Sharetensor.zeros(size, modulus)` | Zero-filled vector |
| `Sharetensor.zeros(rows, cols, modulus)` | Zero-filled matrix |
| `Sharetensor.zeros(shape, modulus)` | Zero-filled tensor from a shape tuple |
| `Sharetensor.ones(shape, mpc, modulus)` | Ones-filled fixed-point matrix |
| `Sharetensor.rand(shape, distribution, mpc)` | Random fixed-point matrix (default modulus) |
| `Sharetensor.rand(shape, distribution, mpc, modulus)` | Random fixed-point matrix (explicit modulus) |
| `Sharetensor.range(start, stop, modulus)` | Public range vector |
| `Sharetensor.collective_load(mpc, path, rows, cols, binary)` | Load partitioned matrix from files across parties |

## Properties and state queries

| Property / Method | Type | Description |
|---|---|---|
| `.ndim` | `int` | Number of dimensions (0 for scalars) |
| `.size` | `int` | Total element count |
| `.shape` | `list[int]` | Tensor shape |
| `.cohort_shape` | `list[int]` | Alias for shape |
| `.partition_shape` | `list[int]` | Alias for shape |
| `.T` | `Sharetensor` | Transpose |
| `.I` | `Sharetensor` | Identity matrix |
| `.is_fp()` | `bool` | Fixed-point representation? |
| `.is_public()` | `bool` | Known to all parties? |
| `.is_empty()` | `bool` | Contains no elements? |
| `.is_partitioned()` | `bool` | Beaver partitions set? |

## Revealing

### `.reveal(mpc) -> plaintext`

Reconstruct the secret by combining shares from all parties. Returns the original plaintext value.

```python
result = (a + b).reveal(mpc)
print(result)  # 55
```

### `.publish(mpc) -> Sharetensor`

Reveals in-place, stores the result, and sets `public = True`. Returns `self`.

## Arithmetic

All standard operators are supported and are automatically rewritten by the `@sequre` compiler pass:

| Operation | Syntax | Protocol |
|---|---|---|
| Addition | `a + b` | Local (no communication) |
| Subtraction | `a - b` | Local |
| Multiplication | `a * b` | Beaver triple protocol |
| Matrix multiply | `a @ b` | Beaver triple |
| Division | `a / b` | Fixed-point iterative protocol |
| Power | `a ** n` | Repeated multiplication |
| Comparison | `a == b`, `a > b`, `a < b` | Bit decomposition protocol |
| Dot product | `a.dot(mpc, b, axis=0)` | Beaver inner product |

In-place variants (`+=`, `-=`, `*=`, `**=`) are also supported.

## Fixed-point support

Sequre uses fixed-point encoding to represent real numbers inside the integer secret-sharing ring. Truncation after multiplication is handled **automatically** by the runtime — whenever both operands of `*`, `@`, `dot`, or `/` are in fixed-point mode (`fp = True`), the result is truncated back to `f` fractional bits before it is returned.

```python
a_fp = a.to_fp()            # Convert to fixed-point representation
result = a_fp * b_fp         # Truncation happens automatically inside mul
plain = result.reveal(mpc)   # Reveals as floating-point
```

Manual truncation is available via `.trunc(mpc.fp)` / `.itrunc(mpc)` for advanced use cases (e.g. custom polynomial evaluation), but is not needed for standard arithmetic.

## Shape and data manipulation

| Method | Description |
|---|---|
| `.expand_dims(axis=0)` | Insert a new axis |
| `.expand_values(n)` | Broadcast to `n` copies |
| `.reshape(shape)` | Reshape the tensor |
| `.flatten()` | Flatten to 1-D |
| `.reverse()` | Reverse along the first axis (in-place) |
| `.filter(mask)` | Boolean mask filtering |
| `.erase_element(index)` | Remove element at index from a vector |
| `.append(other)` | Append another sharetensor as a single element |
| `.extend(other)` | Concatenate elements of another sharetensor |
| `.hstack(other)` | Horizontal stack (side by side) |
| `.vstack(other)` | Vertical stack (top to bottom) |
| `.pad(rows, cols)` | Pad matrix with zeros |
| `.pad_right(size)` | Extend vector with zeros on the right |
| `.pad_left(size)` | Prepend zeros to vector |
| `.pad_with_value(val, size, axis, mpc)` | Pad along axis with a specific value |
| `.sum(axis=0, keepdims=False)` | Sum along axis |
| `.T` | Transpose |

## Conversion

| Method | Description |
|---|---|
| `.to_fp()` | Convert to fixed-point representation |
| `.to_ring(mpc)` | Convert from field to ring |
| `.to_field(mpc)` | Convert from ring to field |
| `.to_bits(mpc, bitlen, delimiter_prime)` | Bit decomposition |
| `.diag(other)` | Create diagonal matrix with values from `other` |
| `.diagonal_contig(antidiagonal=False)` | Extract diagonal elements into a contiguous vector |
| `.diagonal_transpose(skinny)` | Specialized diagonal matrix transpose |
| `.copy()` | Deep copy |

## Partition management (Beaver triples)

| Method | Description |
|---|---|
| `.set_partitions(partitions)` | Store Beaver partitions `(x_r, r)` |
| `.get_partitions()` | Return existing `(x_r, r)` tuple |
| `.get_partitions(mpc, force=False)` | Compute and return Beaver partitions |
| `.validate_partitions(mpc, message="")` | Assert partitions reconstruct to the share |
| `.beaver_reveal(mpc)` | Reveal Beaver partitions as fixed-point floats |

## Truncation

| Method | Description |
|---|---|
| `.trunc(fp, k, m)` | Fixed-point truncation with precision params |
| `.itrunc(mpc, k, m)` | In-place truncation |

## I/O

| Method | Description |
|---|---|
| `Sharetensor.read_beaver_vector(mpc, f, f_mask, length, modulus, binary)` | Read Beaver-masked vector shares from files |
| `Sharetensor.read_beaver_matrix(mpc, f, f_mask, rows, cols, modulus, binary)` | Read Beaver-masked matrix shares from files |
| `Sharetensor.read_filtered_matrix(mpc, f, f_mask, imask, jmask, ...)` | Read filtered matrix rows/columns by index masks |

## Indexing and iteration

| Method | Description |
|---|---|
| `len(s)` | Length of first dimension |
| `bool(s)` | Nonzero check |
| `int(s)` | Cast share to `int` |
| `s[i]` | Slice at index |
| `s[i] = v` | Set slice at index |
| `for row in s` | Iterate along first axis |
