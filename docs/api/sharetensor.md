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

Secret-share a plaintext value. The trusted dealer (party 0) distributes shares to all parties.

```python
from sequre import Sharetensor as Stensor

a = Stensor.enc(mpc, 42)        # Share an integer
v = Stensor.enc(mpc, [1, 2, 3]) # Share a vector
```

### `Sharetensor.zeros(n, modulus)`

Create a zero-filled public Sharetensor of length `n`.

### `Sharetensor.range(start, stop, modulus)`

Create a Sharetensor containing a range of public values.

## Revealing

### `.reveal(mpc) -> plaintext`

Reconstruct the secret by combining shares from all parties. Returns the original plaintext value.

```python
result = (a + b).reveal(mpc)
print(result)  # 55
```

## Arithmetic

All standard operators are supported and are automatically rewritten by the `@sequre` compiler pass:

| Operation | Syntax | Protocol |
|---|---|---|
| Addition | `a + b` | Local (no communication) |
| Subtraction | `a - b` | Local |
| Multiplication | `a * b` | Beaver triple protocol |
| Matrix multiply | `a @ b` | Beaver triple + Strassen |
| Division | `a / b` | Fixed-point iterative protocol |
| Power | `a ** n` | Repeated multiplication |
| Comparison | `a == b`, `a > b`, `a < b` | Bit decomposition protocol |
| Dot product | `a.dot(mpc, b, axis=0)` | Beaver inner product |

## Fixed-point support

Sequre uses fixed-point encoding to represent real numbers inside the integer secret-sharing ring. Truncation after multiplication is handled **automatically** by the runtime — whenever both operands of `*`, `@`, `dot`, or `/` are in fixed-point mode (`fp = True`), the result is truncated back to `f` fractional bits before it is returned.

```python
a_fp = a.to_fp()            # Convert to fixed-point representation
result = a_fp * b_fp         # Truncation happens automatically inside mul
plain = result.reveal(mpc)   # Reveals as floating-point
```

Manual truncation is available via `.trunc(mpc.fp)` / `.itrunc(mpc)` for advanced use cases (e.g. custom polynomial evaluation), but is not needed for standard arithmetic.

## Additional methods

| Method | Description |
|---|---|
| `.expand_values(n)` | Broadcast to `n` copies |
| `.get_partitions(mpc)` | Distribute Beaver partitions |
| `.filter(mask)` | Boolean mask filtering |
| `.T` | Transpose |
| `.sum()` | Sum elements |
| `.flatten_numpy()` | Flatten to 1-D |
| `.reshape_numpy(shape)` | Reshape |
