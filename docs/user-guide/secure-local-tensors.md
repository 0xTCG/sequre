# Secure Local Tensors (Ciphertensor)

`Ciphertensor` sits one layer below `MPU` and represents a **local tensor of CKKS ciphertexts** on a single machine. It provides operator overloading, automatic shape tracking, and encoding-mode awareness.

## What is a Ciphertensor?

A `Ciphertensor[ctype]` wraps a `list[ctype]` — where `ctype` is either `Ciphertext` or `Plaintext` — along with metadata:

| Field | Purpose |
|---|---|
| `_data` | List of ciphertext/plaintext chunks |
| `shape` | Logical tensor dimensions |
| `slots` | Number of CKKS slots per chunk (set by ring degree) |
| `_transposed` | Layout flag |
| `_diagonal_contiguous` | Whether data uses diagonal packing |

A matrix of shape `[m, n]` is chunked into `ceil(m * n / slots)` ciphertexts, each holding up to `slots` values.

## Construction

You typically don't construct `Ciphertensor` directly. It is created by:

1. **MHE encryption**: `mpc.mhe.enc_vector[Ciphertext](values)` returns a list of ciphertexts that you wrap in a `Ciphertensor`.
2. **MPP/MPA internals**: When an `MPU` operation needs encryption, it creates `Ciphertensor` objects automatically.

## Arithmetic

Operators dispatch to the CKKS evaluator:

```python
c = ct_a + ct_b      # Ciphertext-ciphertext addition
c = ct_a * ct_b      # Ciphertext-ciphertext multiply + relinearization
c = ct_a + pt        # Ciphertext-plaintext addition (cheaper)
c = ct_a * pt        # Ciphertext-plaintext multiply (cheaper)
```

### Cost hierarchy

| Operation | Approx. cost |
|---|---|
| CT + CT add | ~0.3 ms |
| CT × PT multiply | ~1.4 ms |
| CT × CT multiply + relin | ~34.4 ms |
| Rotation (Galois) | ~32.9 ms |

Plaintext operations are dramatically cheaper — the `@mhe_cipher_opt` compiler pass automatically reorders expressions to maximize plaintext ops.

## Encoding modes

For matrix operations, how values are packed into ciphertext slots matters for both correctness and performance:

| Mode | Constant | Description |
|---|---|---|
| Row-wise | `ENC_ROW` | One row per ciphertext |
| Column-wise | `ENC_COL` | One column per ciphertext |
| Diagonal | `ENC_DIAG` | Diagonal packing for efficient matrix-vector products |

The `@mhe_enc_opt` compiler pass selects the optimal encoding per matrix multiplication via brute-force search.

## Multiparty synergy

Although `Ciphertensor` represents data on a single machine, certain operations transparently involve collective protocols:

- **Relinearization** after multiplication uses the collective relinearization key
- **Bootstrapping/Refresh** when noise budget runs low involves all parties
- **E2S / S2E** (Encryption-to-Shares / Shares-to-Encryption) converts between HE and MPC representations

This is why `Ciphertensor` assumes a multiparty setup even though it's conceptually a "local" type.

## Next steps

- **[Ciphertensor API](../api/ciphertensor.md)** — Complete reference.
- **[Core MHE Module](../deep-dive-shechi/core-mhe.md)** — The collective protocols that support Ciphertensor.
- **[Lattiseq Overview](../deep-dive-lattiseq/overview.md)** — The CKKS engine underneath.
