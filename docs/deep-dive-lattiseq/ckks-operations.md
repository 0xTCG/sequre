# CKKS Operations

This page covers the CKKS (Cheon-Kim-Kim-Song) scheme as implemented in `stdlib/sequre/lattiseq/ckks.codon`. CKKS enables approximate arithmetic on encrypted complex (or real) vectors.

## CKKS in a nutshell

CKKS encodes a vector of up to $N/2$ complex numbers into a polynomial in $R_Q = \mathbb{Z}_Q[X]/(X^N+1)$, then encrypts it under RLWE:

$$\text{ct} = (c_0, c_1) = (a \cdot s + m + e, \;-a)$$

where $s$ is the secret key, $a$ is uniform random, $m$ is the encoded message, and $e$ is Gaussian noise.

Decryption recovers $c_0 + c_1 \cdot s = m + e$ — the message plus a small noise term.

## Encode / Decode

`EncoderComplex128` maps between `list[complex]` and `Plaintext`.

### Encoding

```python
encoder.encode(values: list[complex], plaintext: Plaintext, log_slots: int)
```

1. Apply the inverse canonical embedding (IDFT on roots of unity)
2. Scale by $\Delta$ (the `default_scale`) to preserve precision
3. Round to integers and place into the polynomial coefficients

### Decoding

```python
values = encoder.decode(plaintext: Plaintext, log_slots: int) -> list[complex]
```

1. Read polynomial coefficients
2. Divide by the current scale
3. Apply the canonical embedding (DFT on roots of unity)

The encoding uses 128-bit complex arithmetic internally for precision.

## Encrypt / Decrypt

### Public-key encryption (`PkEncryptor`)

```python
ciphertext = encryptor.encrypt_new(plaintext) -> Ciphertext
```

Generates a fresh RLWE sample using the public key and adds the plaintext.

### Secret-key decryption (`Decryptor`)

```python
plaintext = decryptor.decrypt_new(ciphertext) -> Plaintext
```

Computes $c_0 + c_1 \cdot s$ to recover the noisy plaintext.

In the multiparty setting, each party computes a **partial decryption** using its `sk_shard`, and the hub aggregates them.

## Evaluator operations

The `Evaluator` holds the `EvaluationKey` (relinearization + rotation keys) and performs:

### Addition

```python
evaluator.add(ct_a, ct_b, ct_out)
```

Coefficient-wise addition of the two ciphertexts. **Noise grows additively.** This is essentially free.

### Scalar operations

```python
evaluator.add_const(ct, constant, ct_out)   # ct + Δ·constant
evaluator.mul_const(ct, constant, ct_out)   # ct × constant
```

`add_const` encodes the constant at the ciphertext's scale and adds. `mul_const` multiplies coefficients directly — the scale doubles.

### Multiplication

```python
evaluator.mul_relin(ct_a, ct_b, do_relin, ct_out)
```

1. **Tensor product**: $(c_0^A, c_1^A) \otimes (c_0^B, c_1^B)$ yields a degree-2 ciphertext $(d_0, d_1, d_2)$
2. **Relinearization** (if `do_relin=True`): uses the `RelinearizationKey` to reduce back to degree 1
3. Scale becomes $\Delta^2$ — must rescale afterward

### Rescale

```python
evaluator.rescale(ct, target_scale, ct_out)
```

Divides the ciphertext modulus by the last prime $q_L$, reducing the scale from $\Delta^2$ back to $\approx\Delta$. **Consumes one level.**

The noise budget is finite: after $L$ multiplications (where $L$ is the number of moduli), the ciphertext must be **bootstrapped** or computation must end.

### Rotation

```python
evaluator.rotate(ct, k, ct_out)
```

Applies a Galois automorphism $\sigma_{5^k}$ to cyclically rotate the plaintext slots by $k$ positions. Requires the corresponding Galois key from the `RotationKeySet`.

### Reduce-add

```python
evaluator.reduce_add(ct, size)
```

Sums the first `size` slots into slot 0 using a log-depth rotation tree. Used for inner products and reductions.

## Level and scale tracking

Each `Ciphertext` tracks:

| Property | Meaning |
|---|---|
| `level()` | Current number of remaining moduli (decreases after rescale) |
| `scale` | Current encoding scale (doubles after multiply, halves after rescale) |

The MPCMHE layer calls `refresh()` when `level() <= bootstrap_min_level`, triggering the collective bootstrap protocol.

## Distributed protocols (`dckks.codon`)

These extend single-party CKKS to the multiparty setting:

| Protocol | Function | Purpose |
|---|---|---|
| `CKGProtocol` | `new_ckg_protocol` | Collective public key generation |
| `RKGProtocol` | `new_rkg_protocol` | Collective relinearization key generation |
| `RTGProtocol` | `new_rot_kg_protocol` | Collective rotation key generation |
| `PCKSProtocol` | `new_pcks_protocol` | Public collective key-switching |
| `RefreshProtocol` | `new_refresh_protocol` | Collective bootstrapping |
| `E2SProtocol` | `new_e2s_protocol` | Encryption-to-Shares (HE → MPC) |
| `S2EProtocol` | `new_s2e_protocol` | Shares-to-Encryption (MPC → HE) |

Each protocol follows the same pattern:

1. Each party generates a **share** from its `sk_shard` + CRP
2. Shares are sent to the **hub** party
3. The hub **aggregates** shares to produce the result

The CRP (Common Reference Polynomial) is generated from a shared PRNG seed, so no party needs to send it.

## Next steps

- **[Lattiseq Overview](overview.md)** — Module structure and ring arithmetic.
- **[Core MHE Module](../deep-dive-shechi/core-mhe.md)** — How `MPCMHE` orchestrates these protocols.
- **[Schemes and Protocols](../crypto-corner/schemes-protocols.md)** — Mathematical foundations.
