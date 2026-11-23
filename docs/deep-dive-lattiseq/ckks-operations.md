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

The encoding computes in double precision (`complex`, two 64-bit floats), so its error grows with the magnitude of the values: a slot of magnitude $|v|$ comes back off by about $|v| \cdot 2^{-52}$.

### Exact encoding

`ExactEncoder` (`exact_encoder.codon`) removes that growth. Protocols that hide a value under a large random mask, encrypt the masked value and remove the mask under encryption need it: with masks of $2^{150}$, the float encoder's error is about $2^{98}$.

Its values are fixed-point integers: a slot holds $z = Z / 2^{f}$, with the real and imaginary parts of $Z$ given as `lattiseq_exact_int` (`Int[LATTISEQ_EXACT_INT_SIZE]`).

- **Encoding** computes the coefficients $\text{round}(\Delta \cdot \text{IFFT}(z))$ with an FFT on fixed-point integers (products in twice the width, roots of unity computed once in multi-precision floats). The FFT keeps enough fractional bits that the value it rounds is within $2^{-16}$ of the true one, whatever the magnitude.
- **Decoding** reconstructs the centered coefficients exactly from their residues (CRT) and returns $\text{round}(2^{f} \cdot \text{FFT}(c) / \Delta)$ the same way.

What remains is the error of CKKS itself: each coefficient is rounded to an integer, which costs at most $N / (2\Delta)$ per slot (about $6 \cdot 10^{-9}$ for $N = 2^{14}$ and $\Delta = 2^{34}$, for any magnitude).

```python
from sequre.lattiseq.exact_encoder import new_exact_encoder

exact = new_exact_encoder(params)
exact.encode_fixed_point(real, imag, frac_bits, plaintext, log_slots)  # lists of lattiseq_exact_int
real, imag = exact.decode_fixed_point(plaintext, log_slots, frac_bits)
```

`encode_coeffs` / `decode_coeffs` do the same on integer coefficient vectors, and `coeffs_to_poly` / `poly_to_coeffs` convert exactly between those and residue form, for protocols that hold shares of a plaintext's coefficients.

Limits:

- **Integer width.** `LATTISEQ_EXACT_INT_SIZE` is 256 bits (512 with 256-bit MPC integers), and about 25 bits of it are working room. Encoding takes $\log_2(\max|z| \cdot \Delta)$ plus those bits. Decoding takes $\log_2 \max|c| + \log_2(\text{slots}) + f - \log_2 \Delta$ plus those bits. Larger values raise `ValueError` instead of losing precision.
- **Wrapping modulo Q.** Encoded coefficients are reduced modulo the plaintext's moduli without a range check. Reduction is a ring homomorphism, so a sum of encodings of large masked values still decodes to the sum of the values, as long as the sum is itself small.
- **Ring.** Only the standard ring is supported.

Measured with 4 threads ($2^{13}$ slots, top level of `PN14QP438`): encoding takes 5.5 ms and decoding 4.8 ms, against 1.3 ms and 10 ms for the float encoder.

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

Coefficient-wise addition of the two ciphertexts. **Noise grows additively.** This is essentially free when both are on
one scale (within $10^{-9}$ relative, the last bits of scales built from the same moduli).

A rescale divides by a prime close to, not equal to, the default scale, so a product's scale and a fresh ciphertext's
differ by about $10^{-5}$. Adding them as if their scales were equal would misread one operand by that much, on every
slot with the same sign. The evaluator brings the operands onto one scale first:

1. At level 0, a product whose rescale is still pending (a scale ratio of $2^{20}$ or more) is rescaled first, with its
   own levels (and moved onto the other operand's scale when it keeps a level to spare): at level 0 the product's scale
   would not fit the modulus. Above level 0 the integer of the next step matches it exactly, and the sum keeps the
   product's scale for its own rescale, as in Lattigo. (Rescaling the product inside the addition would keep a level of
   the sum, but in a recurrence such as Clenshaw's a term kept one level low drags the next sums down, and the
   multiparty layer then refreshes more often. That layer rescales a product added to a ciphertext at its bootstrap
   level itself, so that the sum can be refreshed.)
2. An integer ratio multiplies the smaller-scale operand by that integer: exact, and no level (above $2^{62}$, two
   integer factors).
3. Any other ratio moves an operand that has a level to spare onto the other's scale: a product by the integer
   $\mathrm{round}(q \cdot s_{target} / s)$ and one rescale, exact to about $2^{-30}$. Only that operand's level is spent,
   as the sum is at the lower level anyway. A move costs that rescale's time, so `Ciphertensor` encodes a public operand
   of an addition (a number, a list or an array) at each receiving ciphertext's level and scale, where it needs none.
4. Otherwise (two operands at the sum's level, or a ratio no move reaches exactly) the addition proceeds as in Lattigo,
   without a level: a ratio of 2 or more multiplies the smaller-scale operand by the nearest integer, and a smaller
   ratio counts as equal, a residual of the ratio's distance to 1 (a rescale's $10^{-6}$ to $10^{-4}$). Exactness there
   would spend a level of the sum at the same scale, which near the bootstrap level leaves a sum that can no longer be
   refreshed, and near level 0 one whose values no longer fit the modulus.

Lattigo takes ratios below 2 as equal and multiplies by $\lfloor \text{ratio} \rfloor$ above, whatever the levels.

### Scalar operations

```python
evaluator.add_const(ct, constant, ct_out)   # ct + Δ·constant
evaluator.mul_const(ct, constant, ct_out)   # ct × constant
```

`add_const` encodes the constant at the ciphertext's scale and adds. With `ct_out` another ciphertext, `ct_out` first
receives every component of `ct` and its scale, so `ct` is left as it was; `add_const_new` and `sub_const_new` return a new
ciphertext. `mul_const` multiplies coefficients directly — the scale doubles.

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
