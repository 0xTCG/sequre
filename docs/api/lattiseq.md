# Lattiseq (CKKS)

_Defined in `stdlib/sequre/lattiseq/`_

Lattiseq is a ground-up Codon reimplementation of the Go-based [Lattigo](https://github.com/tuneinsight/lattigo) library. It provides the CKKS approximate homomorphic encryption scheme used by Shechi.

## Module structure

| Module | Description |
|---|---|
| `ckks.codon` | CKKS scheme: `Parameters`, `Plaintext`, `Ciphertext`, `Evaluator`, `Encoder`, `Encryptor`, `Decryptor` |
| `exact_encoder.codon` | `ExactEncoder`: CKKS encoding whose error does not grow with the magnitude of the values |
| `rlwe.codon` | Ring-LWE primitives: `SecretKey`, `PublicKey`, `RelinearizationKey`, `RotationKeySet` |
| `ring.codon` | Polynomial ring arithmetic: `Ring`, `Poly`, NTT, modular reduction |
| `ringqp.codon` | QP-basis ring: `Ring`, `Poly`, `UniformSampler` |
| `params.codon` | Pre-defined CKKS parameter sets |
| `dckks.codon` | Distributed CKKS protocols: PCKS, CKG, RKG, Refresh, E2S, S2E |
| `drlwe.codon` | Distributed RLWE share types: `PCKSShare`, `CKGShare`, `RKGShare`, `RTGShare` |
| `utils.codon` | Modular arithmetic utilities, PRNG |
| `stats.codon` | Precision analysis for CKKS plaintexts |

## Pre-defined parameter sets

| Name | logN | Slots | Q moduli | P moduli | Security |
|---|---|---|---|---|---|
| `PN12QP109` | 12 | 2048 | 2 | 1 | 128-bit |
| `PN13QP218` | 13 | 4096 | 4 | 1 | 128-bit |
| `PN14QP438` | 14 | 8192 | 8 | 2 | 128-bit |
| `PN15QP880` | 15 | 16384 | 16 | 3 | 128-bit |
| `PN16QP1761` | 16 | 32768 | 34 | 4 | 128-bit |

Each set has a corresponding `CI` (continuous integration) variant with smaller parameters for faster testing.

## Key types

### `Parameters`

Holds the CKKS scheme configuration: ring degree, modulus chain, default scale, and slot count.

```python
from sequre.lattiseq.ckks import new_parameters_from_literal
from sequre.lattiseq.params import DEFAULT_PARAMS

params = new_parameters_from_literal(DEFAULT_PARAMS)
slots = params.slots()        # e.g. 4096
max_level = params.max_level() # e.g. 7
```

### `Plaintext` / `Ciphertext`

CKKS plaintext and ciphertext objects. A `Plaintext` holds encoded (but unencrypted) polynomial data. A `Ciphertext` holds encrypted polynomial data.

### `Evaluator`

Performs homomorphic operations on ciphertexts:

| Operation | Method |
|---|---|
| Add | `evaluator.add(ct1, ct2)` |
| Multiply | `evaluator.mul(ct1, ct2)` |
| Relinearize | `evaluator.relinearize(ct)` |
| Rescale | `evaluator.rescale(ct)` |
| Rotate | `evaluator.rotate(ct, k)` |

### `EncoderComplex128`

Encodes/decodes vectors of complex numbers into CKKS plaintexts. It computes in double precision, so its error grows with the magnitude of the values: about $|v| \cdot 2^{-52}$ per slot.

### `ExactEncoder`

_Defined in `exact_encoder.codon`._ Encodes and decodes fixed-point values with an error that does not depend on their magnitude: only CKKS's own rounding of the coefficients remains (at most $N / (2\Delta)$ per slot). Built for protocols that hide values under large random masks. See [CKKS Operations](../deep-dive-lattiseq/ckks-operations.md#exact-encoding).

| Method | Description |
|---|---|
| `new_exact_encoder(params)` | Creates the encoder (standard ring only) |
| `encode_fixed_point(real, imag, frac_bits, plaintext, log_slots)` | Encodes the slots $(\text{real} + i \cdot \text{imag}) / 2^{\text{frac\_bits}}$, given as `lattiseq_exact_int` lists, at the plaintext's level and scale |
| `decode_fixed_point(plaintext, log_slots, frac_bits)` | Returns the real and imaginary parts of the slots times $2^{\text{frac\_bits}}$, rounded |
| `encode_coeffs(real, imag, frac_bits, scale, log_slots)` / `decode_coeffs(coeffs, scale, log_slots, frac_bits)` | The same on integer coefficient vectors, without residues |
| `coeffs_to_poly(coeffs, log_slots, poly)` / `poly_to_coeffs(poly, log_slots)` | Exact conversion between integer coefficients and a polynomial in residue (RNS) form |
| `encode(values, plaintext, log_slots)` / `decode(plaintext, log_slots)` | Float interface, as `EncoderComplex128` |

Values that do not fit its integers (`LATTISEQ_EXACT_INT_SIZE` bits, minus about 25 bits of working room) raise `ValueError` instead of losing precision.

### `PkEncryptor` / `Decryptor`

Public-key encryption and secret-key decryption.

## Distributed protocols (dCKKS)

These protocols coordinate key generation and ciphertext operations across parties:

| Protocol | Purpose |
|---|---|
| `CKG` (Collective Key Gen) | Generate a shared public key from individual secret key shards |
| `RKG` (Relin Key Gen) | Generate shared relinearization keys |
| `RotKG` (Rotation Key Gen) | Generate shared rotation (Galois) keys |
| `PCKS` (Public Collective Key Switch) | Re-encrypt under a different collective key |
| `Refresh` | Refresh a ciphertext's noise budget via collective bootstrapping |
| `E2S` / `S2E` | Convert between encrypted and secret-shared representations |
