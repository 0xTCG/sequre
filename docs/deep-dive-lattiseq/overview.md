# Lattiseq Reimplementation

Lattiseq is Sequre's from-scratch reimplementation of the [Lattigo](https://github.com/tuneinsight/lattigo) homomorphic encryption library — ported from Go into Codon. It provides the CKKS scheme, RLWE primitives, polynomial ring arithmetic, and distributed (multiparty) protocols that underpin every HE operation in Shechi.

## Why reimplement?

Lattigo is written in Go. Sequre's compiler and runtime are built on Codon (compiled Python). Calling Go from Codon via FFI would introduce:

- **Serialization overhead** — every ciphertext crossing the boundary must be marshalled
- **Loss of compiler visibility** — the Sequre IR passes cannot inspect or optimize Go code
- **Deployment friction** — users would need both Codon and Go toolchains

By reimplementing in Codon, Lattiseq gives the compiler full visibility into HE operations, enabling optimizations like `@mhe_cipher_opt` (operation reordering) and `@mhe_enc_opt` (encoding selection).

## Module structure

All files live under `stdlib/sequre/lattiseq/`:

| Module | Lattigo equivalent | Purpose |
|---|---|---|
| `ring.codon` | `ring/` | Polynomial ring $\mathbb{Z}_q[X]/(X^N+1)$ arithmetic, NTT, Montgomery reduction |
| `ringqp.codon` | `ringqp/` | Double-CRT ring $R_Q \times R_P$, uniform sampling |
| `rlwe.codon` | `rlwe/` | RLWE key types (`SecretKey`, `PublicKey`, `RelinearizationKey`, `RotationKeySet`), RLWE parameters |
| `ckks.codon` | `ckks/` | CKKS scheme: `Parameters`, `Ciphertext`, `Plaintext`, `Encoder`, `Encryptor`, `Decryptor`, `Evaluator` |
| `drlwe.codon` | `drlwe/` | Distributed RLWE protocols: share types (`CKGShare`, `RKGShare`, `PCKSShare`, `RTGShare`) |
| `dckks.codon` | `dckks/` | Distributed CKKS: `RefreshProtocol`, E2S/S2E protocols, collective key-gen wrappers |
| `params.codon` | `ckks/params` | Pre-defined parameter sets (PN12 through PN16) |
| `utils.codon` | `utils/` | PRNG, sampling utilities |
| `stats.codon` | — | Precision measurement for CKKS outputs |

## Polynomial ring (`ring.codon`)

The core data structure is `Poly` — a polynomial in $\mathbb{Z}_q[X]/(X^N+1)$ stored in NTT (Number Theoretic Transform) form for $O(N \log N)$ multiplication.

Key operations:

- **NTT / InvNTT** — Forward and inverse number-theoretic transforms
- **Montgomery mul** — Coefficient-wise multiplication in Montgomery form
- **Add / Sub / Neg** — Modular coefficient arithmetic
- **Level-aware variants** — `_mm_*_lvl` methods operate only on the first `level+1` moduli in the RNS chain

The ring supports **RNS (Residue Number System)** representation: each coefficient is stored modulo multiple co-prime moduli $q_0, q_1, \ldots, q_L$, enabling word-sized arithmetic on big integers.

## RingQP (`ringqp.codon`)

Combines two rings $R_Q$ and $R_P$ for key-switching operations. The auxiliary modulus $P$ is used during relinearization to control noise growth:

$$\text{KeySwitch}(ct) \approx \frac{1}{P} \cdot \text{Decompose}(ct) \cdot \text{EvalKey}$$

Also provides `UniformSampler` for generating common reference polynomials (CRPs) from a shared PRNG seed.

## RLWE layer (`rlwe.codon`)

Defines the fundamental key types:

```
SecretKey      — Poly in R_QP (the secret)
PublicKey      — pair of Poly (RLWE encryption of zero)
RelinearizationKey — gadget ciphertexts for degree reduction
RotationKeySet — Galois keys for slot rotations
EvaluationKey  — bundles rlk + rotation keys for the Evaluator
```

RLWE `Parameters` stores ring degree (`logn`), modulus chains (`qi`, `pi`), Gaussian width (`sigma`), and Hamming weight (`h`).

## Pre-defined parameter sets

From `params.codon`:

| Name | logN | logQP | Levels | Slots | Scale | Security |
|---|---|---|---|---|---|---|
| `PN12QP109` | 12 | 109 | 2 | 2048 | $2^{32}$ | 128-bit |
| `PN13QP218` | 13 | 218 | 6 | 4096 | $2^{30}$ | 128-bit |
| `PN14QP438` | 14 | 438 | 10 | 8192 | $2^{34}$ | 128-bit |
| `PN15QP880` | 15 | 880 | 18 | 16384 | $2^{40}$ | 128-bit |
| `PN16QP1761` | 16 | 1761 | 34 | 32768 | $2^{45}$ | 128-bit |

Conjugate-invariant (`*CI`) variants are also available, doubling slot capacity for real-only computations.

`DEFAULT_PARAMS` (used by `MPCMHE.default_setup`) is set in `params.codon` and typically points to `PN14QP438` or `PN15QP880` depending on the application.

## Next steps

- **[CKKS Operations](ckks-operations.md)** — Encode, encrypt, evaluate, rotate.
- **[Core MHE Module](../deep-dive-shechi/core-mhe.md)** — How distributed protocols use Lattiseq.
- **[Lattiseq API](../api/lattiseq.md)** — Complete reference.
