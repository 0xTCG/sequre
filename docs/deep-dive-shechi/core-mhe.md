# Core MHE Module

The `MPCMHE` class in `stdlib/sequre/mpc/mhe.codon` is the runtime engine for **Multiparty Homomorphic Encryption** in Shechi. It orchestrates collective key generation, encryption, decryption, bootstrapping, and protocol switching between HE and MPC worlds.

## Architecture

```
MPCMHE[TP]
├── CryptoParams          (per-party CKKS state)
│   ├── sk_shard          (secret-key share)
│   ├── pk, rlk, rotks    (collective public keys)
│   ├── encoder / encryptor / decryptor / evaluator
│   └── params            (CKKS Parameters)
├── crp_gen               (Common Reference Polynomial sampler)
├── refresh_protocol      (collective bootstrapping)
├── comms: MPCComms[TP]   (network layer)
├── stats: MPCStats       (operation counters)
└── randomness: MPCRandomness
```

## CryptoParams

`CryptoParams` bundles all CKKS scheme objects for one party:

| Field | Type | Description |
|---|---|---|
| `sk_shard` | `SecretKey` | This party's secret-key share (party 0 holds a zero polynomial) |
| `pk` | `PublicKey` | Collectively generated public key |
| `rlk` | `RelinearizationKey` | Collectively generated relinearization key |
| `rotks` | `RotationKeySet` | Collectively generated rotation keys |
| `params` | `Parameters` | CKKS parameter set (ring degree, moduli, scale) |
| `encoder` | `EncoderComplex128` | CKKS complex encoder |
| `encryptor` | `PkEncryptor` | Public-key encryptor |
| `decryptor` | `Decryptor` | Partial decryptor (using `sk_shard`) |
| `evaluator` | `Evaluator` | HE evaluator (add, mul, rotate, rescale) |

## Collective Key Generation

`collective_init(params, prec)` runs the distributed setup:

1. **Secret key**: Each CP generates an independent `sk_shard`. CP0 holds a zero polynomial.
2. **Public key**: `_collective_pub_key_gen` — each party produces a CKG share from its `sk_shard` + common reference polynomial (CRP). Shares are aggregated to form a single `pk`.
3. **Relinearization key**: `_collective_relin_key_gen` — similar round with RKG shares.
4. **Rotation keys**: `_collective_rot_key_gen` — one RTG round per rotation index. Results are cached in debug mode (`_internal_mhe_rtks_*`).
5. **Refresh protocol**: Initialized for later collective bootstrapping.

```python
def collective_init(self, params, prec):
    kgen = new_key_generator(params)
    sk_shard = kgen.gen_secret_key()  # CP0 gets zero poly

    pk  = self._collective_pub_key_gen(params, sk_shard, self.crp_gen)
    rlk = self._collective_relin_key_gen(params, sk_shard, self.crp_gen)
    rtks = self._collective_rot_key_gen(params, sk_shard, self.crp_gen, ...)

    self.crypto_params.initialize(sk_shard, pk, rlk, rtks, prec)
```

## Encryption & Decryption

### `enc_vector[T](values) -> list[T]`

Encodes and encrypts a flat list of values into a list of CKKS ciphertexts (or plaintexts). Values are chunked by `slots` (number of CKKS slots per ciphertext).

- `T = Ciphertext` → encrypt via public key
- `T = Plaintext` → encode only (no encryption)

### `decrypt(x, source_pid) -> list[Plaintext]`

Collective decryption. Behaviour depends on `source_pid`:

| `source_pid` | Meaning |
|---|---|
| `>= 0` | Decrypt ciphertexts held by that party, broadcast result |
| `-1` | Ciphertexts already shared between all parties |
| `-2` | Decrypt each party's ciphertexts one-by-one |

### `reveal[dtype](x, source_pid) -> list[dtype]`

Convenience: decrypt + decode in one call.

## HE Operations

All operations come in **in-place** (`i*`) and **copying** variants:

| Method | Operation |
|---|---|
| `iadd` / `add` | CT + CT element-wise add |
| `isub` / `sub` | CT − CT element-wise subtract |
| `imul` / `mul` | CT × CT/PT multiply (with automatic relinearization & refresh) |
| `iadd_const` / `add_const` | CT + scalar |
| `imul_const` / `mul_const` | CT × scalar |
| `irotate` / `rotate` | Galois rotation by `k` positions |
| `shift` | Slot-level shift with masking (handles cross-ciphertext boundaries) |
| `reduce_add` | Sum across slots within each ciphertext |
| `ineg` / `neg` | Negate |

### Rotation strategy

`irotate` uses direct rotation keys when `k <= LATTISEQ_ROTATION_KEYS_COUNT`. For larger `k`, it falls back to `irotate_butterfly` — a logarithmic decomposition using power-of-2 rotations:

```
k = 13 → binary 1101 → rotations by 1, 4, 8
```

## Refresh & Bootstrap

`refresh(x)` manages the CKKS noise budget:

1. **Rescale** — bring ciphertexts to default scale
2. **Bootstrap** — if any ciphertext's level is at or below `bootstrap_min_level`, run collective bootstrapping

The collective bootstrap uses the dCKKS `RefreshProtocol`:

- Each party generates a `RefreshShare` from its `sk_shard`
- Shares are aggregated at the hub party
- The hub produces a refreshed ciphertext at `max_level`

The `bootstrap_min_level` and `bootstrap_log_bound` parameters control the security/correctness trade-off.

## Protocol Switching: E2S and S2E

These methods convert between the HE and MPC representations:

### `additive_share_vector_to_ciphervector` (MPC → MHE)

1. Convert integer shares to fixed-point if needed
2. Mask with random values bounded by `max_mask_bits`
3. Reveal masked shares
4. Each party encrypts its portion
5. Aggregate encrypted vectors at hub

### `ciphervector_to_additive_share_vector` (MHE → MPC)

1. Broadcast ciphertext from source party
2. Refresh to ensure enough levels
3. Flatten levels across all ciphertexts
4. Each party computes `sk_shard × c1 + mask + smudging_noise`
5. Aggregate at hub → yields `c0 + sum(sk_i × c1) + sum(mask_i) + sum(e_i)`
6. Hub decodes and subtracts its mask to get its additive share
7. Other parties use their mask as their additive share

The smudging noise (`gaussian_sampler`) provides defense-in-depth against leakage from the partial decryption shares.

## Next steps

- **[Ciphertensor Internals](ciphertensor-internals.md)** — How `Ciphertensor` wraps these MHE ops.
- **[Lattiseq Overview](../deep-dive-lattiseq/overview.md)** — The CKKS engine underneath.
- **[MPCMHE API](../api/mpcenv.md)** — Full method reference.
