# MPC Instance

_Defined in `stdlib/sequre/mpc/env.codon`_

`MPCEnv[TP]` is the central runtime environment for all secure computation in Sequre. Every `@sequre`-annotated function receives it as its first argument (`mpc`).

## Type parameter

| Parameter | Description |
|---|---|
| `TP` | Unsigned integer type for MPC arithmetic (default: `mpc_uint` = `UInt[192]`) |

## Fields

| Field | Type | Description |
|---|---|---|
| `pid` | `int` | This party's ID (0 = trusted dealer, 1..N = compute parties) |
| `local` | `bool` | Whether running in local (forked) mode |
| `default_mpc_modulus` | `TP` | Default modulus for MPC arithmetic |
| `default_ciphertensor_encoding` | `str` | Default encoding for ciphertensors |
| `default_allow_mpc_switch` | `bool` | Whether automatic MPC ↔ MHE switching is enabled |
| `stats` | `MPCStats` | Operation counters and byte-transfer tracking |
| `randomness` | `MPCRandomness` | PRG streams and seed management |
| `comms` | `MPCComms[TP]` | Network I/O (send, receive, reveal, broadcast) |
| `arithmetic` | `MPCArithmetic[TP]` | Beaver-triple multiplication and partitioning |
| `polynomial` | `MPCPolynomial[TP]` | Lagrange interpolation, polynomial evaluation |
| `boolean` | `MPCBoolean[TP]` | Bit decomposition, boolean circuits |
| `fp` | `MPCFP[TP]` | Fixed-point truncation and normalization |
| `mhe` | `MPCMHE[TP]` | Multiparty homomorphic encryption (CKKS) |

## Methods

| Method | Description |
|---|---|
| `council(value)` | Collect boolean votes from all parties. Returns `list[bool]`. Used to check and communicate certain properties between parties (e.g., if a variable is CKKS encrypted at each parties). |
| `done()` | Synchronize parties and close all connections |

## Initialization

### Local mode (single machine)

```python
from sequre import local, Sharetensor as Stensor

@local
def my_computation(mpc, x: int):
    s = Stensor.enc(mpc, x)
    # ... secure operations ...
    print(s.reveal(mpc))

my_computation(42)  # mpc is injected automatically
```

The `@local` decorator forks `NUMBER_OF_PARTIES` processes (default 3) on the same machine, each with its own MPC instance.

### Online mode (distributed)

```python
from sequre import mpc, Sharetensor as Stensor

mpc = mpc()  # Reads party ID from sys.argv
s = Stensor.enc(mpc, 42)
print(s.reveal(mpc))
```

Run at each party:
```bash
SEQURE_CP_IPS=ip1,ip2,ip3 ./bin/sequre script.codon <pid>
```

## MHE setup

By default, `mpc()` and `@local` call `mpc.mhe.default_setup()` which:

1. Generates per-party secret key shards
2. Runs collective public key generation (CKG)
3. Runs collective relinearization key generation (RKG)
4. Runs collective rotation key generation

Skip this with `--skip-mhe-setup` if only MPC is needed.

---

## Sub-modules

### `mpc.arithmetic` — `MPCArithmetic[TP]`

_Defined in `stdlib/sequre/mpc/arithmetic.codon`_

Beaver-triple based secure arithmetic over secret-shared values.

| Method | Signature | Description |
|---|---|---|
| `add_public` | `(x, a, modulus)` | Add a public value `a` to secret-shared `x` (only at party 1) |
| `multiply` | `(a, b, modulus)` | Beaver-triple secure element-wise multiplication |
| `multiply_matmul` | `(a, b, modulus)` | Beaver-triple secure matrix multiplication |
| `inner_prod` | `(a, modulus)` | Secure inner product of secret-shared vectors |
| `beaver_inner_prod` | `(ar, am, modulus)` | Inner product on pre-partitioned Beaver shares |
| `beaver_inner_prod_pair` | `(ar, am, br, bm, modulus)` | Pairwise inner product on pre-partitioned Beaver shares |
| `multiply_bulk` | `(a, b, modulus)` | Batch element-wise multiplication over a list of matrices |
| `multiply_mat_bulk` | `(a, b, modulus)` | Batch matrix multiplication over a list of matrices |
| `ring_to_field` | `(target)` | Convert a secret-shared value from ring modulus to field modulus |
| `field_to_ring` | `(target)` | Convert a secret-shared value from field modulus to ring modulus |
| `validate_partitions` | `(x, x_r, r, modulus, message)` | Debug utility: assert that Beaver partitions are correct |
| `reset_stats` | `()` | Reset arithmetic operation counters |
| `print_stats` | `(file_stream=None)` | Print arithmetic stats (partitions, reconstructions) |

---

### `mpc.boolean` — `MPCBoolean[TP]`

_Defined in `stdlib/sequre/mpc/boolean.codon`_

Boolean circuits and comparison protocols over secret-shared values. Implements algorithms from _"Unconditionally Secure Constant-Rounds MPC for Equality, Comparison, Bits and Exponentiation"_ by Damgård et al.

| Method | Signature | Description |
|---|---|---|
| `bit_decomposition` | `(a, bitlen, small_modulus, modulus)` | Decompose secret-shared values into secret-shared bits |
| `demultiplexer` | `(bits_matrix, modulus)` | One-hot decode from a matrix of secret-shared bits |
| `carries` | `(a_bits, b_bits, public, modulus)` | Compute carry bits when adding two secret-shared bit representations |
| `bit_add` | `(a_bits, b_bits, public, modulus)` | Bitwise addition of secret-shared bit decompositions |
| `fan_in_or` | `(a, modulus)` | Compute OR reduction over the columns of a shared bit matrix |
| `prefix_or` | `(a, modulus)` | Compute prefix-OR over the columns of a shared bit matrix |
| `fan_in_and` | `(a, modulus)` | Compute AND reduction (via De Morgan + `fan_in_or`) |
| `prefix_and` | `(a, modulus)` | Compute prefix-AND (via De Morgan + `prefix_or`) |
| `flip_bit` | `(a, modulus)` | Negate shared bits: $1 - a$ |
| `is_positive` | `(a, modulus)` | Test whether a secret-shared value is positive. Returns shared bit |
| `less_than_bits_public` | `(a, b_pub, modulus)` | Bitwise comparison $a < b$ where `b` is public |
| `less_than_bits` | `(a, b, modulus)` | Bitwise comparison $a < b$ where both are secret-shared |
| `less_than_public` | `(a, bpub, modulus)` | Comparison $a < b$ where `b` is public. Works on scalars, 1D, and 2D inputs |
| `less_than` | `(a, b, modulus)` | Comparison $a < b$ on secret-shared values. Works on scalars, 1D, and 2D inputs |
| `not_less_than` | `(a, b, modulus)` | Comparison $a \geq b$ on secret-shared values |
| `not_less_than_public` | `(a, bpub, modulus)` | Comparison $a \geq b$ where `b` is public |
| `greater_than_public` | `(a, bpub, modulus)` | Comparison $a > b$ where `b` is public |
| `greater_than` | `(a, b, modulus)` | Comparison $a > b$ on secret-shared values |
| `not_greater_than` | `(a, b, modulus)` | Comparison $a \leq b$ on secret-shared values |
| `not_greater_than_public` | `(a, bpub, modulus)` | Comparison $a \leq b$ where `b` is public |

---

### `mpc.fp` — `MPCFP[TP]`

_Defined in `stdlib/sequre/mpc/fp.codon`_

Fixed-point arithmetic protocols for secret-shared values.

| Method | Signature | Description |
|---|---|---|
| `trunc` | `(a, modulus, k=MPC_NBIT_K+MPC_NBIT_F, m=MPC_NBIT_F)` | Fixed-point truncation. Divides by $2^m$ under the secret-sharing. Supports both power-of-two and prime moduli |
| `reset_stats` | `()` | Reset truncation counters |
| `print_stats` | `(file_stream=None)` | Print fixed-point stats (truncation count) |

---

### `mpc.polynomial` — `MPCPolynomial[TP]`

_Defined in `stdlib/sequre/mpc/polynomial.codon`_

Polynomial operations over secret-shared values, including Lagrange interpolation and power computation.

| Method | Signature | Description |
|---|---|---|
| `lagrange_interp` | `(x: list[int], y, modulus)` | Lagrange interpolation on given `(x, y)` points |
| `lagrange_interp_simple` | `(y, modulus)` | Lagrange interpolation with implicit $x = [1, 2, \ldots, n]$ |
| `table_lookup` | `(x, table_id, modulus)` | Evaluate a pre-cached polynomial lookup table on secret-shared `x` |
| `powers` | `(x, power, modulus)` | Compute all powers $[x^0, x^1, \ldots, x^{\text{power}}]$ of a secret-shared value using Beaver triples and Pascal's triangle |
| `powers_cached` | `(x_r, r, power, modulus)` | Same as `powers`, but with pre-computed Beaver partition `(x_r, r)` |
| `evaluate_poly` | `(x, coeff, modulus)` | Evaluate polynomials defined by `coeff` matrix at secret-shared points `x` |
| `get_pascal_matrix` | `(power)` | Get (cached) Pascal triangle matrix up to given power |
| `calculate_pascal_matrix` | `(pow)` | Compute Pascal triangle matrix up to given power |

---

### `mpc.comms` — `MPCComms[TP]`

_Defined in `stdlib/sequre/mpc/comms.codon`_

Network communication layer for MPC. Handles point-to-point messaging, collective operations, and secret sharing over TLS-encrypted channels.

#### Fields

| Field | Type | Description |
|---|---|---|
| `pid` | `int` | This party's ID |
| `hub_pid` | `int` | Hub party for aggregation (default: 1) |
| `number_of_parties` | `int` | Total number of parties (including trusted dealer) |
| `local` | `bool` | Whether running in local (AF_UNIX socket) mode |
| `detach_dealer` | `bool` | If `True`, dealer messages are buffered instead of sent immediately |

#### Point-to-point communication

| Method | Signature | Description |
|---|---|---|
| `send` | `(data, to_pid)` | Send (serialize) data to a specific party |
| `receive[T]` | `(from_pid)` | Receive and deserialize data of type `T` from a party |
| `send_as_jar` | `(data, to_pid)` | Serialize and send data to a specific party with size prefix |
| `receive_as_jar[T]` | `(from_pid)` | Receive size-prefixed data and deserialize as type `T` |
| `send_jar_size` | `(data, to_pid)` | Send only the serialized size of data |
| `receive_jar_size` | `(from_pid)` | Receive only a serialized size (returns `int`) |

#### Revealing / reconstruction

| Method | Signature | Description |
|---|---|---|
| `reveal` | `(value, modulus)` | Reconstruct a secret-shared value (all compute parties learn the result) |
| `reveal_at` | `(value, target_pid, modulus)` | Reconstruct a secret-shared value only at a specific party |
| `reveal_no_mod` | `(value)` | Reconstruct without modular reduction |
| `reveal_to_all` | `(value, modulus)` | Reconstruct and additionally share result with the trusted dealer |

#### Secret sharing

| Method | Signature | Description |
|---|---|---|
| `share_from` | `(data, source_pid, modulus)` | Generate additive shares of `data` from a specific party |
| `share_from_trusted_dealer` | `(data, modulus)` | Generate additive shares of `data` from the trusted dealer (CP0) |

#### Collective operations

| Method | Signature | Description |
|---|---|---|
| `collect[T]` | `(value, include_trusted_dealer=False, exclude_parties=Set[int]())` | All-to-all gather: each party sends its value and receives values from all others |
| `collect_at[T]` | `(value, target_pid)` | Gather values at a single target party |
| `broadcast_from[T]` | `(value, source_pid)` | One-to-all broadcast from a source party |
| `send_to_all_from` | `(value, source_pid, include_trusted_dealer=False)` | Send a value from one party to all others |
| `is_broadcast` | `(value)` | Broadcast a boolean consensus: collect, AND-reduce, and broadcast result |
| `sync_parties` | `(lite=True, include_trusted_dealer=True)` | Barrier synchronization across all parties |

#### Dealer buffer management

| Method | Signature | Description |
|---|---|---|
| `flush` | `(to_pid, jar, pickle_size)` | Send or buffer a serialized message |
| `flush_dealer_buffers` | `()` | Flush all buffered dealer messages at once (parallel) |

#### Utilities

| Method | Signature | Description |
|---|---|---|
| `print_fp` | `(value, modulus, debug=False, message="")` | Reveal and print a fixed-point value as a float (debug helper) |
| `sequential` | `(func, skip_dealer, *args, **kwargs)` | Execute `func` sequentially at each party (ordered by PID) |
| `clean_up` | `()` | Close all sockets |
| `reset_stats` | `()` | Reset communication counters |
| `print_stats` | `(file_stream=None)` | Print communication stats (bytes sent, rounds, etc.) |

---

### `mpc.mhe` — `MPCMHE[TP]`

_Defined in `stdlib/sequre/mpc/mhe.codon`_

Multiparty homomorphic encryption based on the CKKS scheme. Provides collective key generation, ciphertext operations, E2S/S2E protocol switching, and collective bootstrapping.

#### Fields

| Field | Type | Description |
|---|---|---|
| `crp_gen` | `UniformSampler` | Common reference polynomial generator |
| `crypto_params` | `CryptoParams` | Aggregated CKKS scheme state (keys, encoder, encryptor, decryptor, evaluator) |
| `refresh_protocol` | `RefreshProtocol` | Collective bootstrapping protocol |
| `bootstrap_min_level` | `int` | Minimum ciphertext level required for safe bootstrapping |
| `bootstrap_log_bound` | `int` | Log2 of the smudging noise bound for bootstrapping |
| `bootstrap_safe` | `bool` | Whether current parameters are safe for collective bootstrapping |

#### Setup & key generation

| Method | Signature | Description |
|---|---|---|
| `default_setup` | `()` | Initialize CKKS with default parameters and generate all collective keys |
| `collective_init` | `(params, prec)` | Full collective initialization with custom CKKS parameters and precision |

#### Encoding & encryption

| Method | Signature | Description |
|---|---|---|
| `enc_vector[T]` | `(values)` | Encode and encrypt a list of values into a list of `Ciphertext` or `Plaintext` (type selected by `T`) |
| `decrypt` | `(x, source_pid=-2)` | Collectively decrypt ciphertexts. `source_pid`: specific party, `-2` = round-robin, `-1` = already broadcast |
| `decode[dtype]` | `(enc)` | Decode a list of `Plaintext` into `list[dtype]` (`int`, `float`, or `complex`) |
| `reveal[dtype]` | `(x, source_pid=-2)` | Decrypt then decode (convenience wrapper) |

#### E2S / S2E protocol switching

| Method | Signature | Description |
|---|---|---|
| `cipher_to_additive_plaintext` | `(ct, hub_pid)` | E2S: convert a `Ciphertext` to an `AdditiveShareBigint` |
| `additive_plaintext_to_cipher` | `(secret_share, hub_pid)` | S2E: convert an `AdditiveShareBigint` back to a `Ciphertext` |
| `additive_share_vector_to_ciphervector` | `(shared_tensor, modulus, is_fp, target_pid=-1)` | MPC→MHE: convert MPC secret shares to a ciphervector |
| `ciphervector_to_additive_share_vector` | `(ciphervector, number_of_elements, modulus, source_pid)` | MHE→MPC: convert a ciphervector to MPC secret shares |

#### Ciphertext arithmetic (in-place)

| Method | Signature | Description |
|---|---|---|
| `ineg` | `(x)` | Negate ciphertexts in-place |
| `iadd` | `(x, y)` | Add ciphertexts in-place: $x \mathrel{+}= y$ |
| `isub` | `(x, y)` | Subtract ciphertexts in-place: $x \mathrel{-}= y$ |
| `imul[T]` | `(x, y, x_is_broadcast, y_is_broadcast, no_refresh)` | Multiply ciphertexts in-place (auto-relinearizes for `Ciphertext × Ciphertext`) |
| `imul_noboot[T]` | `(x, y)` | Multiply ciphertexts in-place without bootstrapping |
| `iadd_const` | `(x, constant)` | Add a scalar constant in-place |
| `isub_const` | `(x, constant)` | Subtract a scalar constant in-place |
| `imul_const` | `(x, constant)` | Multiply by a scalar constant in-place |
| `irotate` | `(x, k)` | Rotate ciphertext slots by `k` positions in-place |
| `irotate_butterfly` | `(x, k)` | Butterfly-decomposed rotation for large `k` |

#### Ciphertext arithmetic (out-of-place)

| Method | Signature | Description |
|---|---|---|
| `neg` | `(x)` | Return negated copy |
| `add` | `(x, y)` | Return sum of ciphervectors |
| `sub` | `(x, y)` | Return difference of ciphervectors |
| `mul` | `(x, y)` | Return product of ciphervectors |
| `mul_noboot` | `(x, y)` | Return product without bootstrapping |
| `add_const` | `(x, constant)` | Return ciphervector + constant |
| `sub_const` | `(x, constant)` | Return ciphervector − constant |
| `mul_const` | `(x, constant)` | Return ciphervector × constant |
| `rotate` | `(x, k)` | Return rotated copy |

#### Level & scale management

| Method | Signature | Description |
|---|---|---|
| `rescale` | `(x, target_scale)` | Rescale ciphertexts to `target_scale` |
| `requires_bootstrap` | `(x, min_level_distance=0)` | Check which ciphertexts need bootstrapping (returns `list[bool]`) |
| `bootstrap` | `(x, is_broadcast=False, min_level_distance=0)` | Collectively bootstrap ciphertexts that need it |
| `refresh` | `(x, is_broadcast=False, min_level_distance=0)` | Rescale then bootstrap as needed (recommended before heavy computation) |
| `drop_level` | `(a, out_level)` | Drop ciphertext level (works on `list[Ciphertext]` and `list[list[Ciphertext]]`) |
| `flatten_levels` | `(x)` | Drop all ciphertexts in a matrix to the minimum level. Returns `(matrix, min_level)` |

#### Shifting & reduction

| Method | Signature | Description |
|---|---|---|
| `shift` | `(x, step)` | Shift ciphervector by `step` slots (inter-ciphertext, with masking) |
| `shift_noboot` | `(x, step)` | Shift without bootstrapping |
| `reduce_add` | `(x, size, keep_dims=True)` | Sum-reduce a ciphervector along the slot axis |

#### Masking

| Method | Signature | Description |
|---|---|---|
| `mask_one[T]` | `(x, idx, complement=False)` | Mask a single slot (or its complement) in a ciphertext or ciphervector |
| `mask_range[T]` | `(x, start, stop, complement=False)` | Mask a range of slots in a ciphertext |

#### Utilities

| Method | Signature | Description |
|---|---|---|
| `zero_cipher` | `()` | Create an encryption of zero |

---

### `mpc.randomness` — `MPCRandomness`

_Defined in `stdlib/sequre/mpc/randomness.codon`_

PRG (pseudorandom generator) stream management for MPC. Each pair of parties shares a PRG stream seeded via a secure key exchange during initialization.

#### Fields

| Field | Type | Description |
|---|---|---|
| `pid` | `int` | This party's ID |
| `prg_states` | `dict[int, tuple[array[u32], int]]` | Saved PRG states keyed by party ID |
| `other_seeds` | `dict[int, seed_t]` | Seeds shared with each other party |
| `global_seed` | `seed_t` | Global seed shared via trusted dealer broadcast |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `reset_streams` | `()` | Reset all PRG streams (private, shared, and global) using current seeds |
| `reset_global_seed` | `()` | Reset the global seed stream (pid = −1) |
| `reset_seed` | `(pid, seed)` | Re-seed and save the PRG state for a specific stream |
| `get_state` | `()` | Get the current PRG generator state |
| `set_state` | `(state)` | Set the PRG generator state |
| `switch_seed` | `(pid)` | Save current private PRG state and load the shared stream for `pid` |
| `restore_seed` | `(pid)` | Save shared stream state for `pid` and restore private PRG state |
| `freeze_seed` | `(pid)` | Like `switch_seed`, but copies the state so the shared stream is not advanced |
| `unfreeze_seed` | `(pid)` | Restore from the frozen shared stream state |
| `seed_switch` | `(pid)` | Context manager: `with randomness.seed_switch(pid): ...` — switches to the shared PRG for `pid` and restores on exit |

---

### `mpc.stats` — `MPCStats`

_Defined in `stdlib/sequre/mpc/stats.codon`_

Tracks all MPC and MHE operation counters and communication metrics. Automatically updated by all sub-modules.

#### Security counters

| Field | Type | Description |
|---|---|---|
| `secure_add_count` | `int` | Secure additions |
| `secure_sub_count` | `int` | Secure subtractions |
| `secure_mul_count` | `int` | Secure multiplications |
| `secure_matmul_count` | `int` | Secure matrix multiplications |
| `secure_matmul_complexity` | `int` | Cumulative matrix multiplication complexity |
| `secure_pow_count` | `int` | Secure exponentiation operations |
| `secure_bootstrap_count` | `int` | HE collective bootstrapping operations |
| `secure_rescale_count` | `int` | HE rescale operations |
| `secure_mhe_mpc_switch_count` | `int` | MHE → MPC switches (E2S, ciphertext count) |
| `secure_mpc_mhe_switch_count` | `int` | MPC → MHE switches (S2E, ciphertext count) |

#### Communication counters

| Field | Type | Description |
|---|---|---|
| `bytes_sent` | `int` | Total bytes transmitted |
| `send_requests` | `int` | Total send operations |
| `receive_requests` | `int` | Total receive operations |
| `rounds` | `int` | Communication rounds |

#### Arithmetic counters

| Field | Type | Description |
|---|---|---|
| `partitions_count` | `int` | Beaver partitions performed |
| `reconstructs_count` | `int` | Beaver reconstructions performed |

#### Fixed-point counters

| Field | Type | Description |
|---|---|---|
| `truncations_count` | `int` | Fixed-point truncations |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `reset_stats` | `()` | Reset all counters |
| `print_stats` | `(msg="", file_stream=None, file_only=False)` | Print all stats with an optional label |
| `reset_security_stats` | `()` | Reset security counters only |
| `print_security_stats` | `(file_stream=None, file_only=False)` | Print security stats only |
| `reset_comms_stats` | `()` | Reset communication counters only |
| `print_comms_stats` | `(file_stream=None, file_only=False)` | Print communication stats only |
| `reset_arithmetic_stats` | `()` | Reset arithmetic counters only |
| `print_arithmetic_stats` | `(file_stream=None, file_only=False)` | Print arithmetic stats only |
| `reset_fp_stats` | `()` | Reset fixed-point counters only |
| `print_fp_stats` | `(file_stream=None, file_only=False)` | Print fixed-point stats only |
| `copy` | `()` | Return a copy of the current stats |
| `__iadd__` | `(other)` | Accumulate another `MPCStats` into this one |

---

### `mpc.collections`

_Defined in `stdlib/sequre/mpc/collections.codon`_

Higher-level secure collection primitives.

| Function | Signature | Description |
|---|---|---|
| `oblivious_get` | `(mpc, secret_keys, key_bit_len, public_array, delimiter_prime, modulus)` | Oblivious array access: retrieve elements from `public_array` at secret indices `secret_keys` using bit-decomposition and demultiplexing |

---

## Context managers

Convenience context managers accessible via methods on `MPCEnv`:

### `mpc.allow_mpc_switch()`

Enables automatic [MPC ↔ MHE switching](../user-guide/switching.md) for the duration of the block. Inside this context, operations like `Ciphertensor.matmul` may automatically convert to Sharetensor (via the E2S protocol), run the operation using Beaver-triple MPC, and convert back (via S2E) when a cost estimator determines this is cheaper than a pure-HE path.

```python
with mpc.allow_mpc_switch():
    result = ct_a @ ct_b  # matmul may use MPC path if cheaper
```

See the dedicated [MPC ↔ MHE Switching](../user-guide/switching.md) page for details.

### `mpc.default_modulus(modulus)`

Temporarily switch the working modulus (e.g., from field to ring). Restores the original modulus on exit.

```python
with mpc.default_modulus(RING_MODULUS):
    # all operations use RING_MODULUS
    ...
```

### `mpc.default_encoding(encoding)`

Temporarily switch the default ciphertensor encoding string. Restores the original encoding on exit.

```python
with mpc.default_encoding("diagonal"):
    ...
```

### `mpc.stats_log(msg, file_path, mode, file_only)`

Log operation statistics (multiplications, communication rounds, bytes sent, etc.) for a code block. Resets counters on entry and prints a summary on exit.

```python
with mpc.stats_log("Training phase"):
    model.fit(mpc, X=X, y=y, step=0.1, epochs=10, momentum=0.9)
```

Write stats to a file instead of (or in addition to) stdout:

```python
with mpc.stats_log("Inference", file_path="results/inference_stats.txt"):
    prediction = model.predict(mpc, X)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `msg` | `str` | `""` | Label printed alongside the stats summary |
| `file_path` | `str` | `""` | If non-empty, write stats to this file |
| `mode` | `str` | `"a+"` | File open mode |
| `file_only` | `bool` | `False` | If `True`, only write to file (suppress stdout) |
