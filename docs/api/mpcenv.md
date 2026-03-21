# MPCEnv

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
| `stats` | `MPCStats` | Operation counters and byte-transfer tracking |
| `randomness` | `MPCRandomness` | PRG streams and seed management |
| `comms` | `MPCComms[TP]` | Network I/O (send, receive, reveal, broadcast) |
| `arithmetic` | `MPCArithmetic[TP]` | Beaver-triple multiplication and partitioning |
| `polynomial` | `MPCPolynomial[TP]` | Lagrange interpolation, polynomial evaluation |
| `boolean` | `MPCBoolean[TP]` | Bit decomposition, boolean circuits |
| `fp` | `MPCFP[TP]` | Fixed-point truncation and normalization |
| `mhe` | `MPCMHE[TP]` | Multiparty homomorphic encryption (CKKS) |

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

The `@local` decorator forks `NUMBER_OF_PARTIES` processes (default 3) on the same machine, each with its own `MPCEnv`.

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

Skip this with `--skip-mhe-setup` if you only need MPC.

## Sub-modules

### `mpc.arithmetic`

| Method | Description |
|---|---|
| `multiply(a, b, modulus)` | Beaver-triple secure multiplication |
| `multiply_matmul(a, b, modulus)` | Secure matrix multiplication |
| `inner_prod(a, modulus)` | Secure inner product |

### `mpc.boolean`

| Method | Description |
|---|---|
| `bit_decomposition(a, bitlen, ...)` | Decompose a shared value into shared bits |
| `demultiplexer(bits, modulus)` | One-hot decode from shared bits |

### `mpc.fp`

| Method | Description |
|---|---|
| `trunc(a, modulus, k, m)` | Fixed-point truncation |

### `mpc.polynomial`

| Method | Description |
|---|---|
| `lagrange_interp(x, y, modulus)` | Lagrange interpolation on shared points |
| `powers(x, power, modulus)` | Compute powers of a shared value |

### `mpc.mhe`

| Method | Description |
|---|---|
| `default_setup()` | Initialize CKKS params and generate collective keys |
| `collective_init(params, prec)` | Full collective initialization with custom params |
| `enc_vector[T](values)` | Encrypt a list into a list of `Ciphertext` or `Plaintext` |

### `mpc.comms`

| Method | Description |
|---|---|
| `send_as_jar(data, to_pid)` | Serialize and send to a specific party |
| `receive_as_jar(from_pid, T=type)` | Receive and deserialize from a party |
| `reveal(share, modulus)` | Reconstruct a shared value |
| `sync_parties()` | Barrier synchronization |
| `collect(value)` | Gather a value from all parties |

### `mpc.stats`

Tracks all operations automatically:

| Counter | Description |
|---|---|
| `secure_mul_count` | Number of secure multiplications |
| `secure_matmul_count` | Number of secure matrix multiplications |
| `bytes_sent` | Total bytes transmitted |
| `rounds` | Communication rounds |
| `partitions_count` | Beaver partitions performed |
| `truncations_count` | Fixed-point truncations |
| `secure_bootstrap_count` | HE bootstrapping operations |

## Context managers

### `AllowMPCSwitch`

Enables automatic switching between MPC and MHE backends mid-computation.

### `ModulusSwitch`

Temporarily switch the working modulus (e.g., from field to ring).

### `StatsLog`

Snapshot and report operation statistics for a code block.
