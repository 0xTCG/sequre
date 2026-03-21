# Running Distributed

This page covers how to run Sequre protocols across multiple machines, including command-line flags, network configuration, TLS setup, and the two execution modes (local and distributed).

---

## Command-line flags

Run `sequre --help` (or pass `--help` to your compiled binary) to see all available flags.

### Runtime flags

These flags can be passed to any Sequre program (both `@local` and `mpc()` programs):

| Flag | Description |
|---|---|
| `--use-ring` | Use a power-of-two ring modulus instead of a prime field modulus. **Warning:** currently unstable — division and square root may produce incorrect results ~50% of the time due to a modulus-switching bug. |
| `--skip-mhe-setup` | Skip the multiparty homomorphic encryption key-generation phase. Useful when your protocol only uses secret sharing (SMC) without any HE operations. |
| `-h`, `--help` | Print usage information and exit. |

The `--local` flag is **only** used by the built-in test runner (`scripts/invoke.codon`). For custom programs, use the `@local` decorator for local execution or the `mpc()` function for distributed execution — see [Execution modes](#execution-modes) below.

```bash
# @local program with all defaults
sequre run my_protocol.codon

# @local program, skip MHE setup (SMC-only protocol)
sequre run my_protocol.codon --skip-mhe-setup

# Distributed run as party 1
sequre run my_protocol.codon 1

# Distributed run as party 2, ring arithmetic
sequre run my_protocol.codon 2 --use-ring
```

### Test/benchmark flags

When using the built-in test runner (`scripts/invoke.codon`), additional flags select which tests or benchmarks to run:

| Flag | Selects |
|---|---|
| `--all` | All tests/benchmarks |
| `--unit` | All unit tests |
| `--e2e` | All end-to-end tests |
| `--helpers` | Helper utility tests |
| `--primitives` | MPC primitive tests |
| `--he` | HE tests |
| `--mpc` | MPC protocol tests |
| `--sharetensor` | Sharetensor tests |
| `--ciphertensor` | Ciphertensor tests |
| `--mpp` | MPP tests |
| `--mpa` | MPA tests |
| `--mpu` | MPU tests |
| `--ir-passes` | Compiler IR pass tests |
| `--stdlib-builtin` | Secure stdlib tests |
| `--lattiseq` | Lattiseq CKKS tests |
| `--mhe` | MHE protocol tests |
| `--lin-alg` | Linear algebra tests |
| `--lin-reg` | Linear regression tests |
| `--log-reg` | Logistic regression tests |
| `--lsvm` | Linear SVM tests |
| `--neural-net` | Neural network tests |
| `--mi` | Multiple imputation tests |
| `--pca` | PCA tests |
| `--king` | KING kinship tests |
| `--gwas` | GWAS tests (all variants) |
| `--credit-score` | Credit score application |
| `--dti` | Drug-target interaction |
| `--opal` | OPAL metagenomics |
| `--ganon` | GANON classification |
| `--genotype-imputation` | Genotype imputation |
| `--mnist` | MNIST classification |
| `--ablation` | Ablation studies |

```bash
# Run all tests locally
sequre run scripts/invoke.codon run-tests --local --all

# Run only sharetensor unit tests
sequre run scripts/invoke.codon run-tests --local --sharetensor

# Benchmark KING kinship locally
sequre run scripts/invoke.codon run-benchmarks --local --king
```

---

## Execution modes

### Local mode (`@local`)

All parties run as forked processes on a single machine, communicating via UNIX sockets. Ideal for development and testing.

```python
from sequre.runtime import local

@local
def my_protocol(mpc):
    # Each forked process has its own mpc.pid (0, 1, 2, ...)
    ...

my_protocol()
```

### Distributed mode (`mpc()`)

Each party runs as a separate process on a separate machine, communicating via TCP/IP with mutual TLS.

```python
from sequre.runtime import mpc as init_mpc

mpc = init_mpc()
# Parse party ID and flags from sys.argv
# mpc.pid, mpc.mhe, etc. are initialized
```

Run each party on its machine:

```bash
# On machine 1 (trusted dealer)
./my_protocol 0

# On machine 2 (compute party 1)
./my_protocol 1

# On machine 3 (compute party 2)
./my_protocol 2
```

---

## Network configuration

_Defined in `stdlib/sequre/settings.codon`_

| Setting | Default | Description |
|---|---|---|
| `DATA_SHARING_PORT` | `9999` | Port for initial data sharing |
| `COMMUNICATION_PORT` | `9000` | Base port for inter-party communication. Subsequent connections use incrementing ports. |
| `NETWORK_DELAY_TIME` | `0` | Simulated network delay in microseconds per `NETWORK_DELAY_THRESHOLD` bytes |
| `NETWORK_DELAY_THRESHOLD` | `5000000` | Bytes threshold for delay simulation |
| `MPC_INT_SIZE` | `192` | Big-integer size for MPC (128, 192, or 256 bits) |
| `LATTISEQ_INT_SIZE` | `512` | Big-integer size for Lattiseq operations |

### IP addresses

Set the `SEQURE_CP_IPS` environment variable to configure party IP addresses:

```bash
export SEQURE_CP_IPS="192.168.1.10,192.168.1.11,192.168.1.12"
```

---

## TLS configuration

_Defined in `stdlib/sequre/network/common.codon`_

Sequre uses **mutual TLS** for all INET (TCP/IP) connections. Both parties in a connection authenticate each other via X.509 certificates signed by a shared CA.

### Certificate environment variables

| Variable | Default | Description |
|---|---|---|
| `SEQURE_CERT_DIR` | `"certs"` | Directory containing all certificate files |
| `SEQURE_CA_CERT_FILE` | `"ca.pem"` | CA certificate filename |
| `SEQURE_PARTY_CERT_FILE` | `"cp{pid}.pem"` | Per-party certificate filename pattern |
| `SEQURE_PARTY_KEY_FILE` | `"cp{pid}-key.pem"` | Per-party private key filename pattern |

### Generating certificates

!!! warning "Development/testing only"
    The provided script generates a self-signed CA and party certificates for **local development and testing**. Do not use it as a production CA or enrollment workflow.

    In production, provision certificates via your own PKI pipeline (enterprise PKI, HashiCorp Vault, cloud/private CA, etc.). Rotate party certificates regularly and keep CA private keys outside Sequre application hosts.

Use the provided script to generate a CA and per-party certificates:

```bash
bash scripts/generate_certs.sh
```

This creates:

- `certs/ca.pem` — CA certificate
- `certs/cp0.pem`, `certs/cp0-key.pem` — Party 0 cert and key
- `certs/cp1.pem`, `certs/cp1-key.pem` — Party 1 cert and key
- `certs/cp2.pem`, `certs/cp2-key.pem` — Party 2 cert and key

### Disabling TLS

Set `SEQURE_USE_TLS=0` to disable TLS for INET channels. **This renders communication insecure** and should only be used for debugging.

!!! warning
    UNIX sockets (used in local mode) are always unencrypted. They rely on OS-level process isolation for security.

---

## Socket internals

_Defined in `stdlib/sequre/network/socket.codon`_

The `CSocket` class abstracts both UNIX and INET sockets:

```python
# INET socket
sock = CSocket(ip_address="192.168.1.10", port="9000")

# UNIX socket
sock = CSocket(unix_file_address="/tmp/sequre_cp0_cp1.sock")
```

| Method | Description |
|---|---|
| `socket()` | Create the underlying OS socket |
| `bind()` | Bind to the configured address |
| `connect()` | Connect to the remote party |
| `open_channel()` | Bind, listen, accept, and TLS handshake (server side) |
| `init_ssl_ctx(pid, is_server)` | Initialize TLS context with certificates |
| `ssl_handshake_connect()` | Perform TLS handshake as client |
| `ssl_handshake_accept()` | Perform TLS handshake as server |

### Connection retry

_Defined in `stdlib/sequre/network/connect.codon`_

The `connect()` function retries up to 100 times with ~1 second delays, handling transient network issues during party startup:

```python
from sequre.network.connect import connect
connect(socket)  # retries automatically
```

---

## Debug mode

Set `DEBUG = True` in `stdlib/sequre/settings.codon` (or recompile with the debug flag) to enable verbose logging of network operations, cost estimates, and protocol steps.
