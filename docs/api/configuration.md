# Configuration

Sequre's behavior is controlled through compile-time constants and environment variables.

## Environment variables

### Network & party setup

| Variable | Default | Description |
|---|---|---|
| `SEQURE_CP_IPS` | _(none)_ | Comma-separated IP addresses of all parties. Sets party count automatically. |
| `SEQURE_CP_COUNT` | `3` | Number of computing parties (including trusted dealer). Ignored if `SEQURE_CP_IPS` is set. |

### Library paths

| Variable | Default | Description |
|---|---|---|
| `SEQURE_GMP_PATH` | _(auto-detected)_ | Path to GMP shared library (`.so` on Linux, `.dylib` on macOS) |
| `SEQURE_OPENSSL_PATH` | _(auto-detected)_ | Path to OpenSSL `libssl` |
| `SEQURE_LIBCRYPTO_PATH` | _(auto-detected)_ | Path to OpenSSL `libcrypto` |

### TLS certificates

| Variable | Default | Description |
|---|---|---|
| `SEQURE_CERT_DIR` | `certs` | Directory containing TLS certificates |
| `SEQURE_CA_CERT_FILE` | `ca.pem` | CA certificate filename |
| `SEQURE_PARTY_CERT_FILE` | `cp{pid}.pem` | Per-party certificate pattern |
| `SEQURE_PARTY_KEY_FILE` | `cp{pid}-key.pem` | Per-party private key pattern |

### Runtime

| Variable | Default | Description |
|---|---|---|
| `CODON_BIN` | _(none)_ | Override the path to the `codon` binary used by the launcher |

## Compile-time constants

_Defined in `stdlib/sequre/settings.codon` and `stdlib/sequre/constants.codon`_

### Integer sizes

| Constant | Default | Description |
|---|---|---|
| `MPC_INT_SIZE` | `192` | Bit width for MPC integers. Options: `128`, `192`, `256`. |
| `LATTISEQ_INT_SIZE` | `512` | Bit width for Lattiseq big integers |

### MPC parameters (derived from `MPC_INT_SIZE`)

| Constant | 128-bit | 192-bit | 256-bit |
|---|---|---|---|
| `MPC_MODULUS_BITS` | 127 | 174 | 251 |
| `MPC_NBIT_K` | 64 | 64 | 128 |
| `MPC_NBIT_F` | 32 | 32 | 64 |
| `MPC_NBIT_V` | 28 | 64 | 56 |

- **`MPC_FIELD_SIZE`**: Mersenne prime field $2^k - c$ where $k$ = `MPC_MODULUS_BITS`
- **`MPC_RING_SIZE`**: Power-of-two ring $2^k$
- **`MPC_NBIT_K`**: The total bit-width in fixed-point representation
- **`MPC_NBIT_F`**: The fractional bit-width in fixed-point representation
- **`MPC_NBIT_V`**: The statistical security padding for fixed-point representation (needed only in comparison protocols on finite fields)

### Lattiseq constants

| Constant | Value | Description |
|---|---|---|
| `LATTISEQ_MAX_LOGN` | 17 | Maximum polynomial ring degree ($\log N$) |
| `LATTISEQ_ROTATION_KEYS_COUNT` | 64 | Number of rotation keys generated |
| `LATTISEQ_DEFAULT_SIGMA` | 3.2 | Gaussian error distribution standard deviation |

### Network

| Constant | Value | Description |
|---|---|---|
| `COMMUNICATION_PORT` | 9000 | Base port for inter-party communication |

### Security toggles

| Constant | Default | Description |
|---|---|---|
| `USE_TLS` | `True` | Enable TLS on all INET channels |
| `USE_SECURE_PRG` | `True` | Use AES-256-CTR CSPRNG instead of SFMT |
| `DEBUG` | `False` | Enable debug mode (enables caching, verbose output) |

!!! warning
    Setting `USE_TLS` or `USE_SECURE_PRG` to `False` disables critical security features and should only be done for debugging.

## Run-script flags

The list of `sequre` CLI arguments:

| Flag | Description |
|---|---|
| `--skip-mhe-setup` | Skip MHE key generation (useful when running MPC-only) |
| `--use-ring` | Use ring arithmetic ($\mathbb{Z}_{2^k}$) instead of field ($\mathbb{F}_p$) |
| `-release` | Enable release mode (better performance, no backtrace) |
