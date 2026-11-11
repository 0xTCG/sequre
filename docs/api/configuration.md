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

### Cryptography

| Variable | Default | Description |
|---|---|---|
| `SEQURE_SIGMA_SMUDGE` | `1024` | Standard deviation of the noise each party adds to its share of a collective decryption (at least 3.2). Larger values hide more of the ciphertext noise at the cost of precision; see [Security assumptions](../crypto-corner/security-assumptions.md#noise-and-masks-in-the-mhe-protocols). |

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

`MPC_INT_SIZE` sets the fixed-point precision: F = 32 fractional bits at 128 and 192 bits, F = 64 at 256 bits. Decor's
nonlinear functions compute the function itself, so their error comes from the fixed-point encodings ($2^{-F}$) and,
at F = 64, from the plaintext evaluations of their terms in double precision ($2^{-52}$ of their size), which then
dominate: the 256-bit build gains about 4 to 6 decimal digits, and its default intervals are wider (see
[Nonlinear Functions](../user-guide/nonlinear-functions.md#precision)).

### Nonlinear functions

| Constant | F = 32 | F = 64 | Description |
|---|---|---|---|
| `DECOR_INTERVAL_EXP` | (−8, 8) | (−20, 20) | Decor exp's default interval |
| `DECOR_INTERVAL_SIGMOID` | (−10, 10) | (−15, 15) | Decor sigmoid's default interval |
| `DECOR_INTERVAL_TANH` | (−5, 5) | (−8, 8) | Decor tanh's default interval |
| `DECOR_INTERVAL_SINH` | (−5, 5) | (−8, 8) | Decor sinh's and cosh's default interval |
| `DECOR_TOLERANCE` | $2^{-12}$ | $2^{-28}$ | Admissibility: the largest accepted error bound, relative to $\max(1, \max \lvert f \rvert)$ on the interval |
| `DECOR_DIV_DEFAULT_NITER` | 4 | 5 | Goldschmidt iterations of Decor's division |
| `DECOR_DIV_DEFAULT_METHOD` | `"goldschmidt"` | | Decor's inverse: `"goldschmidt"` (49 rounds at F = 32; a division x / y adds the product with x and its truncation: 51) or `"newton"` (55) |
| `DECOR_SIGMOID_COMBINER` | `"select"` | | Decor's sigmoid and tanh: the addition theorem with one division (`"select"`), or exp and an inverse (`"exp"`) |
| `CHEBYSHEV_DEGREE` | 20 | | Default degree of the Chebyshev method |
| `NONLINEAR_FOURIER_DEGREE` | 16 | | Default degree of the Fourier method |
| `NONLINEAR_TAYLOR_DEGREE` | 8 | | Default degree of the Taylor method |
| `NONLINEAR_SOFTMAX_WIDTH` | 16 | | softmax evaluates $e^z$ on $(-W, 0]$ and saturates below ($e^{-16} \approx 1.1 \times 10^{-7}$) |
| `LOSS_LOG_INTERVAL` | (1e-4, 1) | | The cross-entropies' log interval (`Sequential.compile(loss_interval=...)`) |
| `PACKING_SIZE` | 64 | 128 | Bits per register in bitwise (XOR) sharing (= `MPC_NBIT_K`) |

### Lattiseq constants

| Constant | Value | Description |
|---|---|---|
| `LATTISEQ_MAX_LOGN` | 17 | Maximum polynomial ring degree ($\log N$) |
| `LATTISEQ_ROTATION_KEYS_COUNT` | 64 | Number of rotation keys generated |
| `LATTISEQ_DEFAULT_SIGMA` | 3.2 | Gaussian error distribution standard deviation |
| `LATTISEQ_SIGMA_SMUDGE` | `SEQURE_SIGMA_SMUDGE` (1024) | Noise standard deviation of collective-decryption shares |
| `MHE_MPC_SWITCH_LAMBDA` | 128 | Statistical security parameter of the HE → MPC switch masks |
| `LATTISEQ_EXACT_INT_SIZE` | 256 (512 if `MPC_INT_SIZE` is 256) | Integer width of the exact encoder (`exact_encoder.codon`) |

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
