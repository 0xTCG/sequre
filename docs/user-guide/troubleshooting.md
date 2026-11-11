# Troubleshooting

## Debug vs release mode

!!! warning "Sequre runs in debug mode by default"
    By default, `sequre run` and `sequre build` compile in **debug mode**. This enables full backtraces on failure but produces significantly slower code. **Always use `-release` for anything other than debugging.**

```bash
# Debug mode (default) — slow, with backtraces
sequre run my_protocol.codon

# Release mode — fast, production-ready
sequre run -release my_protocol.codon

# Building a release binary
sequre build -release my_protocol.codon -o my_protocol
```

Use debug mode only when a backtrace diagnostics are needed. Switch to `-release` for benchmarks, production runs, and any performance-sensitive work.

---

## Missing shared libraries

### libgmp (GMP)

GMP is **bundled** with Sequre releases and auto-detected by the launcher. It checks bundled paths and common system locations.

If auto-detection fails, install GMP and set the path to it to `SEQURE_GMP_PATH` env variable (Sequre repository also provides these libraries in `external/GMP/lib` dir):

```bash
# Linux
sudo apt install libgmp-dev
export SEQURE_GMP_PATH=/usr/lib/x86_64-linux-gnu/libgmp.so

# macOS
brew install gmp
export SEQURE_GMP_PATH=/opt/homebrew/lib/libgmp.dylib
```

### OpenSSL (libssl / libcrypto)

The Sequre launcher auto-detects common OpenSSL paths on Linux and macOS. If auto-detection fails, install OpenSSL and point to it manually:

```bash
# Linux
sudo apt install libssl-dev
export SEQURE_OPENSSL_PATH=/usr/lib/x86_64-linux-gnu/libssl.so.3
export SEQURE_LIBCRYPTO_PATH=/usr/lib/x86_64-linux-gnu/libcrypto.so.3

# macOS
brew install openssl
export SEQURE_OPENSSL_PATH=/opt/homebrew/opt/openssl/lib/libssl.dylib
export SEQURE_LIBCRYPTO_PATH=/opt/homebrew/opt/openssl/lib/libcrypto.dylib
```

### libpython (for `@python` interop)

The `@python` decorator enables calling Python code from Sequre (see [Python interoperability](#python-interoperability) below). It requires `libpython` to be available at runtime.

The `sequre` launcher auto-detects `libpython` by querying `python3` on `PATH` (via `sysconfig`), so most setups need no configuration. If auto-detection fails (no `python3` on `PATH`, or an unusual install layout), or in case of errors like `libpython3.x.so: cannot open shared object file`, set the `CODON_PYTHON` environment variable manually:

```bash
# Linux
export CODON_PYTHON=/usr/lib/x86_64-linux-gnu/libpython3.12.so

# macOS
export CODON_PYTHON=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.12.dylib
```

!!! note
    `libpython` is **not required** unless the program uses the `@python` decorator. Most Sequre programs do not need it.

---

## Python interoperability

Sequre inherits Codon's `@python` decorator, which enables running native Python code (including any pip-installed package) from within a compiled Sequre program. This is useful for data loading, visualization, or calling libraries that don't have a Codon equivalent.

```python
@python
def load_data(path: str) -> List[List[float]]:
    import pandas as pd
    df = pd.read_csv(path)
    return df.values.tolist()

@local
def my_protocol(mpc):
    data = load_data("input.csv")
    # ... use data in secure computation
```

**Requirements:**

- `CODON_PYTHON` must point to the Python shared library — the launcher auto-detects this via `python3` on `PATH`; set it manually if detection fails (see [above](#libpython-for-python-interop))
- The `@python` function body runs in the system Python interpreter — any used packages must be installed in that Python environment
- Arguments and return values are automatically marshalled between Codon and Python types. Use standard types (`int`, `float`, `str`, `List`, `Dict`, `Tuple`) for the function signature

For more details, see [Codon's Python interoperability docs](https://docs.exaloop.io/codon/interoperability/python).

---

## TLS certificate errors

### Missing certificates

In case of connection errors in distributed mode, check if the certificate files exist:

```
certs/
  ca.pem              # CA certificate (shared by all parties)
  cp0.pem             # Party 0 certificate
  cp0-key.pem         # Party 0 private key
  cp1.pem             # Party 1 certificate
  cp1-key.pem         # Party 1 private key
  ...
```

Generate test certificates with:

```bash
./scripts/generate_certs.sh 3 ./certs
```

Override paths via environment variables if needed:

```bash
export SEQURE_CERT_DIR=/path/to/certs
export SEQURE_CA_CERT_FILE=my-ca.pem
```

See [TLS configuration](running-distributed.md#tls-configuration) for the full list of environment variables.

### Expired or invalid certificates

If connections fail with TLS handshake errors, check certificate validity:

```bash
openssl x509 -in certs/cp0.pem -noout -dates
```

The test script generates certificates valid for 365 days by default (configurable via `SEQURE_CERT_DAYS`).

### Disabling TLS for debugging

```bash
export SEQURE_USE_TLS=0
```

!!! danger
    This sends all inter-party communication **unencrypted**. Use only for local debugging, never in production.

---

## Connection failures

### Parties cannot connect

- Verify all parties use the same `SEQURE_CP_IPS` with IPs in the same order
- Check that ports `9000+` and `9999` are open between machines (see [network configuration](running-distributed.md#network-configuration))
- Ensure all parties start within a reasonable time window — the connection has a retry mechanism, but it will eventually time out

### UNIX socket errors (local mode)

Stale socket files (`sock.*`) from a previous crashed run can prevent new connections. The Sequre launcher cleans these up automatically on startup, but if running via `codon` directly:

```bash
rm -f sock.*
```

---

## Nonlinear functions refuse an interval or a degree

Before any communication, every party checks the nonlinear function's parameters (see
[Nonlinear Functions](nonlinear-functions.md#intervals-clipping-and-admissibility)). A failed check raises `ValueError`
at every party, so the parties stay in step. The messages, and what to do:

| Message (default build, F = 32) | Cause | Fix |
|---|---|---|
| `Decor exp on (-16, 16): h term 0 (dealer side, v in [-8, 24]) reaches 2.651e+10, beyond the fixed-point range 2.147e+09 (MPC_NBIT_K - MPC_NBIT_F - 1 = 31 integer bits); narrow the interval or use another method` | a value the protocol encodes (here a power of $e$ on the dealer's side) leaves the fixed-point range, $2^{K-F-1}$ | narrow the interval (exp up to about ±14 at F = 32), or use a 256-bit build (F = 64) |
| `Decor sigmoid on (-14, 14): the combiner's denominator falls to 8.31e-07, below the division's smallest divisor 9.54e-07 (2^-(F - 12)); narrow the interval or use another method` | sigmoid's or tanh's denominator gets too small to divide by on this interval | narrow the interval (sigmoid up to about ±12.3 at F = 32, tanh about ±6.5) |
| `Decor <function> on <interval>: the estimated worst-case error <bound> exceeds the tolerance <tolerance> times <scale> (max(1, max |f|) on the interval); narrow the interval, pass a larger tolerance or use another method` | `decor_error_bound` exceeds `tolerance` times $\max(1, \max \lvert f \rvert)$ on the interval | narrow the interval, pass a larger `tolerance`, or use another method |
| `nonlinear.exp (chebyshev) on (-20, 20): Clenshaw's intermediates reach 4.411e+09, beyond the fixed-point range 2.147e+09; narrow the interval or lower the degree` | the Chebyshev coefficients are too large for this interval and degree | narrow the interval or lower the degree |
| `nonlinear.sin: Decor's sin takes no interval`, `nonlinear.sin: clip applies to functions on an interval (Decor's sin takes none)`, `nonlinear.sin: Decor takes no degree` | Decor's sin, cos, tan and cot take any input (inv any admissible divisor), and Decor takes no degree | drop the argument, or pick a polynomial method |
| `nonlinear.sigmoid: the auto method picks the Chebyshev degree on field shares itself (method chebyshev takes a degree)` | sigmoid's and tanh's default, `"auto"`, chooses the degree of its field interpolant from Decor's error bound | drop the degree, or pass `method="chebyshev"` with it |
| `nonlinear.exp: the Taylor method takes no default interval (a Taylor polynomial is accurate near its centre only): pass one` | Taylor without an interval | pass an interval that suits the series (well inside its radius around the centre) |
| `via_chebyshev (naive) of degree 44 on (-8, 8): the estimated worst-case error 8.36 exceeds the tolerance 0.000244 ...`, `... the powers or the sum reach ...`, `... the scaling to [-1, 1] errs by up to ...` | the secure-powers evaluation of a Chebyshev or Taylor polynomial: large monomial coefficients, powers of a half-width above 1, or a very wide interval | lower the degree, narrow the interval, or use Clenshaw (Chebyshev) or Decor's evaluation |
| `via_decor: cannot run inside MPCEnv.delayed_truncation(): its internal truncations cannot be deferred` | division, square root, Decor and Chebyshev's recurrence truncate internally | call them outside the `delayed_truncation()` block |

Inputs outside the interval do not raise: with `clip=False` they give method-dependent results (for example, Decor's
bounded functions give $f(x - kp)$ on secure types). Pass `clip=True` when inputs may leave the interval. The
Chebyshev method checks only its range, not its accuracy: an interval that reaches a singularity (log on (0, 8)) is
accepted, and the interpolant is inaccurate near it.

---

## Plugin not found

In case of `cannot find plugin 'sequre'`, ensure:

1. Sequre is installed in the Codon plugins directory: `$HOME/.sequre/lib/codon/plugins/sequre/`
2. The directory contains `plugin.toml`, `build/`, and `stdlib/`
3. If using `codon` directly instead of the `sequre` launcher, pass `--plugin=sequre`:

```bash
codon run --plugin=sequre my_protocol.codon
```

In case of `cannot find plugin 'seq'`, add the additional `--plugin=seq` flag to the run or build command.
