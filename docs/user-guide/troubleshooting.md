# Troubleshooting

## Debug vs release mode

!!! warning "Sequre runs in debug mode by default"
    By default, `sequre run` and `sequre build` compile in **debug mode**. This enables full backtraces on failure but produces significantly slower code. **Always use `-release` for anything other than debugging.**

```bash
# Debug mode (default) — slow, with backtraces
sequre run my_protocol.codon

# Release mode — fast, production-ready
sequre -release run my_protocol.codon

# Building a release binary
sequre -release build my_protocol.codon -o my_protocol
```

The `-release` flag must come **immediately after** `sequre` (before `run` or `build`).

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

# macOS (Darwin builds are currently disabled)
# brew install gmp
# export SEQURE_GMP_PATH=/opt/homebrew/lib/libgmp.dylib
```

### OpenSSL (libssl / libcrypto)

The Sequre launcher auto-detects common OpenSSL paths on Linux. If auto-detection fails, install OpenSSL and point to it manually:

```bash
# Linux
sudo apt install libssl-dev
export SEQURE_OPENSSL_PATH=/usr/lib/x86_64-linux-gnu/libssl.so.3
export SEQURE_LIBCRYPTO_PATH=/usr/lib/x86_64-linux-gnu/libcrypto.so.3

# macOS (Darwin builds are currently disabled)
# brew install openssl
# export SEQURE_OPENSSL_PATH=/opt/homebrew/opt/openssl/lib/libssl.dylib
# export SEQURE_LIBCRYPTO_PATH=/opt/homebrew/opt/openssl/lib/libcrypto.dylib
```

### libpython (for `@python` interop)

The `@python` decorator enables calling Python code from Sequre (see [Python interoperability](#python-interoperability) below). It requires `libpython` to be available at runtime.

In case of errors like `libpython3.x.so: cannot open shared object file`, set the `CODON_PYTHON` environment variable:

```bash
# Linux
export CODON_PYTHON=/usr/lib/x86_64-linux-gnu/libpython3.12.so

# macOS (Darwin builds are currently disabled)
# export CODON_PYTHON=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.12.dylib
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

- Set `CODON_PYTHON` to point to the Python shared library (see [above](#libpython-for-python-interop))
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

## `--use-ring` instability

The `--use-ring` flag (ring modulus instead of field modulus) is currently unstable. Division and square root operations may produce incorrect results approximately 50% of the time due to a modulus-switching bug. Avoid this flag for production workloads.

---

## Plugin not found

In case of `cannot find plugin 'sequre'`, ensure:

1. Sequre is installed in the Codon plugins directory: `$HOME/.codon/lib/codon/plugins/sequre/`
2. The directory contains `plugin.toml`, `build/`, and `stdlib/`
3. If using `codon` directly instead of the `sequre` launcher, pass `-plugin sequre`:

```bash
codon run -plugin sequre my_protocol.codon
```

In case of `cannot find plugin 'seq'`, add the additional `-plugin seq` flag to the run or build command.
