# Quickstart

## Prerequisites

- **Linux** (x86_64, aarch64) or **macOS 15 or later** (Apple Silicon / arm64).

## 1. Install Sequre

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/0xTCG/sequre/develop/scripts/install.sh)"
```

This installs Codon and Sequre to `~/.sequre` and offers to add it to your `PATH`. The installer downloads both archives and checks them before it unpacks either, so a failed download leaves an existing install as it was. Its environment variables: `SEQURE_INSTALL_DIR` (another install directory), `SEQURE_ARCHIVE` and `SEQURE_CODON_ARCHIVE` (a URL or a local file in place of the default downloads), and `SEQURE_UPDATE_PATH=yes` or `no` (answers the `PATH` question; with no terminal to ask on, the installer leaves `PATH` alone and says so).

??? note "Manual install"
    The Sequre archive names the Codon release it was built for in `lib/codon/plugins/sequre/codon-version`:

    ```bash
    PLATFORM=$(uname -s | awk '{print tolower($0)}')-$(uname -m)
    D=$(mktemp -d)
    if curl -fL -o "$D/sequre.tar.gz" \
         https://github.com/0xTCG/sequre/releases/latest/download/sequre-$PLATFORM.tar.gz; then
      if CODON_VERSION=$(tar -xzOf "$D/sequre.tar.gz" lib/codon/plugins/sequre/codon-version 2>/dev/null); then
        mkdir -p $HOME/.sequre &&
          curl -fL https://github.com/exaloop/codon/releases/download/$CODON_VERSION/codon-$PLATFORM.tar.gz \
            | tar zxvf - -C $HOME/.sequre --strip-components=1 &&
          tar zxvf "$D/sequre.tar.gz" -C $HOME/.sequre &&
          export PATH=$HOME/.sequre/bin:$PATH
      else
        echo "This Sequre archive names no Codon release (Sequre v0.1.6 or earlier): use the installer above"
      fi
    fi
    rm -rf "$D"
    ```

    Unpack into an empty directory: over an earlier install, this keeps that install's files that the new release
    lacks (the installer replaces `lib/codon/stdlib`, `include/codon`, `lib/codon/plugins/sequre` and the old
    `lib/codon/plugins/seq` itself).

## 2. Run first example

```bash
sequre examples/addmul.codon --local
```

!!! info "Compilation takes a moment"
    Sequre programs compile to native machine code, so the first run may take a few minutes. The launcher shows compilation progress by default.

This forks three processes (a trusted dealer + two compute parties) and runs a simple addition-and-multiplication benchmark on secret-shared data.

Expected output:
```
CP0:    addmul: 0.0
CP0:    innerprod: 0.0
CP1:    addmul: 471.0
CP2:    addmul: 471.0
CP1:    innerprod: 32.0
CP2:    innerprod: 32.0
```

The result `7*13 + 13*19 + 7*19 = 471` was computed entirely on secret-shared data — no party ever saw the raw inputs of another.

## What just happened?

Here is `examples/addmul.codon`:

```python
from sequre import main, sequre, Sharetensor as Stensor

@sequre
def addmul(mpc, a, b, c):
    addmul_result = a * b + b * c + a * c
    print(f"CP{mpc.pid}:\taddmul: {addmul_result.reveal(mpc)}")

@sequre
def innerprod(mpc, a, b):
    innerprod_result = a.dot(mpc, b)
    print(f"CP{mpc.pid}:\tinnerprod: {innerprod_result.reveal(mpc)}")

@main
def run(mpc, a, b, c, x, y):
    a_enc = Stensor.enc(mpc, a)
    b_enc = Stensor.enc(mpc, b)
    c_enc = Stensor.enc(mpc, c)
    x_enc = Stensor.enc(mpc, x)
    y_enc = Stensor.enc(mpc, y)

    addmul(mpc, a_enc, b_enc, c_enc)
    innerprod(mpc, x_enc, y_enc)

if __name__ == "__main__":
    # No need to pass mpc argument when calling sequre method with a main decorator
    run(7, 13, 19, [1, 2, 3], [4, 5, 6])
```

Key concepts:

1. **`@main`** — The entry-point decorator. Pass `--local` to fork all parties on one machine, or omit it to run in distributed (online) mode.
2. **`Stensor.enc(mpc, value)`** — Secret-shares a plaintext integer into additive shares distributed across parties.
3. **`a_enc * b_enc`** — The `@sequre` compiler plugin automatically rewrites arithmetic on secret-shared data into Beaver-triple secure multiplications.
4. **`.reveal(mpc)`** — Reconstructs the secret by combining shares from all parties.

!!! tip "Execution modes"
    `@main` lets the user control the mode via CLI (`--local` for local, online otherwise). Sequre also provides lower-level `@local` and `@online` decorators for hard-coding the execution mode. See [Running Distributed](../user-guide/running-distributed.md#execution-modes) for details.

## Run vs. build

The launcher supports both Codon execution modes:

```bash
# JIT compile and run immediately (default)
sequre examples/addmul.codon --local

# Compile to a standalone binary
sequre build examples/addmul.codon -o addmul
./addmul --local
```

The binary runs without the launcher: on x86-64 Linux `sequre build` links Sequre's alignment shim into it. A program that uses Python interop also needs `CODON_PYTHON` (the launcher sets it for `sequre run`; see [libpython](../user-guide/troubleshooting.md#libpython-for-python-interop)). The binary refers to Sequre's libraries by absolute path, so another machine that runs it needs Sequre installed at the same path (see [Running Distributed](../user-guide/running-distributed.md#manual-mode-mpc)).

## Release vs debug mode

> **Important:** Sequre compiles in **debug mode by default** (with backtraces). Always use `-release` for production and benchmarks — it is significantly faster.

```bash
# Debug mode (default) — slow, with full backtraces on failure
sequre run examples/addmul.codon --local

# Release mode — fast, production-ready
sequre run -release examples/addmul.codon --local

# Building a release binary
sequre build -release examples/addmul.codon -o addmul
```

## Next steps

- **[Basic MPC Tutorial](../tutorials/basic-mpc.md)** — Deeper walkthrough of additive secret sharing and Sharetensor.
- **[Transitioning to MHE](../tutorials/transition-mhe.md)** — When and how to switch to homomorphic encryption with Shechi.
- **[Configuration](../api/configuration.md)** — Environment variables for network, TLS, and parameter tuning.
