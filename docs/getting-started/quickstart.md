# Quickstart

## Prerequisites

- **Linux** (x86_64). macOS (Darwin) builds are currently disabled.

## 1. Install Sequre

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/0xTCG/sequre/develop/scripts/install.sh)"
```

This installs Codon and Sequre to `~/.sequre` and adds it to your `PATH`.

??? note "Manual install"
    ```bash
    mkdir -p $HOME/.sequre
    curl -L https://github.com/exaloop/codon/releases/download/v0.17.0/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz \
      | tar zxvf - -C $HOME/.sequre --strip-components=1
    curl -L https://github.com/0xTCG/sequre/releases/latest/download/sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz \
      | tar zxvf - -C $HOME/.sequre
    export PATH=$HOME/.sequre/bin:$PATH
    ```

## 2. Run first example

```bash
sequre examples/addmul.codon --local
```

!!! info "Compilation takes a moment"
    Sequre programs compile to native machine code, so the first run may take a few minutes. The launcher shows compilation progress by default.

This forks three processes (a trusted dealer + two compute parties) and runs a simple addition-and-multiplication benchmark on secret-shared data.

Expected output:
```
CP0:    addmul: 0
CP0:    innerprod: 0
CP1:    addmul: 471
CP2:    addmul: 471
CP1:    innerprod: 32
CP2:    innerprod: 32
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
