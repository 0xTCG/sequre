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
sequre examples/local_run.codon
```

!!! info "Compilation takes a moment"
    Sequre programs compile to native machine code, so the first run may take a few minutes. The launcher shows compilation progress by default.
    ```

This forks three processes (a trusted dealer + two compute parties) and runs the `mult3` micro-benchmark from Hastings et al.

Expected output:
```
CP0:    mult3: 471
CP1:    mult3: 471
CP2:    mult3: 471
```

The result `7*13 + 13*19 + 7*19 = 471` was computed entirely on secret-shared data — no party ever saw the raw inputs of another.

## What just happened?

Here is `examples/local_run.codon`:

```python
from hastings import mult3
from sequre import local, Sharetensor as Stensor


@local
def mult3_local(mpc, a: int, b: int, c: int):
    a_stensor = Stensor.enc(mpc, a)
    b_stensor = Stensor.enc(mpc, b)
    c_stensor = Stensor.enc(mpc, c)

    print(f"CP{mpc.pid}:\tmult3: "
          f"{mult3(mpc, a_stensor, b_stensor, c_stensor).reveal(mpc)}")


mult3_local(7, 13, 19)
```

Key concepts:

1. **`@local`** — Forks the process into N parties (default 3) and injects an `mpc` context into each.
2. **`Stensor.enc(mpc, value)`** — Secret-shares a plaintext integer into additive shares distributed across parties.
3. **`mult3(mpc, a, b, c)`** — A function annotated with `@sequre` (in `hastings.codon`). The compiler plugin automatically rewrites `a * b`, `b * c`, etc. into Beaver-triple secure multiplications.
4. **`.reveal(mpc)`** — Reconstructs the secret by combining shares from all parties.

!!! tip "Execution modes"
    `@local` is one of three runtime decorators. Use `@online` for distributed runs, or `@main` to let the user control the mode via CLI (`--local` for local, online otherwise). See [Running Distributed](../user-guide/running-distributed.md#execution-modes) for details.

## Run vs. build

The launcher supports both Codon execution modes:

```bash
# JIT compile and run immediately (default)
sequre examples/local_run.codon

# Compile to a standalone binary
sequre build examples/local_run.codon -o local_run
./local_run
```

## Release vs debug mode

> **Important:** Sequre compiles in **debug mode by default** (with backtraces). Always use `-release` for production and benchmarks — it is significantly faster.

```bash
# Debug mode (default) — slow, with full backtraces on failure
sequre run examples/local_run.codon

# Release mode — fast, production-ready
sequre run -release examples/local_run.codon

# Building a release binary
sequre build -release examples/local_run.codon -o local_run
```

## Next steps

- **[Basic MPC Tutorial](../tutorials/basic-mpc.md)** — Deeper walkthrough of additive secret sharing and Sharetensor.
- **[Transitioning to MHE](../tutorials/transition-mhe.md)** — When and how to switch to homomorphic encryption with Shechi.
- **[Configuration](../api/configuration.md)** — Environment variables for network, TLS, and parameter tuning.
