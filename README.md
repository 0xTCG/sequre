# Sequre

### [Docs](https://0xTCG.github.io/sequre/)  ·  [Papers](#citations)  ·  [Examples](examples/)

## What is Sequre?

Sequre is a statically compiled, Pythonic framework for building secure computation pipelines — combining secure multiparty computation (MPC), homomorphic encryption (HE), and multiparty homomorphic encryption (MHE) in a single, high-performance system.

Write Python-like code; the Sequre compiler handles encrypted arithmetic and inter-party communication automatically. Programs compile to native machine code via [Codon](https://github.com/exaloop/codon) with no runtime interpreter overhead.

## Goals

- 🐍 **Pythonic**: Write secure computation protocols in familiar Python syntax — no cryptographic boilerplate
- 🚀 **Fast**: Compiled to native code; outperforms interpreter-based MPC frameworks by orders of magnitude
- 🔀 **Unified**: MPC + HE + MHE in one framework — switch between schemes within a single protocol
- 🧩 **Batteries included**: Built-in linear algebra, statistics, machine learning (linear/logistic regression, PCA, SVM, neural networks), and biomedical pipelines (GWAS, DTI, Metagenomic binning, Kinship estimation)
- 🔒 **Secure by default**: Mutual TLS between parties, automatic key management, secured PRG streams

## Quick start

**Supported platforms:** Linux (x86_64). macOS (Darwin) builds are currently disabled.

Install [Codon](https://github.com/exaloop/codon), then install Sequre:

```bash
mkdir -p $HOME/.codon && \
  curl -L https://github.com/exaloop/codon/releases/download/v0.17.0/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon --strip-components=1

curl -L https://github.com/0xTCG/sequre/releases/latest/download/sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon

export PATH=$HOME/.codon/bin:$PATH
```

Run example:

```bash
git clone --depth 1 https://github.com/0xTCG/sequre.git && cd sequre
sequre examples/local_run.codon
```

> **Note:** The first compilation may take a minute — Sequre programs compile to native code. The launcher shows compilation progress by default.

Or compile to a binary:

```bash
sequre build examples/local_run.codon -o local_run
./local_run
```

> **Note:** Make sure to delete sockets (`rm sock.*`) if running a **local run** pre-built binary. `sequre` command does this automatically, otherwise, but built binaries do not.

### Release vs debug mode

> **Important:** Sequre compiles in **debug mode by default** (with backtraces). Always use `-release` for production and benchmarks — it is significantly faster.

```bash
# Debug mode (default) — slow, with full backtraces on failure
sequre run my_protocol.codon

# Release mode — fast, production-ready
sequre run -release my_protocol.codon

# Building a release binary
sequre build -release my_protocol.codon -o my_protocol
```

## Examples

The [examples/](examples/) directory contains self-contained programs that demonstrate realistic secure-computation workflows. Each generates its own synthetic data, runs locally, and prints results — no external datasets or configuration files needed. For full production pipelines, see [applications/](applications/). For correctness and performance coverage, see [tests/](tests/).

| Example | File | Domain | What it shows |
|---|---|---|---|
| Local execution | `examples/local_run.codon` | Intro | `@local` decorator — forks parties on one machine |
| Online execution | `examples/online_run.codon` | Intro | `@online` decorator — wraps `mpc()` lifecycle |
| Online (controlled) | `examples/online_run_controlled.codon` | Intro | `mpc()` manual setup for distributed runs |
| Main (CLI-controlled) | `examples/main_run.codon` | Intro | `@main` decorator — local with `--local` flag, online otherwise |
| Hastings benchmarks | `examples/hastings.codon` | Benchmarks | `mult3`, `innerprod`, `xtabs` micro-benchmarks with `@main` |
| Credit scoring | `examples/credit_scoring.codon` | Finance | Secure neural-network classification with `MPU` partitioning |
| Genetic kinship | `examples/genetic_kinship.codon` | Genomics | Pairwise kinship estimation on MHE-encrypted genotype data |
| Linear regression | `examples/linear_regression.codon` | Healthcare | Multi-hospital model training with `MPU` and `LinReg` |
| One algorithm, many types | `examples/one_algorithm_many_types.codon` | End-to-end | Same pairwise `l2` on `ndarray`, `Sharetensor`, and `MPU` |

Run any example locally:

```bash
sequre examples/local_run.codon
sequre examples/hastings.codon --local
sequre examples/credit_scoring.codon --local
sequre examples/genetic_kinship.codon --local
sequre examples/linear_regression.codon --local
sequre examples/one_algorithm_many_types.codon --local
```

### Execution modes

Sequre provides three runtime decorators for different execution scenarios, plus a manual `mpc()` function for full control:

| Decorator | When to use |
|---|---|
| `@local` | All parties forked on **one machine** — ideal for development and testing |
| `@online` | Each party is a **separate process/machine** — production distributed runs |
| `@main` | **CLI-controlled**: runs locally when `--local` flag is passed, otherwise runs online |

All three inject an `mpc` context as the first argument — no need to pass it at the call site.

### Local execution (`@local`)

Forks one process per party, communicating via UNIX sockets:

`local_run.codon`:
```python
from sequre import local, Sharetensor as Stensor

@local
def mul3_local(mpc, a: int, b: int, c: int):
    a_stensor = Stensor.enc(mpc, a)
    b_stensor = Stensor.enc(mpc, b)
    c_stensor = Stensor.enc(mpc, c)
    mul3 = a_stensor * b_stensor + b_stensor * c_stensor + a_stensor * c_stensor
    print(f"CP{mpc.pid}:\tmul3: {mul3.reveal(mpc)}")

mul3_local(7, 13, 19)
```

```bash
sequre examples/local_run.codon
```

> **Note:** When working with many local runs, the socket files (`sock.*`)---needed for local communication---may collude in-between the runs and cause connection issues. Make sure to delete the stale files in that case `rm sock.*`.

### Online execution (`@online`)

Wraps the `mpc()` lifecycle in a decorator — each party runs as a separate process:

`online_run.codon`:
```python
from sequre import online, Sharetensor as Stensor

@online
def mul3_online(mpc, a: int, b: int, c: int):
    a_stensor = Stensor.enc(mpc, a)
    b_stensor = Stensor.enc(mpc, b)
    c_stensor = Stensor.enc(mpc, c)
    mul3 = a_stensor * b_stensor + b_stensor * c_stensor + a_stensor * c_stensor
    print(f"CP{mpc.pid}:\tmul3: {mul3.reveal(mpc)}")

mul3_online(7, 13, 19)
```

```bash
# On each machine:
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre online_run.codon <pid>
```

For full manual control without a decorator, use `mpc()` directly (see `online_run_controlled.codon`).

### Main execution (`@main`)

Lets the user control the execution mode via CLI: runs locally when `--local` is passed, otherwise runs online. Best for programs that should work in both modes:

`main_run.codon`:
```python
from sequre import main, Sharetensor as Stensor

@main
def mul3_main(mpc, a: int, b: int, c: int):
    a_stensor = Stensor.enc(mpc, a)
    b_stensor = Stensor.enc(mpc, b)
    c_stensor = Stensor.enc(mpc, c)
    mul3 = a_stensor * b_stensor + b_stensor * c_stensor + a_stensor * c_stensor
    print(f"CP{mpc.pid}:\tmul3: {mul3.reveal(mpc)}")

mul3_main(7, 13, 19)  # Runs locally if --local flag is passed, otherwise runs online
```

```bash
# Local:
sequre examples/main_run.codon --local

# Online (on each machine):
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre examples/main_run.codon <pid>
```

Distributed mode (online and main without `--local`) requires mutual TLS certificates. Sequre handles MHE/MPC key management automatically, but **does not handle the TLS certificates creation/maintenance**. For testing, generate test certificates with `scripts/generate_certs.sh`. For production, use secure CA — see [TLS configuration](https://0xTCG.github.io/sequre/user-guide/running-distributed/#tls-configuration).

### Writing secure functions

The `@sequre` decorator marks functions that operate on secret-shared or encrypted data. The compiler applies MPC/MHE optimizations automatically:

```python
from sequre import sequre, Sharetensor as Stensor

@sequre
def mult3(mpc, a, b, c):
    return a * b + b * c + a * c

@sequre
def innerprod(mpc, a, b):
    return a.dot(mpc, b, axis=0)
```

## Documentation

Please see [0xTCG.github.io/sequre](https://0xTCG.github.io/sequre/) for in-depth documentation, including the [API reference](https://0xTCG.github.io/sequre/api/), [tutorials](https://0xTCG.github.io/sequre/tutorials/), and [network/TLS setup](https://0xTCG.github.io/sequre/user-guide/running-distributed/).

## Citations

- **Shechi** (USENIX Security 2025):  
  Smajlović H, Froelicher D, Shajii A, Berger B, Cho H, Numanagić I.  
  [Shechi: a secure distributed computation compiler based on multiparty homomorphic encryption.](https://dl.acm.org/doi/10.5555/3766078.3766473)  
  *34th USENIX Security Symposium*, 2025.

- **Sequre** (Genome Biology 2023):  
  Smajlović H, Shajii A, Berger B, Cho H, Numanagić I.  
  [Sequre: a high-performance framework for secure multiparty computation enables biomedical data sharing.](https://link.springer.com/article/10.1186/s13059-022-02841-5)  
  *Genome Biology*, 2023.

## Acknowledgements

This project was supported by:

- 🇺🇸 National Science Foundation (NSF)
- 🇺🇸 National Institutes of Health (NIH)
- 🇨🇦 Natural Sciences and Engineering Research Council (NSERC)
- 🇨🇦 Canada Research Chairs
- 🇨🇦 Canada Foundation for Innovation
- 🇨🇦 B.C. Knowledge Development Fund

Built via [Codon](https://github.com/exaloop/codon).
