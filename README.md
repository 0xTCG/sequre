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

Install Sequre (includes Codon):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/0xTCG/sequre/develop/scripts/install.sh)"
```

This installs to `~/.sequre` and adds it to your `PATH`. See [quickstart](https://0xtcg.github.io/sequre/getting-started/quickstart/) for manual install and building from source.

### Run example

```bash
git clone --depth 1 https://github.com/0xTCG/sequre.git && cd sequre
sequre examples/addmul.codon
```

> **Note:** The first compilation may take a minute — Sequre programs compile to native code. The launcher shows compilation progress by default.

Or compile to a binary:

```bash
sequre build examples/addmul.codon -o addmul
./addmul
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

The [examples/](examples/) directory contains self-contained programs that demonstrate secure-computation workflows. Each generates its own synthetic data, runs locally, and prints results — no external datasets or configuration files needed. For full production pipelines, see [applications/](applications/).

| Example | File | Domain | What it shows |
|---|---|---|---|
| Simple expression | `examples/addmul.codon` | Intro | Addition and multiplication example — local with `--local` flag, online otherwise |
| Hastings benchmarks | `examples/hastings.codon` | Benchmarks | `mult3`, `innerprod`, `xtabs` micro-benchmarks |
| Credit scoring | `examples/credit_scoring.codon` | Finance | Secure neural-network classification with `MPU` partitioning |
| Genetic kinship | `examples/genetic_kinship.codon` | Genomics | Pairwise kinship estimation on MHE-encrypted genotype data |
| Linear regression | `examples/linear_regression.codon` | Healthcare | Multi-hospital model training with `MPU` and `LinReg` |
| One algorithm, many types | `examples/one_algorithm_many_types.codon` | End-to-end | Same pairwise `l2` on `ndarray`, `Sharetensor`, and `MPU` |
| Loading private data | `examples/collective_load.codon` | Deployment | Real-world data loading with `MPU.collective_load` (MHE) and `Sharetensor.collective_load` (MPC) |

> **Note:** The examples above use **synthetic data shared from a trusted dealer** for quick experimentation and testing. In real-world deployments, each party holds its own private data on disk and loads it into the secure computation via `collective_load`. See [`examples/collective_load.codon`](examples/collective_load.codon) for a complete working example and the [Loading Private Data](https://0xTCG.github.io/sequre/tutorials/loading-private-data/) tutorial for the full guide.

Run any example locally:

```bash
sequre -release examples/addmul.codon --local
sequre -release examples/hastings.codon --local
sequre -release examples/credit_scoring.codon --local
sequre -release examples/genetic_kinship.codon --local
sequre -release examples/linear_regression.codon --local
sequre examples/one_algorithm_many_types.codon --local
sequre -release examples/collective_load.codon --local
```

### Dispatching a Sequre program (`@main`)

The `@main` decorator is the entry point for every Sequre program. It sets up the MPC/MHE runtime environment and injects an `mpc` context as the first argument. The execution mode is controlled via CLI: pass `--local` to fork all parties on one machine, or omit it to run in distributed (online) mode.

> **Important:** The `@main`-decorated function is the **dispatcher** — it must be called **exactly once** at module level. All secure computation happens inside (or is called from) this function.

```python
from sequre import sequre, main, Sharetensor as Stensor

@sequre
def my_protocol(mpc, a, b, c):
    return a * b + b * c + a * c

@main
def main_call(mpc, a, b, c):
    a_enc = Stensor.enc(mpc, a)
    b_enc = Stensor.enc(mpc, b)
    c_enc = Stensor.enc(mpc, c)
    
    result = my_protocol(mpc, a_enc, b_enc, c_enc)
    print(f"CP{mpc.pid}:\tresult: {result.reveal(mpc)}")

if __name__ == "__main__":
    main_call(7, 13, 19)
```

```bash
# Local (all parties on one machine — development & testing):
sequre my_protocol.codon --local

# Distributed (each party is a separate process/machine — production):
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre my_protocol.codon <pid>
```

The [MPC instance](https://0xtcg.github.io/sequre/api/mpc-instance/) provides access to MPC/MHE essentials (party state, PRG streams, network sockets, and sub-modules for arithmetic, fixed-point, boolean, polynomial, and MHE operations etc.).

> **Note:** When working with many local runs, the socket files (`sock.*`) — needed for local communication — may collide between runs and cause connection issues. Delete stale files with `rm sock.*`.

Distributed mode requires mutual TLS certificates. Sequre handles MHE/MPC key management automatically, but **does not handle TLS certificate creation/maintenance**. For testing, generate test certificates with `scripts/generate_certs.sh`. For production, use a secure CA — see [TLS configuration](https://0xTCG.github.io/sequre/user-guide/running-distributed/#tls-configuration).

Sequre also provides lower-level `@local` and `@online` decorators for hard-coding the execution mode --- see the [documentation](https://0xTCG.github.io/sequre/api/decorators/) --- but `@main` covers both use-cases.

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
