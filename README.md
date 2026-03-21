# Sequre

### [Docs](https://0xTCG.github.io/sequre/)  ·  [Papers](#citations)  ·  [Examples](examples/)

## What is Sequre?

Sequre is a statically compiled, Pythonic framework for building secure computation pipelines — combining secure multiparty computation (MPC), homomorphic encryption (HE), and multiparty homomorphic encryption (MHE) in a single, high-performance system.

Write Python-like code; the Sequre compiler handles secret sharing, encrypted arithmetic, and inter-party communication automatically. Programs compile to native machine code via [Codon](https://github.com/exaloop/codon) with no runtime interpreter overhead.

## Goals

- 🐍 **Pythonic**: Write secure computation protocols in familiar Python syntax — no cryptographic boilerplate
- 🚀 **Fast**: Compiled to native code; outperforms interpreter-based MPC frameworks by orders of magnitude
- 🔀 **Unified**: MPC + HE + MHE in one framework — switch between schemes within a single protocol
- 🧩 **Batteries included**: Built-in linear algebra, statistics, machine learning (linear/logistic regression, PCA, SVM), and GWAS pipelines
- 🔒 **Secure by default**: Mutual TLS between parties, automatic key management, secret-shared arithmetic

## Quick start

**Supported platforms:** Linux (x86_64, aarch64) and macOS (x86_64, Apple Silicon).

Install [Codon](https://github.com/exaloop/codon), then install Sequre:

```bash
mkdir -p $HOME/.codon && \
  curl -L https://github.com/exaloop/codon/releases/download/v0.17.0/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon --strip-components=1

curl -L https://github.com/0xTCG/sequre/releases/download/v0.0.20-alpha/sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon

export PATH=$HOME/.codon/bin:$PATH
```

Run your first secure protocol:

```bash
git clone https://github.com/0xTCG/sequre.git && cd sequre
sequre examples/local_run.codon
```

Or compile to a binary:

```bash
sequre build examples/local_run.codon -o local_run
./local_run
```

## Examples

### Local execution

Use the `@local` decorator — Sequre forks one process per party on your machine, communicating via UNIX sockets:

`local_run.codon`:
```python
from sequre import sequre, local, Sharetensor as Stensor

@sequre
def muls(mpc, a, b, c):
    return a * b + b * c + a * c

@local
def mul_local(mpc, a: int, b: int, c: int):
    a_enc = Stensor.enc(mpc, a)
    b_enc = Stensor.enc(mpc, b)
    c_enc = Stensor.enc(mpc, c)
    print(f"CP{mpc.pid}:\t{muls(mpc, a_enc, b_enc, c_enc).reveal(mpc)}")

mul_local(7, 13, 19)
```

```bash
sequre local_run.codon
```

### Distributed execution

Use `mpc()` — each party runs as a separate process on a separate machine:

`online_run.codon`:
```python
from sequre import mpc, sequre, Sharetensor as Stensor

@sequre
def muls(mpc, a, b, c):
    return a * b + b * c + a * c

mpc = mpc()

a = Stensor.enc(mpc, 7)
b = Stensor.enc(mpc, 13)
c = Stensor.enc(mpc, 19)
print(f"CP{mpc.pid}:\t{muls(mpc, a, b, c).reveal(mpc)}")
```

```bash
# On each machine:
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre online_run.codon <pid>
```

Distributed mode requires mutual TLS certificates. Sequre handles MHE/MPC key management automatically, but **TLS certificates are your responsibility**. For development, generate test certificates with `scripts/generate_certs.sh`. For production, use your own CA — see [TLS configuration](https://0xTCG.github.io/sequre/user-guide/running-distributed/#tls-configuration) in the docs.

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

### Release vs debug mode

> **Important:** Sequre compiles in **debug mode by default** (with backtraces). Always use `-release` for production and benchmarks — it is significantly faster.

```bash
# Debug mode (default) — slow, with full backtraces on failure
sequre run my_protocol.codon

# Release mode — fast, production-ready
sequre -release run my_protocol.codon

# Building a release binary
sequre -release build my_protocol.codon -o my_protocol
```

The `-release` flag must come immediately after `sequre`, before `run` or `build`.

## Documentation

Please see [0xTCG.github.io/sequre](https://0xTCG.github.io/sequre/) for in-depth documentation, including the [API reference](https://0xTCG.github.io/sequre/api/), [tutorials](https://0xTCG.github.io/sequre/tutorials/), and [network/TLS setup](https://0xTCG.github.io/sequre/user-guide/running-distributed/).

Alternatively, you can [build from source](https://0xTCG.github.io/sequre/getting-started/build-from-source/).

## Citations

If you use Sequre or Shechi in your research, please cite the relevant paper(s):

- **Shechi** (USENIX Security 2025):  
  Smajlović H, Froelicher D, Shajii A, Berger B, Cho H, Numanagić I.  
  [Shechi: a secure distributed computation compiler based on multiparty homomorphic encryption.](https://dl.acm.org/doi/10.5555/3766078.3766473)  
  *34th USENIX Security Symposium*, 2025.

- **Sequre** (Genome Biology 2023):  
  Smajlović H, Shajii A, Berger B, Cho H, Numanagić I.  
  [Sequre: a high-performance framework for secure multiparty computation enables biomedical data sharing.](https://link.springer.com/article/10.1186/s13059-022-02841-5)  
  *Genome Biology*, 2023.

- **Sequre** (IEEE ISBI 2022):  
  Smajlović H, Shajii A, Berger B, Cho H, Numanagić I.  
  [Sequre: a high-performance framework for rapid development of secure bioinformatics pipelines.](https://ieeexplore.ieee.org/document/9835664)  
  *IEEE International Symposium on Biomedical Imaging*, 2022.

## Acknowledgements

This project was supported by:

- 🇺🇸 National Science Foundation (NSF)
- 🇺🇸 National Institutes of Health (NIH)
- 🇨🇦 Natural Sciences and Engineering Research Council (NSERC)
- 🇨🇦 Canada Research Chairs
- 🇨🇦 Canada Foundation for Innovation
- 🇨🇦 B.C. Knowledge Development Fund

Built on [Codon](https://github.com/exaloop/codon) and a reimplementation of [Lattigo](https://github.com/tuneinsight/lattigo).
