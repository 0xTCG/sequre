# Sequre & Shechi

**Sequre** is an end-to-end, statically compiled, performance-engineered framework for writing Pythonic secure computation pipelines. Built on [Codon](https://github.com/exaloop/codon), it compiles to native machine code while retaining a Python-like syntax.

**Sequre** is the superset framework. It provides:

- **Additive secret sharing** for fast secure multiparty computation (MPC).
- **Shechi**, the MHE sub-system that enables MPC via multiparty homomorphic encryption based on the CKKS scheme.
- A unified compiler plugin that automatically rewrites arithmetic on secure types into the correct cryptographic protocol calls.

## How Sequre and Shechi relate

| | Sequre | Shechi |
|---|---|---|
| **Scope** | Full framework (superset) | MHE sub-system within Sequre |
| **Primary mechanism** | Additive secret sharing over Mersenne-prime fields | Multiparty CKKS homomorphic encryption |
| **Core type** | `Sharetensor` | `Ciphertensor`, `MPU`, `MPP`, `MPA` |
| **Non-linear ops** | Beaver triples + network rounds | Local HE evaluation + collective key-switching |
| **Best for** | Integer/fixed-point MPC, smaller data scale | Floating-point, batched linear algebra, large scale ML |

## Architecture layers

Sequre is multi-layered to enable operating at different abstraction levels:

```
┌─────────────────────────────────────────────────────┐
│  @sequre / @local decorators                        │  ← Pythonic code
│  Sharetensor  ·  MPU (multiparty_union)             │
├─────────────────────────────────────────────────────┤
│  MPCEnv                                             │  ← Orchestrates MPC + MHE
│  ├─ arithmetic (Beaver triples)                     │
│  ├─ boolean / fp / polynomial                       │
│  └─ mhe (collective CKKS)                           │
├─────────────────────────────────────────────────────┤
│  Ciphertensor                                       │  ← Local HE tensor ops
│  CryptoParams · Evaluator · Encoder                 │
├─────────────────────────────────────────────────────┤
│  Lattiseq                                           │  ← CKKS engine (Lattigo in Codon)
│  Ring · RLWE · dCKKS · dRLWE                        │
└─────────────────────────────────────────────────────┘
```

**Layer 1 — High-level.** Write standard-looking Python. Annotate functions with `@sequre`; the compiler plugin rewrites operators (`+`, `*`, `@`, `==`, …) on secure types into protocol calls.

**Layer 2 — MPCEnv.** The runtime environment that holds party state, PRG streams, network sockets, and sub-modules for arithmetic, fixed-point, boolean, polynomial, and MHE operations.

**Layer 3 — Ciphertensor.** A local tensor of CKKS ciphertexts with operator overloading, automatic scale/level tracking, and transparent collective operations when depth limits are reached.

**Layer 4 — Lattiseq.** A ground-up Codon port of the Go *Lattigo* library. Provides ring arithmetic, RLWE key generation, CKKS encoding/encryption/evaluation, and distributed CKKS protocols (CKG, RKG, PCKS, Refresh).

## Supported applications

Sequre ships with production-grade implementations of:

| Application | Domain | Backend |
|---|---|---|
| GWAS | Genomics | MPC |
| KING | Genomics | MPC + MHE |
| Genotype Imputation | Genomics | MPC / MPP / MPU |
| Multiple Imputation (MI/MICE) | Statistics | MPC + MHE |
| Credit Scoring (Neural Net) | Finance / ML | MPC |
| MNIST | ML | MPC |
| DTI | Drug Discovery / ML | MPC + MHE |
| OPAL | Metagenomics | MPC |
| GANON | Metagenomics | MPC |

## Quick links

- **[Quickstart →](getting-started/quickstart.md)** — Install and run Sequre in minutes.
- **[Basic MPC Tutorial →](tutorials/basic-mpc.md)** — Understand additive secret sharing with `Sharetensor`.
- **[Secure Branching Without if →](tutorials/secure-branching-without-if.md)** — Learn mask-based selection patterns for private control flow.
- **[One Algorithm, Many Secure Types →](tutorials/one-algorithm-many-secure-types.md)** — Reuse the same algorithm across ndarray, Sharetensor, and multiparty encrypted types.
- **[Transitioning to MHE →](tutorials/transition-mhe.md)** — Move from secret sharing to homomorphic encryption with Shechi.
- **[Dropping Down the Stack →](tutorials/dropping-down-the-stack.md)** — Start high-level, then descend through switching, MHE internals, and Lattiseq.
- **[API Reference →](api/index.md)** — Complete reference for all public types and modules.
