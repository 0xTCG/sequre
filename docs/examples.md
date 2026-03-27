# Example Applications

_Located in `applications/`_

Sequre includes several end-to-end privacy-preserving applications that demonstrate the framework's capabilities across genomics, healthcare, and machine learning domains.

---

## Overview

| Application | File | ML Model | Data Type | Description |
|---|---|---|---|---|
| **Credit Score** | `credit_score.codon` | Neural network | MPU (partition) | Privacy-preserving credit scoring classifier |
| **DTI** | `dti.codon` | Neural network | MPU (partition) | Drug-target interaction prediction |
| **GANON** | `ganon.codon` | Classification | MPU | Secure metagenomic classification |
| **Genotype Imputation** | `genotype_imputation.codon` | Linear regression | MPP / MPU | Impute missing genotypes across distributed cohorts |
| **GWAS** | `gwas.codon` | PCA + linear regression | MPU (partition) | Genome-wide association study with secure PCA |
| **KING** | `king.codon` | Kinship coefficients | Sharetensor | Secure kinship estimation (KING-robust method) |
| **MI** | `mi.codon` | Multiple imputation | MPU | Secure multiple imputation with Rubin's rules |
| **MNIST** | `mnist.codon` | Multinomial logistic regression | MPU | Handwritten digit classification |
| **OPAL** | `opal.codon` | Linear SVM | MPU | Metagenomic profiling (secure OPAL pipeline) |

---

## Patterns demonstrated

### `@sequre` entry points

All applications use `@sequre`-annotated functions as their secure computation entry points:

```python
@sequre
def gwas_protocol(mpc):
    X_mpu = MPU(mpc, local_genotypes, "partition")
    ...
```

### MPU with horizontal partitioning

Most applications distribute data across parties using `MPU(mpc, data, "partition")`, where each party holds its own rows:

- **GWAS**: Each hospital/biobank holds patient genotypes
- **DTI**: Each institution holds drug-target pairs
- **Credit Score**: Each party holds customer records

### Protocol switching

Applications like **GWAS** and **Genotype Imputation** use `via_mpc` to switch between MHE and MPC for operations like eigenvalue decomposition and matrix inverse.

### Secure ML pipeline

Applications compose Sequre's ML modules:

- `LinReg` → genotype imputation, GWAS
- `LogReg` (multinomial) → MNIST
- `lsvm_train` → OPAL
- PCA (`random_pca_*`) → GWAS
- MI (`Imputer`, `MI`, `MICE`) → MI application

---

## Configuration

Applications use TOML configuration files in `applications/config/`:

| Config | Application |
|---|---|
| `credit_score.toml` | Credit score neural network parameters |
| `gwas.toml` | GWAS dataset paths and PCA settings |
| `king.toml` | KING kinship parameters |
| `mi.toml` | MI imputation settings |
| `pca.toml` | PCA standalone configuration |

See [Configuration](api/configuration.md) for the full configuration reference.

---

## Loading private data (`collective_load`)

The examples in `examples/` use synthetic data shared from a trusted dealer, which is convenient for testing but does not reflect how production deployments handle private data.  In a real deployment, each party holds its own data on disk and loads it via `collective_load`.

[`examples/collective_load.codon`](https://github.com/0xTCG/sequre/blob/develop/examples/collective_load.codon) demonstrates both protocols in a single file:

| Part | Protocol | Type | Application |
|---|---|---|---|
| Credit scoring | MHE | `MPU.collective_load` (partition) | Neural-network classifier on horizontally partitioned customer records |
| Linear regression | MPC | `Sharetensor.collective_load` | Regression model on additive secret shares of patient data |

Run it with:

```bash
sequre examples/collective_load.codon --local
```

For the full guide, see [Loading Private Data](tutorials/loading-private-data.md).
