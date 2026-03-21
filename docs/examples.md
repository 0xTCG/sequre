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

Applications like **GWAS** and **Genotype Imputation** use `via_mpc` to switch between MHE and SMC for operations like eigenvalue decomposition and matrix inverse.

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
