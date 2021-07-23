# Sequre Framework

## Meaningful ETAs
- Coding preliminary done: 22.8.2021
- Paper written: 26.9.2021
- First release (v0.0.1): 10.10.2021

See [milestones](https://github.com/0xTCG/sequre-dsl/milestones?direction=desc&sort=due_date&state=open) for more details.

## Preliminary statistics

### Standard MPC library

- These are cummulative statistics for computing altogether:
  - QR factorization
  - Tridiagonalization
  - Eigen decomposition
  - Orthonormal basis calculation
- Input matrix size: 50x50

|        | Net workload | Partitions | Reconstructs | Truncations |  LOC  | Runtime (s) |
|:------:|:------------:|:----------:|:------------:|:-----------:|:-----:|:-----------:|
|   C++  |    ~243 MB   |   1094924  |    462142    |    392230   |  ~500 |     ~69     |
|   Seq  |    ~219 MB   |   1309020  |    569190    |    392230   |  ~300 |     ~67     |
| Sequre |    ~193 MB   |   967584   |    569190    |    392230   |  ~80  |     ~61     |

### GWAS

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10

|        | Net workload | Partitions | Reconstructs | Truncations |  LOC  | Runtime (s) |
|:------:|:------------:|:----------:|:------------:|:-----------:|:-----:|:-----------:|
|   C++  |    ~85 MB    |   80717    |     34232    |    28837    | ~2000 |     ~64     |
|   Seq  |    ~83 MB    |   149662   |     68702    |    28813    | ~1000 |     ~58     |
| Sequre |    ~79 MB    |   124599   |     69689    |    28822    |  ~250 |     ~56     |

### Logistic regression

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10
- Number of iterations: 5

|        | Net workload | Partitions | Reconstructs | Truncations |  LOC  | Runtime (s) |
|:------:|:------------:|:----------:|:------------:|:-----------:|:-----:|:-----------:|
|   C++  |    ~48 MB    |    3095    |     1283     |     1007    |  ~600 |    ~114     |
|   Seq  |    ~47 MB    |   1203713  |    601592    |     1002    |  ~350 |     ~65     |
| Sequre |    ~47 MB    |   1202841  |    601583    |     1028    |  ~100 |     ~65     |

### Vanilla neural net for DTI inference

- Number of features: 6903
- Number of target classes: 2
- Number of hidden layers: 1
- Dropout: 0
- Hidden layer size: 100
- Epochs: 10

|        | Net workload | Partitions | Reconstructs | Truncations |  LOC  | Runtime (s) |
|:------:|:------------:|:----------:|:------------:|:-----------:|:-----:|:-----------:|
| PySyft |      n/a     |     n/a    |      n/a     |      n/a    |  n/a  |     n/a     |
|   C++  |    ~353 MB   |    10180   |     3090     |      170    |  ~430 |     ~45     |
|   Seq  |    ~291 MB   |   208180   |    102090    |      150    |  ~260 |     ~90     |
| Sequre |      n/a     |     n/a    |      n/a     |      n/a    |  n/a  |     n/a     |
