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

|        | Net workload |  LOC  | Runtime (s) |
|:------:|:------------:|:-----:|:-----------:|
|   C++  |    ~243 MB   |  ~500 |     ~69     |
|   Seq  |    ~219 MB   |  ~300 |     ~67     |
| Sequre |    ~193 MB   |  ~80  |     ~61     |

### GWAS

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10

|        | Net workload |  LOC  | Runtime (s) |
|:------:|:------------:|:-----:|:-----------:|
|   C++  |    ~85 MB    | ~2000 |     ~64     |
|   Seq  |    ~83 MB    | ~1000 |     ~58     |
| Sequre |    ~79 MB    |  ~250 |     ~56     |

### Logistic regression

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10
- Number of iterations: 5

|        | Net workload |  LOC  | Runtime (s) |
|:------:|:------------:|:-----:|:-----------:|
|   C++  |    ~48 MB    |  ~600 |    ~114     |
|   Seq  |    ~47 MB    |  ~350 |     ~65     |
| Sequre |    ~47 MB    |  ~100 |     ~65     |

### Vanilla neural net for DTI inference

- Number of features: 6903
- Number of target classes: 2
- Number of hidden layers: 1
- Dropout: 0
- Hidden layer size: 100
- Epochs: 10

|        | Net workload |  LOC  | Runtime (s) |
|:------:|:------------:|:-----:|:-----------:|
| PySyft |      n/a     |  n/a  |     n/a     |
|   C++  |    ~353 MB   |  ~430 |     ~45     |
|   Seq  |    ~291 MB   |  ~260 |     ~90     |
| Sequre |    ~246 MB   |  ~150 |     ~90     |
