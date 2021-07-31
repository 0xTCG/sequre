# Sequre Framework

## Meaningful ETAs

See [milestones](https://github.com/0xTCG/sequre-dsl/milestones?direction=desc&sort=due_date&state=open) for more details.

## Preliminary statistics

### Standard MPC library

- These are cummulative statistics for computing altogether:
  - QR factorization
  - Tridiagonalization
  - Eigen decomposition
  - Orthonormal basis calculation
- Input matrix size: 50x50

|        | Online bw (MB) |  LOC  | Runtime (s) |
|:------:|:--------------:|:-----:|:-----------:|
|   C++  |     ~243 MB    |  ~500 |     ~69     |
|   Seq  |     ~219 MB    |  ~300 |     ~67     |
| Sequre |     ~193 MB    |  ~80  |     ~61     |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |  1309020   |      569190     |   392230    |
| Sequre |   967584   |      569190     |   392230    |

### GWAS

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10

|        | Online bw (MB) |  LOC  | Runtime (s) |
|:------:|:--------------:|:-----:|:-----------:|
|   C++  |     ~85 MB     | ~2000 |     ~64     |
|   Seq  |     ~83 MB     | ~1000 |     ~58     |
| Sequre |     ~79 MB     |  ~250 |     ~56     |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |   167711   |      77729      |   28837     |
| Sequre |   142644   |      78721      |   28846     |

### Logistic regression

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10
- Number of iterations: 5

|        | Online bw (MB) |  LOC  | Runtime (s) |
|:------:|:--------------:|:-----:|:-----------:|
|   C++  |     ~48 MB     |  ~600 |    ~114     |
|   Seq  |     ~47 MB     |  ~350 |     ~65     |
| Sequre |     ~47 MB     |  ~100 |     ~57     |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |   1203713  |      601592     |    1002     |
| Sequre |   1202841  |      601583     |    1028     |

### Vanilla neural net for DTI inference

- Number of features: 6903
- Number of target classes: 2
- Number of hidden layers: 1
- Dropout: 0
- Hidden layer size: 100
- Epochs: 10

|        | Online bw (MB) |  LOC  | Runtime (s) |
|:------:|:--------------:|:-----:|:-----------:|
| PySyft |    ~685 MB     |  ~190 |     ~95     |
|   C++  |    ~353 MB     |  ~430 |     ~95     |
|   Seq  |    ~286 MB     |  ~260 |     ~90     |
| Sequre |    ~246 MB     |  ~150 |     ~90     |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |   208180   |      102090     |     150     |
| Sequre |   208120   |      102090     |     246     |
