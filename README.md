# Sequre Framework

## Meaningful ETAs

See [milestones](https://github.com/0xTCG/sequre-framework/milestones?direction=asc&sort=due_date&state=open) for more details.

## Preliminary statistics

### Standard MPC library

- These are cummulative statistics for computing altogether:
  - QR factorization
  - Tridiagonalization
  - Eigen decomposition
  - Orthonormal basis calculation
- Input matrix size: 50x50

|                | Offline bw (MB) | Online bw (MB) |  LOC  | Runtime (s) |
|:--------------:|:---------------:|:--------------:|:-----:|:-----------:|
|   C++ (field)  |       n/a       |      ~243      |  ~500 |     ~69     |
|   Seq (field)  |      ~286       |      ~219      |  ~300 |     ~67     |
| Sequre (field) |      ~286       |      ~193      |  ~80  |     ~61     |
|   Seq (ring)   |       n/a       |       n/a      |  n/a  |     n/a     |
| Sequre (ring)  |       n/a       |       n/a      |  n/a  |     n/a     |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |  1309020   |      569190     |   392230    |
| Sequre |   967584   |      569190     |   392230    |

### GWAS

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10

|                | Offline bw (MB) | Online bw (MB) |  LOC  | Runtime (s) |
|:--------------:|:---------------:|:--------------:|:-----:|:-----------:|
|   C++ (field)  |       n/a       |      ~85       | ~2000 |     ~64     |
|   Seq (field)  |       ~115      |      ~83       | ~1000 |     ~58     |
| Sequre (field) |       ~131      |      ~79       |  ~250 |     ~56     |
|   Seq (ring)   |       n/a       |       n/a      |  n/a  |     n/a     |
| Sequre (ring)  |       n/a       |       n/a      |  n/a  |     n/a     |

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

|                | Offline bw (MB) | Online bw (MB) |  LOC  | Runtime (s) |
|:--------------:|:---------------:|:--------------:|:-----:|:-----------:|
|   C++ (field)  |       n/a       |      ~48       |  ~600 |    ~114     |
|   Seq (field)  |      ~133       |      ~47       |  ~350 |     ~57     |
| Sequre (field) |      ~135       |      ~47       |  ~100 |     ~57     |
|   Seq (ring)   |       n/a       |       n/a      |  n/a  |     n/a     |
| Sequre (ring)  |       n/a       |       n/a      |  n/a  |     n/a     |

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

|                | Offline bw (MB) | Online bw (MB) |  LOC  | Runtime (s) |
|:--------------:|:---------------:|:--------------:|:-----:|:-----------:|
| PySyft (field) |        0        |      ~685      |  ~190 |     ~95     |
|   C++ (field)  |       n/a       |      ~353      |  ~430 |     ~95     |
|   Seq (field)  |      ~406       |      ~286      |  ~260 |     ~90     |
| Sequre (field) |       n/a       |      ~258      |  ~150 |     ~90     |
|   Seq (ring)   |       n/a       |       n/a      |  n/a  |     n/a     |
| Sequre (ring)  |       n/a       |       n/a      |  n/a  |     n/a     |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |   208180   |      102090     |     150     |
| Sequre |   208120   |      102090     |     150     |
