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

#### Performance stats

|        | Offline bw (MB) | Online bw (MB) |  LOC  | Field runtime (s) | Ring runtime (s) |
|:------:|:---------------:|:--------------:|:-----:|:-----------------:|:----------------:|
|   C++  |       n/a       |      ~243      |  ~500 |        ~69        |        n/a       |
|   Seq  |      ~286       |      ~219      |  ~300 |        ~54        |        ~38       |
| Sequre |      ~286       |      ~193      |  ~80  |        ~49        |        ~34       |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |  1309020   |      569190     |   392230    |
| Sequre |   967584   |      569190     |   392230    |

### GWAS

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10

#### Performance stats

|        | Offline bw (MB) | Online bw (MB) |  LOC  | Field runtime (s) | Ring runtime (s) |
|:------:|:---------------:|:--------------:|:-----:|:-----------------:|:----------------:|
|   C++  |       n/a       |      ~85       | ~2000 |        ~64        |        n/a       |
|   Seq  |       ~115      |      ~83       | ~1000 |        ~39        |        ~12       |
| Sequre |       ~131      |      ~79       |  ~250 |        ~37        |        ~11       |

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

#### Performance stats

|        | Offline bw (MB) | Online bw (MB) |  LOC  | Field runtime (s) | Ring runtime (s) |
|:------:|:---------------:|:--------------:|:-----:|:-----------------:|:----------------:|
|   C++  |       n/a       |      ~48       |  ~600 |        ~114       |        n/a       |
|   Seq  |      ~133       |      ~47       |  ~350 |        ~46        |        ~46       |
| Sequre |      ~135       |      ~47       |  ~100 |        ~47        |        ~47       |

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

#### Performance stats

|        | Offline bw (MB) | Online bw (MB) |  LOC  | Field runtime (s) | Ring runtime (s) |
|:------:|:---------------:|:--------------:|:-----:|:-----------------:|:----------------:|
| PySyft |        0        |      ~685      |  ~190 |         n/a       |         ~95      |
|   C++  |       n/a       |      ~353      |  ~430 |         ~95       |         n/a      |
|   Seq  |      ~406       |      ~286      |  ~260 |         ~62       |         ~20      |
| Sequre |      ~406       |      ~258      |  ~150 |         ~60       |         ~18      |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |   208180   |      102090     |     150     |
| Sequre |   208120   |      102090     |     150     |
