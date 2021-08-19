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

#### Network and code complexity stats

|        | Offline bw (MB) | Online bw (MB) |  LOC  |
|:------:|:---------------:|:--------------:|:-----:|
|   C++  |       n/a       |      ~243      |  ~500 |
|   Seq  |      ~286       |      ~219      |  ~300 |
| Sequre |      ~286       |      ~193      |  ~80  |

#### Performance stats

|                | Runtime (s) |
|:--------------:|:-----------:|
|   C++ (field)  |     ~69     |
|   Seq (field)  |     ~67     |
| Sequre (field) |     ~61     |
|   Seq (ring)   |     ~39     |
| Sequre (ring)  |     ~35     |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |  1309020   |      569190     |   392230    |
| Sequre |   967584   |      569190     |   392230    |

### GWAS

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10

#### Network and code complexity stats

|        | Offline bw (MB) | Online bw (MB) |  LOC  |
|:------:|:---------------:|:--------------:|:-----:|
|   C++  |       n/a       |      ~85       | ~2000 |
|   Seq  |       ~115      |      ~83       | ~1000 |
| Sequre |       ~131      |      ~79       |  ~250 |

#### Performance stats

|                | Runtime (s) |
|:--------------:|:-----------:|
|   C++ (field)  |     ~64     |
|   Seq (field)  |     ~58     |
| Sequre (field) |     ~56     |
|   Seq (ring)   |     ~14     |
| Sequre (ring)  |     ~12     |

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

#### Network and code complexity stats

|        | Offline bw (MB) | Online bw (MB) |  LOC  |
|:------:|:---------------:|:--------------:|:-----:|
|   C++  |       n/a       |      ~48       |  ~600 |
|   Seq  |      ~133       |      ~47       |  ~350 |
| Sequre |      ~135       |      ~47       |  ~100 |

#### Performance stats

|                | Runtime (s) |
|:--------------:|:-----------:|
|   C++ (field)  |    ~114     |
|   Seq (field)  |     ~57     |
| Sequre (field) |     ~57     |
|   Seq (ring)   |     ~54     |
| Sequre (ring)  |     ~53     |

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

#### Network and code complexity stats

|        | Offline bw (MB) | Online bw (MB) |  LOC  |
|:------:|:---------------:|:--------------:|:-----:|
| PySyft |        0        |      ~685      |  ~190 |
|   C++  |       n/a       |      ~353      |  ~430 |
|   Seq  |      ~406       |      ~286      |  ~260 |
| Sequre |      ~406       |      ~258      |  ~150 |

#### Performance stats

|                | Runtime (s) |
|:--------------:|:-----------:|
| PySyft (field) |     ~95     |
|   C++ (field)  |     ~95     |
|   Seq (field)  |     ~90     |
| Sequre (field) |     ~89     |
|   Seq (ring)   |     ~21     |
| Sequre (ring)  |     ~20     |

#### Internal stats

|        | Partitions | Reconstructions | Truncations |
|:------:|:----------:|:---------------:|:-----------:|
|   Seq  |   208180   |      102090     |     150     |
| Sequre |   208120   |      102090     |     150     |
