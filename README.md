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
|   C++  |      n/a     |     n/a    |      n/a     |     n/a     |  ~500 |     n/a     |
|   Seq  |    ~219 MB   |   1309020  |     569190   |    392230   |  ~300 |     ~94     |
| Sequre |    ~193 MB   |   967584   |     569190   |    392230   |  ~80  |     ~88     |

### GWAS

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10

|        | Net workload | Partitions | Reconstructs | Truncations |  LOC  | Runtime (s) |
|:------:|:------------:|:----------:|:------------:|:-----------:|:-----:|:-----------:|
|   C++  |      n/a     |     n/a    |      n/a     |     n/a     | ~2000 |     n/a     |
|   Seq  |    ~83 MB    |   149662   |     68702    |    28813    | ~1000 |     ~70     |
| Sequre |    ~79 MB    |   124599   |     69689    |    28822    |  ~250 |     ~80     |
