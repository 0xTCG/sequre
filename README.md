# Sequre Framework

## Meaningful ETAs
- Coding preliminary done: 22.8.2021
- Paper written: 26.9.2021
- First release (v0.0.1): 10.10.2021

See [milestones](https://github.com/0xTCG/sequre-dsl/milestones?direction=desc&sort=due_date&state=open) for more details.

## Preliminary statistics

### Stdandard MPC library

- These are cummulative statistics for computing altogether:
  - QR factorization
  - Tridiagonalization
  - Eigen decomposition
  - Orthonormal basis calculation
- Input matrix size: 50x50

|        | Net workload | Partitions | Reconstructs | Truncations |  LOC  |
|:------:|:------------:|:----------:|:------------:|:-----------:|:-----:|
|   C++  |      n/a     |     n/a    |      n/a     |     n/a     |  ~500 |
|   Seq  |    ~556 MB   |   1309020  |     569190   |    392230   |  ~300 |
| Sequre |    ~509 MB   |   1308864  |     569190   |    392230   |  ~80  |

### GWAS

- Number of individuals: 1000
- SNPs count: 1000
- Covs count: 10

|        | Net workload | Partitions | Reconstructs | Truncations |  LOC  |
|:------:|:------------:|:----------:|:------------:|:-----------:|:-----:|
|   C++  |      n/a     |     n/a    |      n/a     |     n/a     | ~2000 |
|   Seq  |    ~200 MB   |   149662   |     68702    |    28813    | ~1000 |
| Sequre |    ~192 MB   |   149391   |     69689    |    28822    |  ~250 |
