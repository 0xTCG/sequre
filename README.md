# Sequre-MHE

Sequre-MHE is an end-to-end, statically compiled and performance engineered, Pythonic framework for building efficient secure multiparty computation (MPC), homomorphic encryption (HE), and multiparty homomorphic encryption (MHE) pipelines in bioinformatics.

## Quick start

### Install

Sequre-MHE can **only** be built from source at the moment.
To install Sequre-MHE, first clone the repository:
```bash
git clone git@github.com:0xTCG/sequre-mhe.git && cd sequre-mhe
```
And then run the install script:
```bash
source scripts/install.sh
```

### Test run

Execute
```bash
scripts/run.sh -release benchmarks --playground --local
```
to run the sample code from [playground.codon](playground.codon) that contains the benchmarks from [Hastings _et al._](https://github.com/MPC-SoK/frameworks).

This run will execute the code in a local, single machine, environment over inter-process communication channels (AF_UNIX). For running the codebase in a different environment, see [run instructions](#run-instructions).

## Running playground, tests, and benchmarks

For running [tests](#running-tests), [benchmarks](#running-benchmarks), and [playground](#running-playground), we recommend using the `scripts/run.sh` script:
```bash
scripts/run.sh -release <program> [<pid>] [--local] [--use-ring] [--unit | --all]
```
where:
- `<program>` is either `tests` or `benchmarks`.
- `<pid>` is optional ID of computing party if the run is [online](#sequres-network-config).
- `--local` flag triggers the [local](#sequres-network-config) run, intead of online, using the inter-process communication instead of TCP. **Note:** `<pid>` is ignored if the `--local` flag is present.
- `--use-ring` flag coerces usage of $2^k$ rings for MPC subroutines that are generally faster but introduce a slight inaccuracy ( $\pm 1/2^{20}$ ) to the fixed-point arithmetic. Without the flag, Sequre defaults to a finite field instead. **Note:** `--use-ring` is ignored while running tests. Tests are executed on both rings and fields.
- `--unit` flag restricts the tests to unit test only.
- `--all` flag enables both unit and end-to-end tests of applications (i.e. GWAS, DTI, Opal, and Ganon). If set while running benchmarks, all benchmarks will be executed.

Example invocation of unit tests in a `localhost` in an online network environment: (use multiple terminals for clear output)
```bash
scripts/run.sh -release tests --unit 0 & \
scripts/run.sh -release tests --unit 1 & \
scripts/run.sh -release tests --unit 2
```

**Note:** Each run bellow is executed in a local setup. Online run is also possible. See [example](#online-run) above for a step-by-step guide and/or [Sequre's network config](#sequres-network-config) for details.

### Running playground

[Playground](playground.codon) contains the three MPC benchmarks from [Hastings _et al._](https://github.com/MPC-SoK/frameworks).
Use it to quickly explore Sequre and its [features](https://github.com/0xTCG/sequre/discussions/2).

Example invocation:
```bash
scripts/run.sh -release benchmarks --local --use-ring --playground
```

### Running tests

To run all [unit tests](tests/unit_tests) execute:
```bash
scripts/run.sh -release tests --local --unit
```

This will execute all unit tests [locally](#sequres-network-config), on a single machine.

Replace the `--unit` flag with `--all` flag to include the end-to-end tests for genome-wide association study, drug-target interaction inference, and metagenomic classifiers as well:
```bash
scripts/run.sh -release tests --local --all
```

### Running benchmarks

To benchmark any applications run:
```bash
scripts/run.sh -release benchmarks --local --<app>
```
where `<app>` can be:
- `lattiseq` for running Lattiseq microbenchmarks.
- `king` for running kinship coefficients estimation on top of a subsampled lung cancer dataset from [Qing _et al._](https://www.nature.com/articles/ng.2456).
- `pca` for running Sequre's PCA subroutine on top on top of a subsampled lung cancer dataset from [Qing _et al._](https://www.nature.com/articles/ng.2456).
- `lin-alg` for running Sequre's linear algebra subroutines
- `gwas` for running genome-wide association study on top of a toy dataset from [Cho _et al._](https://github.com/hhcho/secure-gwas).
- `dti` for running drug-target inference on top of a reduced STITCH dataset from [Hie _et al._](https://github.com/brianhie/secure-dti).
- `opal` for running Opal (metagenomic binning) with 0.1x and 15x coverage of the complete Opal dataset from [Yu _et al._](https://github.com/yunwilliamyu/opal).
- `ganon`  for running Ganon (metagenomic binning) on top of a single read from the complete Opal dataset from [Yu _et al._](https://github.com/yunwilliamyu/opal).
- `genotype-imputation` for running genotype imputation on top of 1,500 samples from Idash 2019 competition dataset from [Kim _et al._](https://www.sciencedirect.com/science/article/pii/S240547122100288X).

Benchmark results are stored in the [results](results) folder.

## Sequre's network config

Sequre can operate in two network modes:
- Local: using the inter-process communication (AF_UNIX) sockets.
- Online: using the TCP (AF_INET) sockets.

If using the online mode, make sure to configure the network within Sequre's [settings file](stdlib/sequre/settings.codon) at each machine separately.

Example network configuration (`stdlib/sequre/settings.codon` --- the IP addresses are fictional):
```python
# IPs
TRUSTED_DEALER = '8.8.8.8'  # Trusted dealer
COMPUTING_PARTIES = [
    '9.9.9.9',  # First computing party (CP1)
    '10.10.10.10'  # Second computing party (CP2)
    ]
```

**Note:** Make sure to set the same network settings (IP addresses) at each computing party.

## Via docker
**Note:** Docker runs are currently enabled **only** for the [local](#sequres-network-config) network environments.
[Playground, test, and benchmark runs](#running-playground-tests-and-benchmarks) can be executed via docker:
```bash
docker run --privileged --rm hsmile/sequre-mhe:manylinux <command>
```
where `<command>` can be any of the (local network) [commands above](#running-playground-tests-and-benchmarks).
For example:
```bash
docker run --privileged --rm -m 16GB hsmile/sequre-mhe:manylinux scripts/run.sh -release tests --unit --local
```
will run all the unit tests in the local network environment.
