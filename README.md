# Sequre (with Shechi)

Sequre and Shechi are an end-to-end, statically compiled and performance engineered, Pythonic framework for building efficient secure multiparty computation (MPC), homomorphic encryption (HE), and multiparty homomorphic encryption (MHE) pipelines in bioinformatics.

## Installation

**Note:** Sequre/Shechi runs only on Linux at the moment.

Install [Codon](https://github.com/exaloop/codon) first:
```bash
mkdir $HOME/.codon && curl -L https://github.com/exaloop/codon/releases/download/v0.17.0/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon --strip-components=1
```

Then install Sequre:
```bash
curl -L https://github.com/0xTCG/sequre/releases/download/v0.0.20-alpha/sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon/lib/codon/plugins
```

Afterwards, add alias for sequre command:
```bash
alias sequre="find . -name 'sock.*' -exec rm {} \; && CODON_DEBUG=lt $HOME/.codon/bin/codon run --disable-opt="core-pythonic-list-addition-opt" -plugin sequre"
```

Finally, clone the repository:
```bash
git clone https://github.com/0xTCG/sequre.git && cd sequre
```
and run Sequre examples:
```bash
sequre examples/local_run.codon
```

## Run

Check the code in the [examples](examples/) for quick insight into Sequre.

### Online run

At each party run:
```bash
SEQURE_CP_IPS=<ip1>,<ip2>,...,<ipN> sequre examples/online_run.codon <pid>
```
where `<ipN>` denotes the IP address of each party and `<pid>` denotes the ID of the party.

For example, in a two-party setup with a trusted dealer, run (IP addresses are random):
```bash
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre examples/online_run.codon 0
```
at a trusted dealer (CP0).

```bash
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre examples/online_run.codon 1
```
at the first party (CP1).

```bash
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre examples/online_run.codon 2
```
at the second party (CP2).

### Local run

To simulate the run on a single machine over multiple processes run:

```bash
sequre examples/local_run.codon --skip-mhe-setup
```

This will simulate the run in a two-party setup with a trusted dealer.

_**Note:** `--skip-mhe-setup` flag disables the homomorphic encryption setup since `examples/local_run.codon` does not require homomorphic encryption._

### Release mode

For (much) better performance but without debugging features such as backtrace, add `-release` flag immediatelly after `sequre` command:

```bash
sequre -release examples/local_run.codon --skip-mhe-setup
```

## Benchmarks (USENIX Security 2025)

**Note:** We generate all data at random for easier testing. For the original data (from dbGaP under accession phs000716.v1.p1), please consult the authors.

Run all USENIX Security 2025 benchmarks after cloning Sequre repository and checking out the artifact branch:
```bash
git clone -b artifact https://github.com/0xTCG/sequre.git && cd sequre
```

### Local run (single machine)
```bash
sequre -release scripts/invoke.codon run-benchmarks --local --jit --stdlib-builtin --king --pca --gwas-without-norm
```

### Online run
Set each `<ipN>` to the respective IP address and `<pid>` to the respective ID. Please see the [online run example above](online_run).

```bash
SEQURE_CP_IPS=<ip1>,<ip2>,...,<ipN> sequre -release scripts/invoke.codon run-benchmarks --jit --stdlib-builtin --king --pca --gwas-without-norm <pid>
```

### Check accuracy

Check the accuracy of all solutions against the ground truth:

```bash
python scripts/accuracy.py
```
