# Sequre

Sequre is an end-to-end, statically compiled and performance engineered, Pythonic framework for building efficient secure multiparty computation (MPC) pipelines in bioinformatics.

## Quick start

### Install

Sequre can **only** be built from source at the moment.
To install Sequre, first clone the repository:
```bash
git clone --depth 1 --branch v0.0.1-alpha git@github.com:0xTCG/sequre.git && cd sequre
```
And then run the install script:
```bash
scripts/install.sh
```
This will:
- Clone and install a version of [Seq](https://github.com/seq-lang/seq) that contains Sequre-related intermediate representation (IR) transformations.
- Build the source for both Seq and Sequre.

### Test run

Execute
```bash
scripts/run.sh playground --local
```
to run the sample code from [playground.seq](playground.seq) that contains the benchmarks from [Hastings _et al._](https://github.com/MPC-SoK/frameworks).

This run will execute the code in a local, single machine, environment over inter-process communication channels (AF_UNIX). For running the codebase in a different environment, see [run instructions](#run-instructions).

## Run instructions

Use `./sequre` script to execute Sequre both on server and client end.

Server run command: _(`<pid>` denotes the ID of the computing party: 0, 1, 2, 3, ...)_
```bash
./sequre foo.seq <pid>
```

Client run command:
```bash
./sequre bar.seq
```

See the [example](#running-the-example) for a sample run at the `localhost`.

## Running the example

The [example](example) folder contains the running example of a typical multiparty computation use-case in Sequre. It implements a secure variant of [PlassClass](https://github.com/Shamir-Lab/PlasClass)---a binary classification tool for distinguishing whether a genomic sequence
originates from a plasmid sequence or a chromosomal segment.

Folder contains:
- `client.seq` - Local source code executed by each client (data owner) locally. It contains a data processing step, followed by a secret sharing routine that initiates secure computing on the servers.
- `server.seq` - Online source code executed by each untrusted computing party. It contains a data pooling routine that gathers the secret-shared data from the clients and conducts secure training of a linear support vector machine on top of it.

### Localhost run

To run the example locally, execute `example/server.seq` in a separate terminal for each computing party `<pid>`:
```bash
./sequre example/server.seq <pid>
```

Finally, initiate the secret sharing of the data and, consequentially, secure training on top of it by running the client's code:

```bash
./sequre example/client.seq
```

Example (condensed into a single terminal for simplicity):
```bash
./sequre example/server.seq 0 & \
./sequre example/server.seq 1 & \
./sequre example/server.seq 2 & \
./sequre example/client.seq
```
**Note:** Expect obfuscated output (and possibly some minor warning messages) if running in a single terminal. Each party will output the results into the same terminal.

### Online run

To run the same procedure on multiple machines, [install Sequre](#quick-start) and reconfigure the network within Sequre's [settings file](dsl/settings.seq) at each machine separately.

Example network configuration (`dsl/settings.seq` --- the IP addresses are fictional):
```python
# IPs
TRUSTED_DEALER = '8.8.8.8'  # Trusted dealer
COMPUTING_PARTIES = [
    '9.9.9.9',  # First computing party (CP1)
    '10.10.10.10'  # Second computing party (CP2)
    ]
```

Then at `8.8.8.8` run
```bash
./sequre example/server.seq 0
```

At `9.9.9.9` run:
```bash
./sequre example/server.seq 1
```

At `10.10.10.10` run:
```bash
./sequre example/server.seq 2
```

And finally, at your client's machine, run:
```bash
./sequre example/client.seq
```

**Note:** Make sure to set the same network settings (IP addresses) at each computing party, including the client.


## Sequre's network config

Sequre can operate in two network modes:
- Local: using the inter-process communication (AF_UNIX) sockets.
- Online: using the TCP (AF_INET) sockets.

If using the online mode, make sure to configure the network within Sequre's [settings file](dsl/settings.seq) at each machine separately.

Example network configuration (`dsl/settings.seq` --- the IP addresses are fictional):
```python
# IPs
TRUSTED_DEALER = '8.8.8.8'  # Trusted dealer
COMPUTING_PARTIES = [
    '9.9.9.9',  # First computing party (CP1)
    '10.10.10.10'  # Second computing party (CP2)
    ]
```

**Note:** `./sequre` command operates only in an online setup at the moment.


## Running playground, tests, and benchmarks

For running [tests](#running-tests), [benchmarks](#running-benchmarks), and [playground](#running-playground), we recommend using the `scripts/run.sh` script:
```bash
srcipts/run.sh <program> [<pid>] [--local] [--use-ring] [--unit]
```
where:
- `<program>` is either `tests`, `benchmarks`, or `playground`.
- `<pid>` is optional ID of computing party if the run is [online](#sequres-network-config).
- `--local` flag triggers the [local](#sequres-network-config) run, intead of online, using the inter-process communication instead of TCP. **Note:** `<pid>` is ignored if the `--local` flag is present.
- `--use-ring` flag coerces usage of $2^k$ rings for MPC subroutines that are generally faster but introduce a slight inaccuracy ($\pm 1/2^{20}$) in fixed point numbers. Without the flag, Sequre defaults to a finite field instead. **Note:** `--use-ring` is ignored while running tests. Tests are executed on both rings and fields.
- `--unit` flag restricts the tests to unit test only. By default, both unit and end-to-end tests of applications (GWAS, DTI, Opal, and Ganon) are executed.

Example invocation of unit tests in a `localhost` in an online network environment: (use multiple terminals for clear output)
```bash
srcipts/run.sh tests --unit 0 & \
srcipts/run.sh tests --unit 1 & \
srcipts/run.sh tests --unit 2
```

**Note:** Each run bellow is executed in a local setup. Online run is also possible. See [example](#online-run) above for a step-by-step guide and/or [Sequre's network config](#sequres-network-config) for details.

### Running playground

[Playground](playground.seq) contains the three MPC benchmarks from [Hastings _et al._](https://github.com/MPC-SoK/frameworks).
Use it to quickly explore Sequre and its features.

Example invocation:
```bash
scripts/run.sh playground --local --use-ring
```

### Running tests

To run all [unit tests](tests/unit_tests) execute:
```bash
scripts/run.sh tests --unit --local
```

This will execute all unit tests [locally](#sequres-network-config), on a single machine.

Drop the `--unit` flag to include the end-to-end tests for genome-wide association study, drug-target interaction inference, and metagenomic classifiers as well:
```bash
scripts/run.sh tests --local
```

### Running benchmarks

To benchmark all applications run:
```bash
scripts/run.sh benchmarks --local
```

Include `--use-ring` flag to re-run all benchmarks on rings instead of fields:
```bash
scripts/run.sh benchmarks --local --use-ring
```

This will benchmark the following applications in Sequre:
- Sequre's linear algebra subroutines
- Genome-wide association study on top of a toy dataset from [Cho _et al._](https://github.com/hhcho/secure-gwas).
- Drug-target inference on top of a reduced STITCH dataset from [Hie _et al._](https://github.com/brianhie/secure-dti).
- Opal (metagenomic binning) with 0.1x and 15x coverage of the complete Opal dataset from [Yu _et al._](https://github.com/yunwilliamyu/opal)
- Ganon (metagenomic binning) on top of a single read from the complete Opal dataset from [Yu _et al._](https://github.com/yunwilliamyu/opal)

Benchmark results are stored in the [results](results) folder.

## License

Sequre is published under the [academic public license](LICENSE.md).
