# Sequre

## Quick start

### Install

Sequre can **only** be built from source at the moment.
To install Sequre, first clone the repository:
```bash
git clone git@github.com:0xTCG/sequre.git && cd sequre
```
And then run the install script:
```bash
scripts/install.sh
```
This will:
- Clone and install a version of [Seq](https://github.com/seq-lang/seq) that contains Sequre-related intermediate representation (IR) transformations.
- Build the source of both Seq and Sequre.

### Test run

Execute
```bash
scripts/run.sh release playground --local
```
to run the sample code from [playground.seq](playground.seq) that contains the benchmarks from [Hastings _et al._](https://github.com/MPC-SoK/frameworks).

This run will execute the code in a local, single machine, environment over inter-process communication channels (AF_UNIX). For running the codebase in a different environment, see [run instructions](#run-instructions).

## Run instructions

## Running the example

The [example](example) folder contains the running example of a typical multiparty computation use-case in Sequre. It implements a secure variant of [PlassClass](https://github.com/Shamir-Lab/PlasClass)---a binary classification tool for distinguishing whether a genomic sequence
originates from a plasmid sequence or a chromosomal segment.

Folder structure:
- `client.seq` - Local source code executed by each client (data owner) locally. It contains a data processing step, followed by a secret sharing routine that initiates secure computing on the servers.
- `server.seq` - Online source code executed by each untrusted computing party. It contains a data pooling routine that gathers the secret-shared data from the clients and conducts secure training of a linear support vector machine on top of it.

### Localhost run

To run the example locally, execute `server.seq` in a separate terminal for each computing party `<pid>`:
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
TRUSTED_DEALER = '8.8.8.8'  # CP0
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


## Running tests

## Running benchmarks

## License

Sequre is published under the [academic public license](LICENSE.md).
