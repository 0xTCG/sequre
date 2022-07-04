# Sequre

## Quick start

## Running the example

The `example` folder contains the running example of a typical multiparty computation use-case in Sequre. It implements a secure variant of [Plassclass](https://github.com/Shamir-Lab/PlasClass)---a binary classification tool for distinguishing whether a sequence
originates from a plasmid sequence or a chromosomal segment.

Folder structure:
- `client.seq` - Local (offline) source code executed by each client (data owner) locally. It contains a data processing step, followed by a secret sharing routine that initiates secure computing on the servers.
- `server.seq` - Online source code executed by each untrusted computing party. It contains data pooling routine that gathers the secret-shared data from the clients and executes secure training of a linear support vector machine on top of it.

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

To run the same procedure on multiple machines, [install](#quick-start) Sequre and reconfigure the network within Sequre's [settings file](dsl/settings.seq) at each machine separatelly.

Example network configuration at `dsl/settings.seq` (the addresses are fictional):
```python
# IPs
TRUSTED_DEALER = '8.8.8.8'  # localhost
COMPUTING_PARTIES = [
    '9.9.9.9',  # First computing party
    '10.10.10.10'  # Second computing party
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

**Note:** Make sure to have the same network settings (IP addresses) set at each computing party, including the client.


## Running tests

## Running benchmarks

## Licence
