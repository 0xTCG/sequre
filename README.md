# Sequre

Sequre is an end-to-end, statically compiled and performance engineered, Pythonic framework for building efficient secure multiparty computation (MPC), homomorphic encryption (HE), and multiparty homomorphic encryption (MHE) pipelines in bioinformatics.

## Disclaimer

Sequre is an open-source research project still intended for academic use only. For commercial use or any other use that requires attested security, please contact us at hsmajlovic@uvic.ca. 

## Installation

**Note:** Sequre runs only on Linux at the moment.

Install [Codon](https://github.com/exaloop/codon) first:
```bash
mkdir $HOME/.codon && curl -L https://github.com/0xTCG/sequre-mhe/releases/download/v0.0.2-alpha/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon --strip-components=1
```

Then install Sequre:
```bash
curl -L https://github.com/0xTCG/sequre-mhe/releases/download/v0.0.4-alpha/sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon/lib/codon/plugins
```

Afterwards, add alias for sequre command:
```bash
alias sequre="find . -name 'sock.*' -exec rm {} \; && $HOME/.codon/bin/codon run -plugin sequre -plugin seq"
```

Finally, you can run Sequre as:
```bash
sequre examples/local_run.codon
```

## Examples

Check the code in the [examples](examples/) for quick insight into Sequre.

### Online run

At each party run:
```bash
sequre examples/online_run.codon <pid>
```
where `<pid>` denotes the ID of an underlying party.

For example, in a two-party setup with a trusted dealer, run:
```bash
sequre examples/online_run.codon 0
```
at a trusted dealer (CP0).

```bash
sequre examples/online_run.codon 1
```
at the first party (CP1).

```bash
sequre examples/online_run.codon 2
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
