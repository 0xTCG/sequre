# Sequre-MHE

Sequre-MHE is an end-to-end, statically compiled and performance engineered, Pythonic framework for building efficient secure multiparty computation (MPC), homomorphic encryption (HE), and multiparty homomorphic encryption (MHE) pipelines in bioinformatics.

## Installation

**Note:** Sequre runs only on Linux at the moment.

Install [Codon](https://github.com/exaloop/codon) first:
```bash
mkdir $HOME/.codon && curl -L https://github.com/0xTCG/sequre-mhe/releases/download/v0.0.1-beta/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon --strip-components=1
```

Then install Sequre:
```bash
curl -L https://github.com/0xTCG/sequre-mhe/releases/download/v0.0.1-beta/sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon/lib/codon/plugins
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

At trusted dealer (CP0):
```bash
sequre examples/online_run.codon 0
```

At first computing party (CP1):
```bash
sequre examples/online_run.codon 1
```

At second computing party (CP2):
```bash
sequre examples/online_run.codon 2
```

### Local run

```bash
sequre examples/local_run.codon --skip-mhe-setup
```

_**Note:** `--skip-mhe-setup` flag disables the homomorphic encryption setup since `playground.codon` does not require homomorphic encryption._

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
