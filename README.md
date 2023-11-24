# Sequre-MHE

Sequre-MHE is an end-to-end, statically compiled and performance engineered, Pythonic framework for building efficient secure multiparty computation (MPC), homomorphic encryption (HE), and multiparty homomorphic encryption (MHE) pipelines in bioinformatics.

## Quick start

### Installation

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
alias sequre="find  . -name 'sock.*' -exec rm {} \; && $HOME/.codon/bin/codon run -plugin sequre -plugin seq"
```

Finally, you can run Sequre as:
```bash
sequre playground.codon
```

### Test run

Run
```bash
sequre -debug playground.codon --skip-mhe-setup
```
to run the sample code from [playground.codon](playground.codon) that contains the benchmarks from [Hastings _et al._](https://github.com/MPC-SoK/frameworks).

`playground.codon` is executed in a local, single machine, environment over inter-process communication channels (AF_UNIX). For running the codebase in a different environment, see [run instructions](#run-instructions).

_**Note:** `--skip-mhe-setup` flag disables the homomorphic encryption setup since `playground.codon` does not require homomorphic encryption._

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
