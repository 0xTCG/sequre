# Sequre (with Shechi)

Sequre and Shechi are an end-to-end, statically compiled and performance engineered, Pythonic framework for building efficient secure multiparty computation (MPC), homomorphic encryption (HE), and multiparty homomorphic encryption (MHE) pipelines.

## Installation

**Supported platforms:** Linux (x86_64, aarch64) and macOS (x86_64, Apple Silicon).

Install [Codon](https://github.com/exaloop/codon) first:
```bash
mkdir -p $HOME/.codon && curl -L https://github.com/exaloop/codon/releases/download/v0.17.0/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon --strip-components=1
```

Then install Sequre:
```bash
curl -L https://github.com/0xTCG/sequre/releases/download/v0.0.20-alpha/sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon
```

This installs the `sequre` binary to `$HOME/.codon/bin/sequre` alongside the Sequre and Seq plugins.
Add `$HOME/.codon/bin` to your `PATH` if you haven't already (the Codon tarball does not do this automatically):
```bash
export PATH=$HOME/.codon/bin:$PATH
```
To make this permanent, add the line above to your `~/.bashrc`, `~/.zshrc`, or equivalent shell profile.

The launcher preserves inline env vars like `SEQURE_CP_IPS=...` for the actual `codon run` process.
It also auto-detects common OpenSSL library paths when `SEQURE_LIBCRYPTO_PATH` / `SEQURE_OPENSSL_PATH` are not explicitly set.

### Run vs build

Sequre supports the two Codon execution modes:
- `run` (default): JIT-compile and execute immediately (`sequre run <file> [args]`).
- `build`: Produce a standalone binary you can execute later (`sequre build -o <out> [build-opts]`).

If mode is omitted, `run` is assumed.

## Run

Clone the repository:
```bash
git clone https://github.com/0xTCG/sequre.git && cd sequre
```
and check the code in the [examples](examples/) for quick insight into Sequre.

### Local run

```bash
sequre examples/local_run.codon
```

Or build a binary instead:
```bash
sequre build examples/local_run.codon -o local_run
./local_run
```

This will simulate the run in a two-party setup with a trusted dealer.

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

### TLS certificates

While Sequre handles MHE/MPC key-management under the hood, it relies on user-provided certificates for setting up secure channels between the parties.

The following files are expected under `certs/` (or path configured by `SEQURE_CERT_DIR` env var) on each host:
- `ca.pem` (adjustable via `SEQURE_CA_CERT_FILE` env var)
- `cp<pid>.pem`  (adjustable via `SEQURE_PARTY_CERT_FILE` env var)
- `cp<pid>-key.pem` (adjustable via `SEQURE_PARTY_KEY_FILE` env var)

For development/testing, you can generate local certificates with:
```bash
./scripts/generate_certs.sh <num_parties> ./certs
```

Production guidance:
- Manage certificates with your own PKI workflow (for example, enterprise PKI, Vault PKI, or cloud/private CA).
- Do not use `scripts/generate_certs.sh` as a production CA/enrollment workflow.
- Rotate party certificates regularly and keep CA private keys outside Sequre application hosts.

### Release mode

For (much) better performance but without debugging features such as backtrace, add `-release` flag immediatelly after `sequre` command:

```bash
sequre -release examples/local_run.codon --skip-mhe-setup
```

_**Note:** `--skip-mhe-setup` flag disables the homomorphic encryption setup since `examples/local_run.codon` runs only Sequre (SMC)._
