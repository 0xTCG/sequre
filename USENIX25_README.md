## Benchmarks (USENIX Security 2025)

**Note:** We generate all data at random for easier testing. For the original data (from dbGaP under accession phs000716.v1.p1), please consult the authors.

After [installing Sequre](README.md) clone Sequre repository and then run all benchmarks as follows:
```bash
git clone https://github.com/0xTCG/sequre.git && cd sequre
```

### Local run (single machine)
```bash
sequre -release scripts/invoke.codon run-benchmarks --local --jit --lattiseq --mpc --mhe --stdlib-builtin --king --pca --gwas-without-norm --ablation
```

This will evaluate microbenchmarks **(Table 2; Section 10.2)**, basic and complex workflows **(Figure 7; Section 10.2)**, and MNIST **(Section 10.5)** for both Sequre and Shechi.

### Online run
Local run should suffice all functionality tests. However, online run is also possible. Set each `<ipN>` to the respective IP address and `<pid>` to the respective ID. Please see the online run in original [README](README.md) for more details.

```bash
SEQURE_CP_IPS=<ip1>,<ip2>,...,<ipN> sequre -release scripts/invoke.codon run-benchmarks --jit --stdlib-builtin --king --pca --gwas-without-norm <pid>
```

### Check accuracy

Check the accuracy of all solutions against the ground truth:

```bash
python scripts/accuracy.py
```

### Other workflows

Run EVA basic workflows **(Figure 7; Section 10.2)** via docker:
```bash
docker run --rm --privileged hsmile/eva:bench
```

Run HEFactory basic workflows **(Figure 7; Section 10.2)** via docker:
```bash
docker run --rm --privileged hsmile/hefactory:latest
```

Run Lattigo basic workflows **(Figure 7; Section 10.2)** via docker:
```bash
docker run -it --rm --privileged hsmile/lattigo:bench
```

Run SEAL micro-benchmarks **(Table 2; Section 10.2)** via docker:
```bash
docker run --rm --privileged hsmile/seal:bench
```

Run MP-SPDZ micro-benchmarks and basic workflows **(Table 2 and Figure 7; Section 10.2)** via docker:
```bash
docker run --rm --privileged hsmile/mpspdz:bench
```

Run Lattigo micro-benchmarks **(Table 2; Section 10.2)** via docker:
```bash
docker run -it --rm --privileged hsmile/lattigo:micro
```

Run Lattigo Kinship **(Figure 7; Section 10.2)** via docker:
```bash
docker run -it --rm --privileged hsmile/lattigo:king
```

Run Lattigo GWAS (with PCA) **(Figure 7; Section 10.2)** via docker:
```bash
docker run -it --rm --privileged hsmile/lattigo:gwas
```
