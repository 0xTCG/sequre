# Build from Source

This page describes how to compile Sequre and all its dependencies from source. Most users should use the [pre-built binaries](quickstart.md) instead.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| `clang` / `clang++` | C++17 support | GCC may work but is untested |
| `cmake` | ≥ 3.20 | |
| `ninja` | any | Build generator |
| `git` | any | |

## Overview

Sequre is a Codon compiler plugin. Building it from source requires building the full toolchain:

```
1. LLVM  (Codon's fork)
     ↓
2. Codon  (the compiler)
     ↓
3. Seq   (Codon plugin — bioinformatics types)
     ↓
4. Sequre  (Codon plugin — secure computation)
```

The automated script `scripts/install.sh` handles all four steps. You can also run each step manually.

---

## Automated build

From the Sequre repository root:

```bash
git clone https://github.com/0xTCG/sequre.git && cd sequre
source scripts/install.sh
```

This clones and builds LLVM, Codon, Seq, and Sequre into sibling directories, then installs the Sequre plugin into the local Codon installation. The entire build takes a significant amount of time (primarily due to LLVM).

If you already have some components built, set these environment variables **before** running the script to skip those steps:

| Variable | Default | Description |
|---|---|---|
| `SEQURE_LLVM_PATH` | `./codon-llvm` | Path to LLVM build directory (skipped if `install/lib/cmake/llvm` exists inside) |
| `SEQURE_CODON_PATH` | `./codon` | Path to Codon build directory (skipped if `install/` exists inside) |
| `SEQURE_SEQ_PATH` | `./codon-seq` | Path to Seq build directory |

---

## Manual build

### 1. Build LLVM

Sequre requires Codon's fork of LLVM:

```bash
git clone --depth 1 -b codon https://github.com/exaloop/llvm-project codon-llvm
cd codon-llvm

cmake -S llvm -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_TARGETS_TO_BUILD=all

cmake --build build
cmake --install build --prefix=$(pwd)/install
```

### 2. Build Codon

```bash
git clone https://github.com/exaloop/codon.git
cd codon

cmake -S . -B build -G Ninja \
    -DLLVM_DIR="<llvm-path>/install/lib/cmake/llvm" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

cmake --build build --config Release
cmake --install build --prefix=$(pwd)/install
```

Replace `<llvm-path>` with the absolute path to your LLVM build directory from step 1.

### 3. Build the Seq plugin

```bash
git clone https://github.com/exaloop/seq.git codon-seq
cd codon-seq

cmake -S . -B build -G Ninja \
    -DLLVM_DIR="<llvm-path>/install/lib/cmake/llvm" \
    -DCODON_PATH="<codon-path>/install" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

cmake --build build --config Release
cmake --install build --prefix="<codon-path>/install/lib/codon/plugins/seq"
```

### 4. Build Sequre

```bash
git clone https://github.com/0xTCG/sequre.git
cd sequre

cmake -S . -B build -G Ninja \
    -DLLVM_DIR="<llvm-path>/install/lib/cmake/llvm" \
    -DCODON_PATH="<codon-path>/install" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

cmake --build build --config Release
cmake --install build --prefix="<codon-path>/install/lib/codon/plugins/sequre"
```

After this, the `codon` binary at `<codon-path>/install/bin/codon` can load the Sequre plugin:

```bash
<codon-path>/install/bin/codon run -plugin sequre examples/local_run.codon
```

---

## Docker

Pre-built Docker images are available for CI and reproducible builds:

```bash
# Full Sequre build environment (Ubuntu)
docker pull hsmile/sequre:latest

# LLVM base image only
docker pull hsmile/llvm:17
```

The Dockerfiles in `docker/sequre/` and `docker/llvm/` mirror the manual build steps above and can serve as a reference for specific platforms (Ubuntu, manylinux).

---

## Verifying the build

Run the test suite to confirm everything works:

```bash
<codon-path>/install/bin/codon run -plugin sequre scripts/invoke.codon run-tests --local --all
```

Or if you have the `sequre` launcher installed:

```bash
sequre scripts/invoke.codon run-tests --local --all
```
