# Build from Source

This page describes how to compile Sequre and all its dependencies from source.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| `clang` / `clang++` | C++20 support | GCC may work but is untested |
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

## Build instructions

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

Replace `<llvm-path>` with the absolute path to LLVM build directory from step 1.

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
<codon-path>/install/bin/codon run --plugin=sequre examples/addmul.codon
```

---

## AVX-512 alignment workaround (Linux x86_64)

Codon (through v0.19.6) miscompiles heap-resident SIMD vectors on x86-64 CPUs
with AVX-512 — Sequre's MHE/CKKS layer triggers it as a `SIGSEGV` in `--local`
runs. After building, apply the same workaround the installers use:

```bash
# from the sequre repo root, against your Codon install prefix
scripts/apply_avx512_workaround.sh <codon-path>/install
```

It patches Codon's `Ptr` load/store to be unaligned and builds an over-align
`LD_PRELOAD` shim at `<codon-path>/install/lib/sequre_align64.so`. The `sequre`
launcher preloads the shim automatically; if you invoke `codon` directly, set
`LD_PRELOAD` to that shim yourself. It is a no-op on non-x86_64.

---

## Docker

The simplest reproducible build is via Docker — it builds Sequre against a pinned
Codon/LLVM toolchain (manylinux, for portable binaries):

```bash
scripts/install-via-docker.sh ~/.sequre
```

This uses `docker/local-build/` (Codon 0.19.6, LLVM-codon 20.1.7) and applies the
AVX-512 workaround automatically; the CI release build in
`.github/actions/build-manylinux/` mirrors it. (The older `docker/sequre/` and
`docker/llvm/` images predate the 0.19.6 toolchain and are unmaintained.)

---

## Verifying the build

Run the test suite to confirm everything works:

```bash
<codon-path>/install/bin/codon run --plugin=sequre scripts/invoke.codon run-tests --local --all
```

Or if `sequre` launcher is installed:

```bash
sequre scripts/invoke.codon run-tests --local --all
```
