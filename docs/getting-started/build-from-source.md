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

Sequre requires Codon's fork of LLVM. The release build pins LLVM-codon 20.1.7, Codon v0.19.6 and seq v0.11.5; the commands below check out the same versions:

```bash
git clone --depth 1 -b codon-20.1.7 https://github.com/exaloop/llvm-project codon-llvm
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
git clone --depth 1 -b v0.19.6 https://github.com/exaloop/codon.git
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
git clone --depth 1 -b v0.11.5 https://github.com/exaloop/seq.git codon-seq
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

### 5. The launcher, the alignment shim and GMP

The release archives also carry the `sequre` launcher, the x86-64 alignment shim and a GMP library. Add them to the source build:

```bash
# The launcher
cc -O2 -o <codon-path>/install/bin/sequre sequre_launcher.c

# Linux x86_64: Codon's Ptr patch and the alignment shim (see below)
scripts/apply_avx512_workaround.sh <codon-path>/install

# GMP, where the Sequre runtime looks for the bundled copy (<plugin>/lib): the repository's library on x86-64 Linux
# and arm64 macOS; on Linux aarch64 one built from source (or skip this and rely on a system libgmp)
GMP_DIR=<codon-path>/install/lib/codon/plugins/sequre/lib
mkdir -p "$GMP_DIR"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) cp external/GMP/lib/libgmp.so "$GMP_DIR/" ;;
  Darwin-arm64) cp external/GMP/lib/libgmp.dylib "$GMP_DIR/" ;;
  Linux-aarch64) scripts/build_gmp.sh "$GMP_DIR/libgmp.so" ;;
esac

<codon-path>/install/bin/sequre examples/addmul.codon --local
```

The launcher adds what a bare `codon` command lacks: `--plugin=sequre`, `--disable-opt=core-pythonic-list-addition-opt` (Sequre's list `+` is element-wise; Codon would otherwise rewrite `l1 + l2 + l3` as a concatenation), the alignment shim and `CODON_PYTHON`. To run `codon` directly, pass the two flags yourself, and on Linux x86_64 preload the shim:

```bash
# Linux x86_64 (elsewhere, leave out the LD_PRELOAD)
LD_PRELOAD=<codon-path>/install/lib/sequre_align64.so \
  <codon-path>/install/bin/codon run --disable-opt=core-pythonic-list-addition-opt --plugin=sequre examples/addmul.codon --local
```

---

## x86-64 alignment workaround (Linux x86_64)

Codon (through v0.19.6) emits aligned vector moves for heap-resident SIMD
vectors, which the garbage collector only 16-byte-aligns, so on x86-64 CPUs
with AVX2 or AVX-512 Sequre's MHE/CKKS layer crashes with a `SIGSEGV`. After
building, apply the same workaround the installers use:

```bash
# from the sequre repo root, against your Codon install prefix
scripts/apply_avx512_workaround.sh <codon-path>/install
```

It patches Codon's `Ptr` load/store to be unaligned, builds the over-align
shim at `<codon-path>/install/lib/sequre_align64.so` (rebuilt when
`sequre_align64.c` is newer than it) and the launcher. The `sequre` launcher
preloads the shim for `sequre run` and links it into the binaries of
`sequre build`; if you invoke `codon` directly, set `LD_PRELOAD` to the shim for
`codon run`, and pass `-l :sequre_align64.so` to `codon build`. It is a no-op on
non-x86_64.

---

## Docker

The simplest reproducible build is via Docker — it builds Sequre against a pinned
Codon/LLVM toolchain (manylinux, for portable binaries):

```bash
scripts/install-via-docker.sh ~/.sequre
```

This uses `docker/local-build/` (Codon 0.19.6, LLVM-codon 20.1.7, seq 0.11.5) and applies the
x86-64 alignment workaround automatically; it runs on Linux only (on macOS, use the installer). The CI release build in
`.github/actions/build-manylinux/` mirrors it. (The older `docker/sequre/` and
`docker/llvm/` images predate the 0.19.6 toolchain and are unmaintained.)

---

## Verifying the build

Run the test suite from the repository root to confirm everything works:

```bash
<codon-path>/install/bin/sequre scripts/invoke.codon run-tests --local --all
```

With an installed release, the same command is `sequre scripts/invoke.codon run-tests --local --all`.
