#!/bin/sh -l
set -e

# Build script for sequre (Codon/LLVM/seq already pre-installed in image)
# Usage: /build-sequre.sh /path/to/sequre/source

SRC=$1
if [ -z "$SRC" ]; then
  echo "Usage: $0 <sequre-source-path>"
  exit 1
fi

# setup - detect if apt-get available (affects cmake target)
TEST=1
if [ -n "$(command -v apt-get)" ]; then
  TEST=0
fi

cd "$SRC"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCODON_PATH=$HOME/.sequre \
  -DLLVM_DIR=/opt/llvm-codon/lib/cmake/llvm \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX

if [ $TEST -eq 1 ]; then
  cmake --build build
else
  cmake --build build --target sequre
fi

cmake --install build --prefix=$HOME/.sequre/lib/codon/plugins/sequre

# Sequre's numpy is a fork (its ndarray is defined differently and is
# incompatible with Codon's own; for now — the plan is to eventually migrate
# Sequre to use Codon's numpy directly). It is shipped inside the plugin (via
# the `install(DIRECTORY stdlib ...)` rule) and imported exclusively through
# the `sequre.stdlib.numpy` namespace, so it never collides with Codon's
# native `numpy`, which is left intact.

# Build sequre launcher binary (auto-preloads the AVX-512 over-align shim)
$CC -O2 -o $HOME/.sequre/bin/sequre "$SRC/sequre_launcher.c"

# Build the AVX-512 over-align shim into lib/ (bundled via the lib/ tar entry
# below; install-via-docker.sh patches Codon's Ptr post-extraction). x86_64 only.
if [ "$(uname -m)" = "x86_64" ]; then
  $CC -shared -fPIC -O2 -o $HOME/.sequre/lib/sequre_align64.so "$SRC/sequre_align64.c" -ldl
fi

# Bundle GMP library
SEQURE_PREFIX=$HOME/.sequre/lib/codon/plugins/sequre
mkdir -p $SEQURE_PREFIX/lib
cp "$SRC/external/GMP/lib/libgmp.so" $SEQURE_PREFIX/lib/libgmp.so

# Create tarball in /output (include codon runtime so `sequre` launcher can find it)
tar czvf /output/sequre-linux-$(uname -m).tar.gz \
  -C $HOME/.sequre \
  bin/ \
  lib/

echo "Build complete!"
