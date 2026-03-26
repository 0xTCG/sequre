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

# Build sequre launcher binary
$CC -O2 -o $HOME/.sequre/bin/sequre "$SRC/sequre_launcher.c"

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
