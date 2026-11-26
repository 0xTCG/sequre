#!/bin/sh -l
set -e

# Build script for sequre (Codon/LLVM already pre-installed in image)
# Usage: /build-sequre.sh /path/to/sequre/source

SRC=$1
if [ -z "$SRC" ]; then
  echo "Usage: $0 <sequre-source-path>"
  exit 1
fi

cd "$SRC"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCODON_PATH=$HOME/.sequre \
  -DLLVM_DIR=/opt/llvm-codon/lib/cmake/llvm \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX
cmake --build build --target sequre
cmake --install build --prefix=$HOME/.sequre/lib/codon/plugins/sequre

# Build the sequre launcher (it preloads the x86-64 over-align shim for `sequre run` and links it into the binaries
# of `sequre build`)
$CC -O2 -o $HOME/.sequre/bin/sequre "$SRC/sequre_launcher.c"

# Build the x86-64 over-align shim into lib/ (bundled via the lib/ tar entry below; install-via-docker.sh patches
# Codon's Ptr post-extraction). x86_64 only.
if [ "$(uname -m)" = "x86_64" ]; then
  $CC -shared -fPIC -O2 -o $HOME/.sequre/lib/sequre_align64.so "$SRC/sequre_align64.c" -ldl
fi

# Bundle GMP built for the target, as the release build does: the repository's x86-64 copy, or one built from source
# on other architectures
SEQURE_PREFIX=$HOME/.sequre/lib/codon/plugins/sequre
mkdir -p $SEQURE_PREFIX/lib
case "$(uname -m)" in
  x86_64) cp "$SRC/external/GMP/lib/libgmp.so" $SEQURE_PREFIX/lib/libgmp.so ;;
  *) sh "$SRC/scripts/build_gmp.sh" $SEQURE_PREFIX/lib/libgmp.so ;;
esac

# Create tarball in /output (include codon runtime so `sequre` launcher can find it)
TARBALL=/output/sequre-linux-$(uname -m).tar.gz
tar czf "$TARBALL" \
  -C $HOME/.sequre \
  bin/ \
  lib/

# Every binary in it must be built for this machine
sh "$SRC/scripts/check_machine_types.sh" linux-$(uname -m) "$TARBALL"

echo "Build complete!"
