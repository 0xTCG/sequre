#!/bin/sh -l
set -e

# setup
TEST=1
if [ -n "$(command -v apt-get)" ]
then
  TEST=0
fi

case "$(uname -s)" in
  Darwin*)    OPT=/opt;;
  *)          OPT=/opt
esac

mkdir $HOME/.sequre
cd $HOME/.sequre
curl -L https://github.com/exaloop/codon/releases/download/v0.17.0/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - --strip-components=1
mkdir -p $OPT
LLVM_TAR=$(curl -L https://github.com/exaloop/llvm-project/releases/download/codon-15.0.1/llvm-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz -o /tmp/llvm.tar.gz && echo /tmp/llvm.tar.gz)
LLVM_TOP=$(tar tzf "$LLVM_TAR" | head -1 | cut -d/ -f1)
if [ "$LLVM_TOP" = "opt" ]; then
  tar zxvf "$LLVM_TAR" -C /
else
  tar zxvf "$LLVM_TAR" -C $OPT
fi
rm -f "$LLVM_TAR"
cd $HOME
curl -L https://github.com/exaloop/seq/releases/download/v0.11.3/seq-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C .sequre/lib/codon/plugins

# Use LLVM-codon clang on Linux; system clang on macOS (darwin LLVM tarball has no clang)
if [ -f "$OPT/llvm-codon/bin/clang" ]; then
  CC=$OPT/llvm-codon/bin/clang
  CXX=$OPT/llvm-codon/bin/clang++
else
  CC=$(command -v clang)
  CXX=$(command -v clang++)
fi

cd $1
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCODON_PATH=$HOME/.sequre \
  -DLLVM_DIR=$OPT/llvm-codon/lib/cmake/llvm \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX
if [ $TEST -eq 1 ]
then
  cmake --build build
else
  cmake --build build --target sequre
fi
cmake --install build --prefix=$HOME/.sequre/lib/codon/plugins/sequre

# Build sequre launcher binary
$CC -O2 -o $HOME/.sequre/bin/sequre $1/sequre_launcher.c

# Bundle platform-appropriate GMP library
SEQURE_PREFIX=$HOME/.sequre/lib/codon/plugins/sequre
mkdir -p $SEQURE_PREFIX/lib
case "$(uname -s)" in
  Darwin*) cp $1/external/GMP/lib/libgmp.dylib $SEQURE_PREFIX/lib/libgmp.dylib ;;
  *)       cp $1/external/GMP/lib/libgmp.so     $SEQURE_PREFIX/lib/libgmp.so    ;;
esac

tar czvf sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz -C $HOME/.sequre bin/sequre lib/codon/plugins/sequre lib/codon/plugins/seq
echo "Done"
