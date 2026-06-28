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
curl -L https://github.com/exaloop/codon/releases/download/v0.19.6/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - --strip-components=1
mkdir -p $OPT
LLVM_TAR=$(curl -L https://github.com/exaloop/llvm-project/releases/download/codon-20.1.7/llvm-codon-20.1.7-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.bz2 -o /tmp/llvm.tar.bz2 && echo /tmp/llvm.tar.bz2)
LLVM_TOP=$(tar -tjf "$LLVM_TAR" | head -1 | cut -d/ -f1)
if [ "$LLVM_TOP" = "opt" ]; then
  tar -jxvf "$LLVM_TAR" -C /
else
  tar -jxvf "$LLVM_TAR" -C $OPT
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

# Sequre's numpy is a fork, not patches — Sequre's ndarray is
# defined differently and is incompatible with Codon's own (for now; the
# plan is to eventually migrate Sequre to use Codon's numpy directly).
# `import numpy` resolves from $CODON_PATH/lib/codon/stdlib
# (the package root), not from the plugin's stdlib dir, so Codon's numpy/ is
# wholesale replaced here rather than merged file-by-file, to avoid leaving
# any of Codon's original numpy files behind referencing a mismatched
# ndarray type.
CODON_STDLIB=$HOME/.sequre/lib/codon/stdlib
rm -rf $CODON_STDLIB/numpy
cp -r $1/stdlib/numpy $CODON_STDLIB/numpy

# Build sequre launcher binary
$CC -O2 -o $HOME/.sequre/bin/sequre $1/sequre_launcher.c

# Bundle platform-appropriate GMP library
SEQURE_PREFIX=$HOME/.sequre/lib/codon/plugins/sequre
mkdir -p $SEQURE_PREFIX/lib
case "$(uname -s)" in
  Darwin*) cp $1/external/GMP/lib/libgmp.dylib $SEQURE_PREFIX/lib/libgmp.dylib ;;
  *)       cp $1/external/GMP/lib/libgmp.so     $SEQURE_PREFIX/lib/libgmp.so    ;;
esac

# Note: only Sequre's own files are included (not the whole lib/codon/stdlib
# tree) so the tarball stays scoped to Sequre's own files; install.sh
# extracts Codon itself separately, and removes Codon's own numpy/ before
# extracting this tarball so Sequre's fork fully replaces it.
tar czvf sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz -C $HOME/.sequre \
  bin/sequre \
  lib/codon/plugins/sequre \
  lib/codon/plugins/seq \
  lib/codon/stdlib/numpy
echo "Done"
