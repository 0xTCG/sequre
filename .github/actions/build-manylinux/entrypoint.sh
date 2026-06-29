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

# Pick the C/C++ compiler:
#  - macOS: Apple's system clang. It knows the SDK sysroot for both the plugin
#    build and the standalone launcher. (The LLVM-codon darwin tarball now ships
#    a clang too, but it has no default sysroot, so system headers like glob.h
#    are not found when building the launcher directly.)
#  - Linux: the LLVM-codon clang, to match the bundled LLVM toolchain.
case "$(uname -s)" in
  Darwin*)
    CC=$(command -v clang)
    CXX=$(command -v clang++)
    ;;
  *)
    if [ -f "$OPT/llvm-codon/bin/clang" ]; then
      CC=$OPT/llvm-codon/bin/clang
      CXX=$OPT/llvm-codon/bin/clang++
    else
      CC=$(command -v clang)
      CXX=$(command -v clang++)
    fi
    ;;
esac

# On manylinux_2_28 the default toolchain is a gcc-toolset under /opt/rh, which
# the LLVM-codon clang does not auto-detect, so it cannot find libstdc++ (or the
# C++20 headers) when linking the plugin. Point clang at it via a config file,
# matching how Codon builds its own plugins (see exaloop/codon
# .github/build-linux/Dockerfile.linux-x86_64). No-op off manylinux: the glob is
# empty (e.g. on macOS), so no config file is written.
if [ -f "$OPT/llvm-codon/bin/clang" ]; then
  GCC_INSTALL_DIR=$(ls -d /opt/rh/gcc-toolset-*/root/usr/lib/gcc/*/* 2>/dev/null | sort -V | tail -1)
  if [ -n "$GCC_INSTALL_DIR" ]; then
    echo "--gcc-install-dir=$GCC_INSTALL_DIR" > "$OPT/llvm-codon/bin/clang.cfg"
    echo "--gcc-install-dir=$GCC_INSTALL_DIR" > "$OPT/llvm-codon/bin/clang++.cfg"
  fi
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
