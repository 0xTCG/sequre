#!/bin/sh -l
set -e

# setup
TEST=1
if [ -n "$(command -v apt-get)" ]
then
  TEST=0
fi

case "$(uname -s)" in
  Darwin*)    OPT=/usr/local;;
  *)          OPT=/opt
esac

mkdir $HOME/.codon
cd $HOME/.codon
curl -L https://github.com/exaloop/codon/releases/download/v0.16.3/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - --strip-components=1
cd /
curl -L https://github.com/exaloop/llvm-project/releases/download/codon-15.0.1/llvm-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf -

cd $1
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCODON_PATH=$HOME/.codon \
  -DLLVM_DIR=/opt/llvm-codon/lib/cmake/llvm \
  -DCMAKE_C_COMPILER=/opt/llvm-codon/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/llvm-codon/bin/clang++
if [ $TEST -eq 1 ]
then
  cmake --build build
else
  cmake --build build --target sequre
fi
cmake --install build --prefix=$HOME/.codon/lib/codon/plugins/sequre
tar czvf sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz -C $HOME/.codon/lib/codon/plugins sequre
echo "Done"
