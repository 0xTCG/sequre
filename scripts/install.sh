git clone --depth 1 -b release/12.x https://github.com/llvm/llvm-project
mkdir -p llvm-project/llvm/build
(cd llvm-project/llvm/build && cmake .. \
    -DCMAKE_INSTALL_PREFIX=$HOME/llvm-build
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_TARGETS_TO_BUILD=host && make && make install)

git clone https://github.com/HarisSmajlovic/seq.git
cd seq
git checkout sequre-v0.0.1
git pull
mkdir stdlib/sequre
cp -r ../dsl/* stdlib/sequre/

mkdir build
(cd build && cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DLLVM_DIR=$HOME/llvm-build/lib/cmake/llvm \
         -DCMAKE_C_COMPILER=clang \
         -DCMAKE_CXX_COMPILER=clang++)
cmake --build build --config Release
