export CC=clang
export CXX=clang++ 

git clone --depth 1 -b release/12.x https://github.com/llvm/llvm-project
mkdir -p llvm-project/llvm/build
cd llvm-project/llvm/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_TARGETS_TO_BUILD=host
make
make install

git clone https://github.com/HarisSmajlovic/seq.git
cd seq
git checkout feature/sequre-v0.0.1
mkdir stdlib/sequre
cp -r dsl/* stdlib/sequre/

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LIBEXT=so
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export LIBEXT=dylib
fi

mkdir build
ln -s $(pwd)/deps/lib/libomp.${LIBEXT} $(pwd)/build/libomp.${LIBEXT}
export SEQ_HTSLIB=$(pwd)/deps/lib/libhts.${LIBEXT}
mkdir build
(cd build && cmake .. -DCMAKE_BUILD_TYPE=Release \
                      -DLLVM_DIR=$(llvm-config --cmakedir) \
                      -DCMAKE_C_COMPILER=${CC} \
                      -DCMAKE_CXX_COMPILER=${CXX})
cmake --build build --config Release
