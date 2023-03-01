CC=clang
CXX=clang++
CODON_PATH=$(pwd)/codon
LLVM_INSTALL_PATH=$(pwd)/codon-llvm/install

# Build LLVM
git clone --depth 1 -b codon https://github.com/exaloop/llvm-project codon-llvm
cmake -S codon-llvm/llvm -G Ninja \
           -B codon-llvm/build \
           -DCMAKE_BUILD_TYPE=Release \
           -DLLVM_INCLUDE_TESTS=OFF \
           -DLLVM_ENABLE_RTTI=ON \
           -DLLVM_ENABLE_ZLIB=OFF \
           -DLLVM_ENABLE_TERMINFO=OFF \
           -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
           -DLLVM_BUILD_TOOLS=OFF \
           -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_PATH
cmake --build codon-llvm/build
cmake --install codon-llvm/build

# Build Codon
git clone https://github.com/HarisSmajlovic/codon.git
cd codon
git checkout sequre
git pull

cmake -S . -B build -G Ninja \
    -DLLVM_DIR=$LLVM_INSTALL_PATH/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX
cmake --build build --config Release
cmake --install build --prefix=install

cd ..

# Build Seq
git clone https://github.com/exaloop/seq.git
cd seq

cmake -S . -B build -G Ninja \
    -DLLVM_DIR=$LLVM_INSTALL_PATH/lib/cmake/llvm \
    -DCODON_PATH=$CODON_PATH/install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX
cmake --build build --config Release

cd ..

# Build Sequre
cmake -S . -B build -G Ninja \
    -DLLVM_DIR=$LLVM_INSTALL_PATH/lib/cmake/llvm \
    -DCODON_PATH=$CODON_PATH/install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX
cmake --build build --config Release

# Export paths to installed libs
export SEQ_PATH=$(pwd)/seq
export SEQURE_PATH=$(pwd)
