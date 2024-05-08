FROM ubuntu:latest
LABEL Name=llvm Version=15.0.1

# Install dependencies
RUN \
    apt-get update && \
    apt-get -y install cmake protobuf-compiler git wget && \
    apt install -y clang

# Add CPM.cmake
RUN wget -O $(llvm-config --cmakedir)/../CPM.cmake https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake

# Build LLVM
RUN \
    git clone --depth 1 -b codon https://github.com/exaloop/llvm-project \
    && cmake -S llvm-project/llvm -B llvm-project/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_TARGETS_TO_BUILD=all \
    && cmake --build llvm-project/build \
    && cmake --install llvm-project/build --prefix=llvm-project/install
