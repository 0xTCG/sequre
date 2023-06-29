FROM ubuntu:latest
LABEL Name=llvm Version=12.0.1

# Install dependencies
RUN \
    apt-get update && \
    apt-get -y install cmake protobuf-compiler git wget && \
    apt install -y clang

# Build LLVM 12
RUN \
    git clone --depth 1 -b codon https://github.com/llvm/llvm-project && \
    mkdir -p llvm-project/llvm/build && \
    (cd llvm-project/llvm/build && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_TARGETS_TO_BUILD=host && make && make install)

# Add CPM.cmake
RUN wget -O $(llvm-config --cmakedir)/../CPM.cmake https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake
