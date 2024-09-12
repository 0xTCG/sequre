FROM ubuntu:latest

ENV HECO_PATH=/heco

RUN apt-get update
RUN apt-get install -y git cmake clang lld ninja-build

RUN git clone --depth 1 https://github.com/MarbleHE/HECO.git ${HECO_PATH} && cd ${HECO_PATH}
WORKDIR ${HECO_PATH}

RUN git submodule update --init --recursive --progress

RUN mkdir dependencies/llvm-project/build && \
    cd dependencies/llvm-project/build && \
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_TARGETS_TO_BUILD=X86 \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_LLD=ON \
        -DLLVM_CCACHE_BUILD=OFF \
        -DLLVM_INSTALL_UTILS=ON \
        -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF && \
    cmake --build . --target check-mlir 

RUN cd dependencies/SEAL && \
    cmake -S . -B build && \
    cmake --build build && \
    cmake --install build

RUN mkdir build && \
    cmake -S . -B build -DMLIR_DIR=dependencies/llvm-project/build/lib/cmake/mlir && \
    cmake --build build --target heco && \
    cmake --build build --target emitc-translate

RUN apt-get install -y vim
COPY benchmark.cpp ${HECO_PATH}/evaluation/benchmark/
COPY heco_input/*.mlir ${HECO_PATH}/evaluation/benchmark/heco_input/
COPY heco_output/*.h ${HECO_PATH}/evaluation/benchmark/heco_output/
COPY util/BenchmarkHelper.h ${HECO_PATH}/evaluation/util

RUN ./evaluation/benchmark/heco_helper.sh
RUN cmake --build build --target benchmark
RUN ./build/bin/benchmark
