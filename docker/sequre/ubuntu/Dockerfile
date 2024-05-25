FROM hsmile/llvm:latest
LABEL Name=sequre Version=0.0.1

# Install dependencies
RUN apt-get update
RUN apt install -y \
    ninja-build openssl vim time \
    cmake libbz2-dev git patch clang

# Env vars
ENV CODON_PYTHON=/usr/lib64/libpython3.so
ENV SEQURE_LLVM_PATH=/llvm-project/llvm
ENV SEQURE_CODON_PATH=/codon
ENV SEQURE_SEQ_PATH=/codon-seq
ENV SEQURE_NUMPY_PATH=/codon-numpy
ENV SEQURE_PATH=/sequre
ENV CC=clang
ENV CXX=clang++

# Install Codon
RUN \
    git clone https://github.com/exaloop/codon.git ${SEQURE_CODON_PATH}  && \
    cd $SEQURE_CODON_PATH && \
    cmake -S . -B build -G Ninja \
        -DLLVM_DIR=$(llvm-config --cmakedir) \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} && \
    cmake --build build --config Release && \
    cmake --install build --prefix=${SEQURE_CODON_PATH}/install

# Install Seq
RUN \
    git clone --depth 1 https://github.com/exaloop/seq.git ${SEQURE_SEQ_PATH} && \
    cd $SEQURE_SEQ_PATH && \
    cmake -S . -B build -G Ninja \
        -DLLVM_DIR=$(llvm-config --cmakedir) \
        -DCODON_PATH=${SEQURE_CODON_PATH}/install \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} && \
    cmake --build build --config Release && \
    cmake --install build --prefix=${SEQURE_SEQ_PATH}/install

# Install Sequre
RUN \
    git clone --depth 1 -b sheqi https://github.com/0xTCG/sequre.git ${SEQURE_PATH} && \
    cd ${SEQURE_PATH} && \
    rm -rf build && \
    mkdir build && \
    cmake -S . -B build -G Ninja \
        -DLLVM_DIR=${SEQURE_LLVM_PATH}/install/lib/cmake/llvm \
        -DCODON_PATH=${SEQURE_CODON_PATH}/install \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} && \
    cmake --build build --config Release

WORKDIR /sequre

# Compile
RUN CODON_DEBUG=lt scripts/run.sh -release build --build-only
