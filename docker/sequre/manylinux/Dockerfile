FROM llvm:manylinux as codon-llvm

RUN yum -y update
# This line might  be needed; we used nvidia centos that lacked these, but manylinux has them methinks
# RUN yum -y install centos-release-scl-rh epel-release
RUN yum -y install \
    ninja-build libuuid-devel openssl openssl-devel vim \
    libsodium-devel cmake3 zlib-devel bzip2-devel git patch perl-Data-Dumper

# Env vars
ENV CODON_PYTHON=/usr/lib64/libpython3.so
ENV SEQURE_LLVM_PATH=/opt/llvm-codon
ENV SEQURE_CODON_PATH=/codon
ENV SEQURE_SEQ_PATH=/codon-seq
ENV SEQURE_NUMPY_PATH=/codon-numpy
ENV SEQURE_PATH=/sequre
ENV CC=${SEQURE_LLVM_PATH}/bin/clang
ENV CXX=${SEQURE_LLVM_PATH}/bin/clang++

# Install Codon
RUN \
    git clone --branch sequre https://github.com/HarisSmajlovic/codon.git ${SEQURE_CODON_PATH}  && \
    cd $SEQURE_CODON_PATH && \
    cmake -S . -B build -G Ninja \
        -DLLVM_DIR=${SEQURE_LLVM_PATH}/install/lib/cmake/llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} && \
    cmake --build build --config Release && \
    cmake --install build --prefix=${SEQURE_CODON_PATH}/install

# Install Seq
RUN \
    git clone https://github.com/exaloop/seq.git ${SEQURE_SEQ_PATH} && \
    cd $SEQURE_SEQ_PATH && \
    cmake -S . -B build -G Ninja \
        -DLLVM_DIR=${SEQURE_LLVM_PATH}/install/lib/cmake/llvm \
        -DCODON_PATH=${SEQURE_CODON_PATH}/install \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} && \
    cmake --build build --config Release && \
    cmake --install build --prefix=${SEQURE_SEQ_PATH}/install

# Install NumPy
RUN \
    git clone --branch docker https://github.com/HarisSmajlovic/codon-numpy.git ${SEQURE_NUMPY_PATH} && \
    cd $SEQURE_NUMPY_PATH && \
    cmake -S . -B build -G Ninja \
        -DLLVM_DIR=${SEQURE_LLVM_PATH}/install/lib/cmake/llvm \
        -DCODON_PATH=${SEQURE_CODON_PATH}/install \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ && \
    cmake --build build --config Release && \
    cmake --install build --prefix=${SEQURE_NUMPY_PATH}/install

# Install Sequre
RUN \
    git clone https://github.com/0xTCG/sequre-mhe.git ${SEQURE_PATH} && \
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
