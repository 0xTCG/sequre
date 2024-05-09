FROM hsmile/llvm:latest
LABEL Name=sequre Version=0.0.1

# Args
ARG SEQURE_CODON_PATH=codon
ARG SEQURE_SEQ_PATH=seq
ARG SEQURE_PATH=sequre
ARG CC=clang
ARG CXX=clang++

# Install dependencies
RUN \
    apt-get update && \
    apt install -y libbz2-dev

# Install Codon
RUN \
    git clone https://github.com/exaloop/codon.git ${SEQURE_CODON_PATH}  && \
    cd $SEQURE_CODON_PATH && \
    cmake -S . -B build \
        -DLLVM_DIR=$(llvm-config --cmakedir) \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} && \
    cmake --build build --config Release && \
    cmake --install build --prefix=install

# Install Seq
RUN \
    git clone https://github.com/exaloop/seq.git ${SEQURE_SEQ_PATH} && \
    cd $SEQURE_SEQ_PATH && \
    cmake -S . -B build \
        -DLLVM_DIR=$(llvm-config --cmakedir) \
        -DCODON_PATH=${SEQURE_CODON_PATH}/install \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} && \
    cmake --build build --config Release && \
    cmake --install build --prefix=install

# Install Sequre
RUN \
    git clone https://github.com/0xTCG/sequre.git ${SEQURE_PATH} && \
    cd ${SEQURE_PATH} && \
    rm -rf build && \
    mkdir build && \
    cmake -S . -B build \
        -DLLVM_DIR=$(llvm-config --cmakedir) \
        -DCODON_PATH=${SEQURE_CODON_PATH}/install \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} && \
    cmake --build build --config Release

WORKDIR /sequre
