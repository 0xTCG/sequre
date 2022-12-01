FROM llvm-12:manylinux as codon-llvm

RUN yum -y update
# This line might  be needed; we used nvidia centos that lacked these, but manylinux has them methinks
# RUN yum -y install centos-release-scl-rh epel-release
RUN yum -y install \
    ninja-build libuuid-devel openssl openssl-devel vim \
    libsodium-devel cmake3 zlib-devel git patch perl-Data-Dumper

# Setup work environment
WORKDIR /sequre

# Install Seq & Sequre-IR
RUN \
    git clone https://github.com/HarisSmajlovic/seq.git && \
    cd seq && \
    git checkout sequre-v0.0.1 && \
    mkdir build

RUN \
    cd seq/build && cmake3 .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/opt/llvm-codon/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/llvm-codon/bin/clang++ \
        -DLLVM_DIR=/opt/llvm-codon/lib/cmake/llvm

RUN cd seq && cmake3 --build build --config Release
    
# Add Sequre to Seq
COPY . .
RUN \
    mkdir seq/stdlib/sequre && \
    cp -r dsl/* seq/stdlib/sequre/
