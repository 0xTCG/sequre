FROM hsmile/llvm:12.0.1
LABEL Name=sequre Version=0.0.1

# Setup work environment
WORKDIR /sequre

# Install Seq & Sequre-IR
RUN \
    git clone https://github.com/HarisSmajlovic/seq.git && \
    cd seq && \
    git checkout sequre-v0.0.1 && \
    git pull && \
    mkdir build && \
    (cd build && cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_DIR=$(llvm-config --cmakedir) \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++) && \
    cmake --build build --config Release

# Add Sequre to Seq
ADD . .
RUN \
    mkdir seq/stdlib/sequre && \
    cp -r dsl/* seq/stdlib/sequre/
