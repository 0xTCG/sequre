CC=clang
CXX=clang++
export SEQURE_PATH=$(pwd)

# Requirements
if (echo a version 3.20.0; cmake --version) | sort -Vk3 | tail -1 | grep -q cmake
then
    echo "CMake version ok."
else
    echo "Error! CMake version invalid. Make sure to use CMake version >3.20.0" >&2
    return
fi

if [[ -z "${SEQURE_LLVM_PATH}" ]]; then
    export SEQURE_LLVM_PATH=$SEQURE_PATH/codon-llvm
    echo "LLVM path is not set. Using the local path: ${SEQURE_LLVM_PATH}"
fi

if [[ -z "${SEQURE_CODON_PATH}" ]]; then
    export SEQURE_CODON_PATH=$SEQURE_PATH/codon
    echo "Codon path is not set. Using the local path: ${SEQURE_CODON_PATH}"
fi

if [[ -z "${SEQURE_SEQ_PATH}" ]]; then
    export SEQURE_SEQ_PATH=$SEQURE_PATH/codon-seq
    echo "Seq-lang path is not set. Using the local path: ${SEQURE_SEQ_PATH}"
fi

# Build LLVM
if [ -d "${SEQURE_LLVM_PATH}/install/lib/cmake/llvm" ] 
then
    echo "Found existing LLVM installation." 
else
    echo "LLVM not installed. Proceeding with the installation ..."

    rm -rf $SEQURE_LLVM_PATH
    git clone --depth 1 -b codon https://github.com/exaloop/llvm-project $SEQURE_LLVM_PATH
    cd $SEQURE_LLVM_PATH
    
    cmake -S llvm -G Ninja -B build \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_INCLUDE_TESTS=OFF \
            -DLLVM_ENABLE_RTTI=ON \
            -DLLVM_ENABLE_ZLIB=OFF \
            -DLLVM_ENABLE_TERMINFO=OFF \
            -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
            -DLLVM_BUILD_TOOLS=OFF \
            -DCMAKE_INSTALL_PREFIX="${SEQURE_LLVM_PATH}/install"
    if [ $? -eq 0 ]; then
        echo "LLVM built."
    else
        echo "Error! LLVM build failed" >&2
        return
    fi

    cmake --build build
    if [ $? -eq 0 ]; then
        echo "LLVM installed."
    else
        echo "Error! LLVM installation failed" >&2
        return
    fi

    cmake --install build
    if [ $? -eq 0 ]; then
        echo "LLVM exported."
    else
        echo "Error! LLVM export failed" >&2
        return
    fi
fi

# Build Codon
if [ -d "${SEQURE_CODON_PATH}/install" ] 
then
    echo "Found existing Codon installation." 
else
    echo "Codon not installed. Proceeding with the installation ..."
    rm -rf $SEQURE_CODON_PATH
    git clone --branch sequre https://github.com/HarisSmajlovic/codon.git $SEQURE_CODON_PATH
    cd $SEQURE_CODON_PATH

    cmake -S . -B build -G Ninja \
        -DLLVM_DIR="${SEQURE_LLVM_PATH}/install/lib/cmake/llvm" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=$CC \
        -DCMAKE_CXX_COMPILER=$CXX
    if [ $? -eq 0 ]; then
        echo "Codon built."
    else
        echo "Error! Codon build failed" >&2
        return
    fi

    cmake --build build --config Release
    if [ $? -eq 0 ]; then
        echo "Codon installed."
    else
        echo "Error! Codon installation failed" >&2
        return
    fi

    cmake --install build --prefix="${SEQURE_CODON_PATH}/install"
    if [ $? -eq 0 ]; then
        echo "Codon exported."
    else
        echo "Error! Codon export failed" >&2
        return
    fi

fi

# Build Seq
if [ -d "${SEQURE_SEQ_PATH}/install" ] 
then
    echo "Found existing Seq-lang installation." 
else
    echo "Seq-lang not installed. Proceeding with the installation ..."
    rm -rf $SEQURE_SEQ_PATH
    git clone https://github.com/exaloop/seq.git $SEQURE_SEQ_PATH
    cd $SEQURE_SEQ_PATH

    cmake -S . -B build -G Ninja \
        -DLLVM_DIR="${SEQURE_LLVM_PATH}/install/lib/cmake/llvm" \
        -DCODON_PATH="${SEQURE_CODON_PATH}/install" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=$CC \
        -DCMAKE_CXX_COMPILER=$CXX
    if [ $? -eq 0 ]; then
        echo "Seq-lang built."
    else
        echo "Error! Seq-lang build failed" >&2
        return
    fi

    cmake --build build --config Release
    if [ $? -eq 0 ]; then
        echo "Seq-lang installed."
    else
        echo "Error! Seq-lang installation failed" >&2
        return
    fi

    cmake --install build --prefix="${SEQURE_SEQ_PATH}/install"
    if [ $? -eq 0 ]; then
        echo "Seq-lang exported."
    else
        echo "Error! Seq-lang export failed" >&2
        return
    fi

fi

# Build Codon-numpy
if [ -d "${SEQURE_NUMPY_PATH}/install" ] 
then
    echo "Found existing Codon-numpy installation." 
else
    echo "Codon-numpy not installed. Proceeding with the installation ..."
    rm -rf $SEQURE_NUMPY_PATH
    git clone https://github.com/HarisSmajlovic/codon-numpy.git $SEQURE_NUMPY_PATH
    cd $SEQURE_NUMPY_PATH

    cmake -S . -B build -G Ninja \
        -DLLVM_DIR="${SEQURE_LLVM_PATH}/install/lib/cmake/llvm" \
        -DCODON_PATH="${SEQURE_CODON_PATH}/install" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++
    if [ $? -eq 0 ]; then
        echo "Codon-numpy built."
    else
        echo "Error! Codon-numpy build failed" >&2
        return
    fi

    cmake --build build --config Release
    if [ $? -eq 0 ]; then
        echo "Codon-numpy installed."
    else
        echo "Error! Codon-numpy installation failed" >&2
        return
    fi

    cmake --install build --prefix="${SEQURE_NUMPY_PATH}/install"
    if [ $? -eq 0 ]; then
        echo "Codon-numpy exported."
    else
        echo "Error! Codon-numpy export failed" >&2
        return
    fi

fi

# Build Sequre
cd $SEQURE_PATH
rm -rf build
mkdir build
cmake -S . -B build -G Ninja \
    -DLLVM_DIR="${SEQURE_LLVM_PATH}/install/lib/cmake/llvm" \
    -DCODON_PATH="${SEQURE_CODON_PATH}/install" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX
if [ $? -eq 0 ]; then
    echo "Sequre built."
else
    echo "Error! Sequre build failed" >&2
    return
fi

cmake --build build --config Release
if [ $? -eq 0 ]; then
    echo "Sequre installed."
else
    echo "Error! Sequre installation failed" >&2
    return
fi
