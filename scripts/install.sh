export CC=clang
export CXX=clang++ 

git clone https://github.com/HarisSmajlovic/seq.git
cd seq
git checkout sequre-seq-v0.10
./scripts/deps.sh 4
cd ..

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LIBEXT=so
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export LIBEXT=dylib
fi

mkdir seq/stdlib/sequre
cp -r dsl/* seq/stdlib/sequre/
      
cd seq
mkdir build
ln -s $(pwd)/deps/lib/libomp.${LIBEXT} $(pwd)/build/libomp.${LIBEXT}
export SEQ_HTSLIB=$(pwd)/deps/lib/libhts.${LIBEXT}
(cd build && cmake .. -DCMAKE_BUILD_TYPE=Release \
                    -DSEQ_DEP=$(pwd)/../deps \
                    -DCMAKE_C_COMPILER=${CC} \
                    -DCMAKE_CXX_COMPILER=${CXX})
cmake --build build --config Release -- VERBOSE=1
