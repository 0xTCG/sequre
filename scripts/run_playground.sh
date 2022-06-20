#!/usr/bin/env bash

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;

echo "Copying DSL to Seq ..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CP_OPTIONS=-ruf
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export CP_OPTIONS=-Rf
fi

cp $CP_OPTIONS dsl/* seq/stdlib/sequre/

echo "Compiling playground ..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    GC_INITIAL_HEAP_SIZE=7144000000 /usr/bin/time -v seq/build/seqc run -release client.seq test-all --test-run $1
elif [[ "$OSTYPE" == "darwin"* ]]; then
    GC_INITIAL_HEAP_SIZE=7144000000 seq/build/seqc run -release client.seq test-all --test-run $1
fi

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
