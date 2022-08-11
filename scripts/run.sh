#!/usr/bin/env bash

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;

echo "Setting up Sequre ..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CP_OPTIONS=-ruf
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export CP_OPTIONS=-Rf
fi

cp $CP_OPTIONS dsl/* new_seq/stdlib/sequre/
if [[ ${*:2} == *"--local"* ]]; then
    mv new_seq/stdlib/sequre/network/unix_socket.seq new_seq/stdlib/sequre/network/socket.seq
else
    mv new_seq/stdlib/sequre/network/inet_socket.seq new_seq/stdlib/sequre/network/socket.seq
fi

echo "Compiling $1 ..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    GC_INITIAL_HEAP_SIZE=8179869184 GC_LIMIT=8179869184 /usr/bin/time -v new_seq/build/seqc run -release scripts/invoke.seq run-$1 ${*:2}
elif [[ "$OSTYPE" == "darwin"* ]]; then
    GC_INITIAL_HEAP_SIZE=8179869184 GC_LIMIT=8179869184 new_seq/build/seqc run -release scripts/invoke.seq run-$1 ${*:2}
fi

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
