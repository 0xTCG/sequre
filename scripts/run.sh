#!/usr/bin/env bash

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;

echo "Setting up Sequre ..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CP_OPTIONS=-ruf
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export CP_OPTIONS=-Rf
fi

cp $CP_OPTIONS dsl/* seq/stdlib/sequre/
if [[ ${*:2} == *"--local"* ]]; then
    mv seq/stdlib/sequre/network/unix_socket.seq seq/stdlib/sequre/network/socket.seq
else
    mv seq/stdlib/sequre/network/inet_socket.seq seq/stdlib/sequre/network/socket.seq
fi

echo "Compiling $1 ..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    GC_INITIAL_HEAP_SIZE=8589934592 /usr/bin/time -v seq/build/seqc run -release client.seq run-$1 ${*:2}
elif [[ "$OSTYPE" == "darwin"* ]]; then
    GC_INITIAL_HEAP_SIZE=8589934592 seq/build/seqc run -release client.seq run-$1 ${*:2}
fi

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
