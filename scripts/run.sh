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
if [[ ${*:3} == *"--local"* ]]; then
    mv seq/stdlib/sequre/network/unix_socket.seq seq/stdlib/sequre/network/socket.seq
else
    mv seq/stdlib/sequre/network/inet_socket.seq seq/stdlib/sequre/network/socket.seq
fi

echo "Compiling $2 ..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    GC_INITIAL_HEAP_SIZE=8144000000 /usr/bin/time -v seq/build/seqc run -$1 client.seq run-$2 ${*:3}
elif [[ "$OSTYPE" == "darwin"* ]]; then
    GC_INITIAL_HEAP_SIZE=8144000000 seq/build/seqc run -$1 client.seq run-$2 ${*:3}
fi

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
