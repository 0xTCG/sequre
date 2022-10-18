#!/usr/bin/env bash

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;

echo "Setting up Sequre ..."
SEQURE_PATH=seq/stdlib/sequre

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CP_OPTIONS=-ruf
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export CP_OPTIONS=-Rf
fi

rm -rf $SEQURE_PATH
mkdir $SEQURE_PATH
cp $CP_OPTIONS dsl/* $SEQURE_PATH
if [[ ${*:2} == *"--local"* ]]; then
    mv $SEQURE_PATH/network/unix_socket.seq $SEQURE_PATH/network/socket.seq
else
    mv $SEQURE_PATH/network/inet_socket.seq $SEQURE_PATH/network/socket.seq
fi

echo "Compiling $1 ..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    GC_INITIAL_HEAP_SIZE=8179869184 GC_LIMIT=8179869184 seq/build/seqc run -release scripts/invoke.seq run-$1 ${*:2}
elif [[ "$OSTYPE" == "darwin"* ]]; then
    GC_INITIAL_HEAP_SIZE=8179869184 GC_LIMIT=8179869184 seq/build/seqc run -release scripts/invoke.seq run-$1 ${*:2}
fi

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
