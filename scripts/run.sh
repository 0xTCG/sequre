#!/usr/bin/env bash

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;

echo "Setting up Sequre ..."
SEQURE_PATH=$(pwd)
SEQURE_STDLIB=$SEQURE_PATH/stdlib/sequre

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CP_OPTIONS=-uf
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export CP_OPTIONS=-f
fi

if [[ ${*:3} == *"--local"* ]]; then
    cp $CP_OPTIONS $SEQURE_STDLIB/network/unix_socket.codon $SEQURE_STDLIB/network/socket.codon
else
    cp $CP_OPTIONS $SEQURE_STDLIB/network/inet_socket.codon $SEQURE_STDLIB/network/socket.codon
fi

echo "Numpy plugin searched at $SEQURE_NUMPY_PATH ..."
echo "Seq plugin searched at $SEQURE_SEQ_PATH ..."
echo "Sequre plugin searched at $SEQURE_PATH ..."
echo "Compiling $2 in $1 mode ..."
OMP_PROC_BIND=close GC_INITIAL_HEAP_SIZE=8179869184 codon run -plugin $SEQURE_PATH -plugin $SEQURE_SEQ_PATH -plugin $SEQURE_NUMPY_PATH $1 scripts/invoke.codon run-$2 ${*:3}

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
