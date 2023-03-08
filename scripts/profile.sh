#!/usr/bin/env bash

echo "Setting up Sequre ..."
SEQURE_PATH=$(pwd)
SEQURE_STDLIB=$SEQURE_PATH/stdlib/sequre

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CP_OPTIONS=-uf
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export CP_OPTIONS=-f
fi

echo "Seq plugin searched at $SEQ_PATH ..."
echo "Sequre plugin searched at $SEQURE_PATH ..."
echo "Building $1 ..."
codon build -exe -plugin $SEQURE_PATH -plugin $SEQ_PATH $1 -o temp_perf.exe

echo "Profiling $1 ..."
perf stat -e L1-dcache-load-misses:u,LLC-load-misses:u,cache-misses:u,cache-references:u,branch-misses:u,page-faults:u,cycles:u ./temp_perf.exe
