#!/usr/bin/env bash

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

echo "Compiling Codon ..."
(cd seq && cmake --build build --config Release)
