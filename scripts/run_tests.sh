#!/usr/bin/env bash

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;

echo "Copying DSL to Seq ..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CP_OPTIONS=-ru
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export CP_OPTIONS=-R
fi

cp $CP_OPTIONS dsl/* seq/stdlib/sequre/

echo "Compiling tests ..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    /usr/bin/time -v seq/build/seqc run -release client.seq test-all
elif [[ "$OSTYPE" == "darwin"* ]]; then
    seq/build/seqc run -release client.seq test-all
fi

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
