#!/usr/bin/env bash

echo "Cleaning up sockets ..."
find . -name 'sock.*' -exec rm {} \;

if [[ -z "${SEQURE_CODON_PATH}" ]]; then
    echo "Error! SEQURE_CODON_PATH env variable not set" >&2
    return
fi

if [[ -z "${SEQURE_SEQ_PATH}" ]]; then
    echo "Error! SEQURE_SEQ_PATH env variable not set" >&2
    return
fi

if [[ -z "${SEQURE_PATH}" ]]; then
    echo "Error! SEQURE_PATH env variable not set" >&2
    return
fi

if [ ! -d "${SEQURE_CODON_PATH}" ]
then
    echo "Codon not installed at ${SEQURE_CODON_PATH}" >&2
    return
fi

if [ ! -d "${SEQURE_SEQ_PATH}" ]
then
    echo "Seq-lang not installed at ${SEQURE_SEQ_PATH}" >&2
    return
fi

echo "Codon path: $SEQURE_CODON_PATH"
echo "Seq-lang plugin path: $SEQURE_SEQ_PATH"
echo "Sequre plugin path: $SEQURE_PATH"

if [[ $* == *--jit* ]]
then
    echo "Running $2 in $1 mode ..."
    /usr/bin/time -v $SEQURE_CODON_PATH/build/codon run --disable-opt="core-pythonic-list-addition-opt" -plugin $SEQURE_PATH -plugin $SEQURE_SEQ_PATH $1 scripts/invoke.codon run-$2 ${*:3}
    echo "Cleaning up sockets ..."
    find . -name 'sock.*' -exec rm {} \;
else
    if [[ $* == *--build* ]]
    then
        rm -f ./sequrex
        echo "Compiling $2 in $1 mode ..."
        CC=clang CXX=clang++ $SEQURE_CODON_PATH/build/codon build --disable-opt="core-pythonic-list-addition-opt" -plugin $SEQURE_PATH -plugin $SEQURE_SEQ_PATH $1 -o sequrex scripts/invoke.codon
    fi

    if [ ! -f "./sequrex" ]
    then
        echo "Sequre is not built. Make sure to add --build flag in first run." >&2
        return
    fi

    if [[ ! $* == *--build-only* ]]
    then
        echo "Running $2 in $1 mode ..."
        /usr/bin/time -v ./sequrex run-$2 ${*:3}
    fi
fi

echo "Cleaning up sockets ..."
find . -name 'sock.*' -exec rm {} \;
