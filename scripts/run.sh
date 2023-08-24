#!/usr/bin/env bash

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;

if [[ -z "${SEQURE_CODON_PATH}" ]]; then
    echo "Error! SEQURE_CODON_PATH env variable not set" >&2
    return
fi

if [[ -z "${SEQURE_SEQ_PATH}" ]]; then
    echo "Error! SEQURE_SEQ_PATH env variable not set" >&2
    return
fi

if [[ -z "${SEQURE_NUMPY_PATH}" ]]; then
    echo "Error! SEQURE_NUMPY_PATH env variable not set" >&2
    return
fi

if [[ -z "${SEQURE_PATH}" ]]; then
    echo "Error! SEQURE_PATH env variable not set" >&2
    return
fi

if [ ! -d "${SEQURE_CODON_PATH}/install" ]
then
    echo "Codon not installed at ${SEQURE_CODON_PATH}" >&2
    return
fi

if [ ! -d "${SEQURE_SEQ_PATH}/install" ]
then
    echo "Seq-lang not installed at ${SEQURE_SEQ_PATH}" >&2
    return
fi

if [ ! -d "${SEQURE_NUMPY_PATH}/build" ]
then
    echo "Codon-numpy not installed at ${SEQURE_NUMPY_PATH}" >&2
    return
fi

echo "Codon found at $SEQURE_CODON_PATH"
echo "Seq-lang plugin found at $SEQURE_SEQ_PATH"
echo "Numpy plugin found at $SEQURE_NUMPY_PATH"
echo "Sequre plugin found at $SEQURE_PATH"
echo "Compiling $2 in $1 mode ..."
GC_INITIAL_HEAP_SIZE=8179869184 codon run -plugin $SEQURE_PATH -plugin $SEQURE_SEQ_PATH -plugin $SEQURE_NUMPY_PATH $1 scripts/invoke.codon run-$2 ${*:3}

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
