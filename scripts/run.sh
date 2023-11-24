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
echo "Compiling $2 in $1 mode ..."
GC_INITIAL_HEAP_SIZE=8179869184 $SEQURE_CODON_PATH/bin/codon run -plugin $SEQURE_PATH -plugin $SEQURE_SEQ_PATH $1 scripts/invoke.codon run-$2 ${*:3}

echo "Cleaning up sockets ..."
find . -name 'sock.*' -exec rm {} \;
