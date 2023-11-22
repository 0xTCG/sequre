#!/bin/bash

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;

if [[ -z "${SEQURE_CODON_PATH}" ]]; then
    echo "Error! SEQURE_CODON_PATH env variable not set" >&2
    return
fi

if [ ! -d "${SEQURE_CODON_PATH}/install" ]
then
    echo "Codon not installed at ${SEQURE_CODON_PATH}" >&2
    return
fi

echo "Codon path: $SEQURE_CODON_PATH"
echo "Compiling $2 in $1 mode ..."
$SEQURE_CODON_PATH/install/bin/codon run -plugin sequre -plugin seq ${*:1}

echo "Cleaning up sockets ..."
find  . -name 'sock.*' -exec rm {} \;
