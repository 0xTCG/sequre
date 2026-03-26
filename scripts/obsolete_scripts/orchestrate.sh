#!/usr/bin/env bash

if [[ -z "${SEQURE_CP_IPS}" ]]; then
    echo "Error! SEQURE_CP_IPS env variable not set" >&2
    return
fi

SEQURE_CP_IPS_ARR=(${SEQURE_CP_IPS//,/ })
SEQURE_CP_IPS_ARR_SIZE=${#SEQURE_CP_IPS_ARR[@]}
SEQURE_CP_IPS_ARR_SIZE=$(($SEQURE_CP_IPS_ARR_SIZE - 1))

parallel -S ${SEQURE_CP_IPS//,/ -S } "podman run -e 'CODON_DEBUG=lt' -e 'SEQURE_CP_IPS=${SEQURE_CP_IPS}' --privileged hsmile/sheqi:latest scripts/run.sh ${*:1} {}" ::: $(seq -s" " 0 $SEQURE_CP_IPS_ARR_SIZE)
