#!/usr/bin/env bash

if [[ -z "${SEQURE_CP_IPS}" ]]; then
    echo "Error! SEQURE_CP_IPS env variable not set" >&2
    return
fi

ssh location docker run --rm -t -e "CODON_DEBUG='lt'" -e "SEQURE_CP_IPS=127.0.0.1,127.0.0.1,127.0.0.1" -p 9001:9001 -p 9002:9002 hsmile/sequre scripts/run.sh -debug tests --mpc 0