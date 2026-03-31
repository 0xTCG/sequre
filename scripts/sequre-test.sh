#!/usr/bin/env bash

cd /workspaces/sequre && \
bash -c "CODON_DEBUG=lt $HOME/.codon/bin/codon run --disable-opt='core-pythonic-list-addition-opt' -plugin sequre \"$1\" --skip-mhe-setup 2>&1"
