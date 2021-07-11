#!/usr/bin/env bash

echo "Copying DSL to Seq ..."
cp -ru dsl/* seq/stdlib/sequre/
echo "Running playground ..."
GC_INITIAL_HEAP_SIZE=838612000 seq/build/seqc run -release client.seq test --test-run
