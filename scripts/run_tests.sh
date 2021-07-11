#!/usr/bin/env bash

echo "Copying DSL to Seq ..."
cp -ru dsl/* seq/stdlib/sequre/
echo "Running tests ..."
seq/build/seqc run -release client.seq test
