#!/bin/sh

echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
echo 'alias python=python3' >> ~/.bashrc
source ~/.bashrc

echo "replace github.com/hhcho/sfgwas-private => /sfgwas-private" >> ./go.mod

START=0
END=2
LOG_PREFIX=stdout/runtime
TRY=1

TESTNAME=TestMatMultProtocol

for (( i = $START; i <= $END; i++ )) 
do
  echo "Running PID=$i"
  CMD="PID=$i go test -run ${TESTNAME} -timeout 48h | tee /dev/tty > ${LOG_PREFIX}_party${i}_try${TRY}.txt"
  if [ $i = $END ]; then
    eval $CMD
  else
    eval $CMD &
  fi
done

toml set --toml-path='config/matmult/configGlobal.toml' --to-bool 'ATA' true
toml set --toml-path='config/matmult/configGlobal.toml' --to-bool 'ATA' true

for (( i = $START; i <= $END; i++ )) 
do
  echo "Running PID=$i"
  CMD="PID=$i go test -run ${TESTNAME} -timeout 48h | tee /dev/tty > ${LOG_PREFIX}_party${i}_try${TRY}.txt"
  if [ $i = $END ]; then
    eval $CMD
  else
    eval $CMD &
  fi
done

toml set --toml-path='config/king/configLocal.Party1.toml' 'simple_data_path' '/data/king_sample.txt'
toml set --toml-path='config/king/configLocal.Party2.toml' 'simple_data_path' '/data/king_sample.txt'
toml set --toml-path='config/matmult/configLocal.Party1.toml' 'simple_data_path' '/data/king_sample.txt'
toml set --toml-path='config/matmult/configLocal.Party2.toml' 'simple_data_path' '/data/king_sample.txt'

TESTNAME=TestKingProtocol

toml set --toml-path='config/king/configGlobal.toml' --to-bool 'minimum' true
toml set --toml-path='config/king/configGlobal.toml' --to-bool 'sqrt' false

for (( i = $START; i <= $END; i++ )) 
do
  echo "Running PID=$i"
  CMD="PID=$i go test -run ${TESTNAME} -timeout 48h | tee /dev/tty > ${LOG_PREFIX}_party${i}_try${TRY}.txt"
  if [ $i = $END ]; then
    eval $CMD
  else
    eval $CMD &
  fi
done

toml set --toml-path='config/king/configLocal.Party1.toml' --to-int 'number_of_columns' 8192
toml set --toml-path='config/king/configLocal.Party2.toml' --to-int 'number_of_columns' 8192
toml set --toml-path='config/king/configGlobal.toml' --to-bool 'minimum' false
toml set --toml-path='config/king/configGlobal.toml' --to-bool 'sqrt' true

for (( i = $START; i <= $END; i++ )) 
do
  echo "Running PID=$i"
  CMD="PID=$i go test -run ${TESTNAME} -timeout 48h | tee /dev/tty > ${LOG_PREFIX}_party${i}_try${TRY}.txt"
  if [ $i = $END ]; then
    eval $CMD
  else
    eval $CMD &
  fi
done
