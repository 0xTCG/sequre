#!/bin/sh


echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
echo 'alias python=python3' >> ~/.bashrc
source ~/.bashrc

START=0
END=2
TESTNAME=TestMatMultProtocol
LOG_PREFIX=stdout/runtime
TRY=1
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
