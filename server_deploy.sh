#!/bin/sh

while [ -n "$1" ]  
do  
  case "$1" in   
    --num_server)  
        echo ">>> Number of api servers:$2"
        SNUM=$2
        shift
        ;; 
    *)  
        echo "$1 is not an option"  
        ;;  
  esac  
  shift  
done

for ((i = 1 ; i <= $SNUM ; i++)); do
  sbatch scripts/batch_api_server.slurm
  sleep 1
done