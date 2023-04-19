#!/usr/bin/env sh
  
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=OpenDialogLab_S2 --gpus-per-task=1 --mpi=pmi2 --ntasks-per-node=1 --job-name=bloom --kill-on-bad-exit=1 -w SH-IDC1-10-140-24-70 python api_server.py 2>&1|tee log/download-$now.log &