#!/usr/bin/env sh
  
mkdir -p log
job_name=arena
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=OpenDialogLab_S2 --mpi=pmi2 --ntasks-per-node=1 --job-name=$job_name --kill-on-bad-exit=1 python webui.py arena --port 8888 2>&1|tee log/$job_name-webui.log &