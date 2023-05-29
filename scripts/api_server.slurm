#!/usr/bin/env sh
  
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=OpenDialogLab_S2 --gpus-per-task=1 --mpi=pmi2 --ntasks-per-node=1 --job-name=ChatGLM --kill-on-bad-exit=1 python api.py server --api=ChatGLMAPI --wrapper=Flask 2>&1|tee log/download-$now.log &