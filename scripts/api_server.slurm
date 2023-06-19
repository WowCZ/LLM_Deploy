#!/usr/bin/env sh
  
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
# srun --partition=OpenDialogLab_S2 --gpus-per-task=1 --mpi=pmi2 --ntasks-per-node=1 --job-name=Bloomz --kill-on-bad-exit=1 python api.py server --api=ChatGLMAPI --api_version=7b1 --wrapper=Flask 2>&1|tee log/download-$now.log &
srun --partition=OpenDialogLab_S2 --mpi=pmi2 --ntasks-per-node=1 --job-name=SenseChat --kill-on-bad-exit=1 python api.py server --api=SenseChatAPI --wrapper=Flask 2>&1|tee log/download-$now.log &