#!/usr/bin/env sh
  
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
job_name=BCChat
srun --partition=OpenDialogLab_S2 --gpus-per-task=1 --mpi=pmi2 --ntasks-per-node=1 --job-name=$job_name --kill-on-bad-exit=1 python api.py server --api=BaiChuanAPI --api_version=chinese-chat --wrapper=Flask 2>&1|tee log/$job_name-$now.log &
# srun --partition=OpenDialogLab_S2 --mpi=pmi2 --ntasks-per-node=1 --job-name=$job_name --kill-on-bad-exit=1 python api.py server --api=GPT4API --wrapper=Flask 2>&1|tee log/$job_name-$now.log &