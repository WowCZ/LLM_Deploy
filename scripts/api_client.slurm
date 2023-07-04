#!/usr/bin/env sh
  
mkdir -p log
job_name=zhihu
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=OpenDialogLab_S2 --mpi=pmi2 --ntasks-per-node=1 --job-name=$job_name --kill-on-bad-exit=1 python api.py client --model_name gpt4 --batch_size 1 --max_length 32000 --evaluation_tasks zhihu --inference_path resource/zhihu --dump_type incremental 2>&1|tee log/$job_name-client.log &