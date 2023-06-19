#!/bin/sh
#SBATCH -J vicuna-13b
#SBATCH --error=log/slurm-%A_%a.err
#SBATCH --output=log/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --partition=OpenDialogLab_S2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

python api.py server --api=VicunaAPI --api_version '13b' --wrapper=Flask