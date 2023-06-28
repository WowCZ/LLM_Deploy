#!/bin/sh
#SBATCH -J baichuan
#SBATCH --error=log/slurm-%A_%a.err
#SBATCH --output=log/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --partition=OpenDialogLab_S2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

python api.py server --api=BaiChuanAPI --api_version 'chinese-vicuna' --wrapper=Flask