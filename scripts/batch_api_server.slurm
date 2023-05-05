#!/bin/sh
#SBATCH -J Bloom-7B1
#SBATCH --error=log/slurm-%A_%a.err
#SBATCH --output=log/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=OpenDialogLab_S2
#SBATCH --ntasks-per-node=1

python api_server.py --api=BloomAPI --server=Flask