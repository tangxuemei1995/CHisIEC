#! /bin/bash

#SBATCH --partition=IAI_SLURM_A100
#SBATCH --job-name=llama
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=8gpu
#SBATCH --cpus-per-task=10
#SBATCH --time 3:00:00
python3 test.py