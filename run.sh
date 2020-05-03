#!/bin/sh

#SBATCH -o output.log
#SBATCH -p K80q 
#SBATCH --gres=gpu:1
#SBATCH -n 1

bash sample_training.sh xs
