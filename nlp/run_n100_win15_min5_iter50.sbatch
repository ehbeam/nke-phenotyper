#!/bin/bash

#SBATCH --job-name=glove
#SBATCH --output=logs/glove.%j.out
#SBATCH --error=logs/glove.%j.err
#SBATCH --time=07-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

srun sh run_n100_win15_min5_iter50.sh
