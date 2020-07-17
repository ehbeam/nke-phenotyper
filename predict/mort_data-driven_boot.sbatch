#!/bin/bash

#SBATCH --job-name=mort_dd_boot
#SBATCH --output=logs/mort_dd_boot.%j.out
#SBATCH --error=logs/mort_dd_boot.%j.err
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load anaconda/3
srun python3 mort_data-driven_boot.py
