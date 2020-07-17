#!/bin/bash

#SBATCH --job-name=suic_dd
#SBATCH --output=logs/suic_dd.%j.out
#SBATCH --error=logs/suic_dd.%j.err
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load anaconda/3
srun python3 suic_data-driven.py
