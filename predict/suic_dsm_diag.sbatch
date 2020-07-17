#!/bin/bash

#SBATCH --job-name=suic_diag
#SBATCH --output=logs/suic_diag.%j.out
#SBATCH --error=logs/suic_diag.%j.err
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load anaconda/3
srun python3 suic_dsm_diag.py
