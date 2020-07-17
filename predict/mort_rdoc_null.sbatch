#!/bin/bash

#SBATCH --job-name=mort_rdoc_null
#SBATCH --output=logs/mort_rdoc_null.%j.out
#SBATCH --error=logs/mort_rdoc_null.%j.err
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load anaconda/3
srun python3 mort_rdoc_null.py
