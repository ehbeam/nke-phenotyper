#!/bin/bash

#SBATCH --job-name=meds_ap_diag
#SBATCH --output=logs/meds_ap_diag.%j.out
#SBATCH --error=logs/meds_ap_diag.%j.err
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load anaconda/3
srun python3 meds_ap_dsm_diag.py
