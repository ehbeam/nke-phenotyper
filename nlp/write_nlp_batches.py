#!/usr/bin/python

import os, shutil

n_ids = 259993
batch_size = 1000

for i in range(0, n_ids, batch_size):

    comm = "nlp.preproc_batch({}, batch_size={})".format(i, batch_size)
    pyfile = open("nlp_batch_{:06d}.py".format(i), "w+")
    pyfile.write("#!/bin/python\nimport sys\nsys.path.append('..')\nimport nlp\n{}".format(comm))
    pyfile.close()
    
    bashfile = open("nlp_batch_{:06d}.sbatch".format(i), "w+")
    lines = ["#!/bin/bash\n",
             "#SBATCH --job-name={:06d}_nlp".format(i),
             "#SBATCH --output=logs/{:06d}_nlp.%j.out".format(i),
             "#SBATCH --error=logs/{:06d}_nlp.%j.err".format(i),
             "#SBATCH --time=00-00:45:00",
             "#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task=1",
             "#SBATCH --mem-per-cpu=16G",
             "#SBATCH --mail-type=FAIL",
             "#SBATCH --mail-user=ebeam@stanford.edu\n",
             "module load anaconda/3",
             "srun python3 nlp_batch_{:06d}.py".format(i, i)]
    for line in lines:
        bashfile.write(line + "\n")
    bashfile.close()