#!/usr/bin/python3

import os

endpoints = ["meds", "meds_ad", "meds_ap", "ther", "admi", "mort", "suic"]
models = ["", "_boot", "_null"]
frameworks = ["data-driven", "rdoc", "dsm", "dsm_diag", "combo"]

abbrev = {"data-driven": "dd", "rdoc": "rdoc", "dsm": "dsm", "dsm_diag": "diag", "combo": "combo"}

for endpoint in endpoints:
    for model in models:

        fit_dir = "fits/{}{}".format(endpoint, model)
        if not os.path.exists(fit_dir):
            os.makedirs(fit_dir)

        for framework in frameworks:
        
            # Python file
            comm = "predict.run_logreg{}('{}', endpoint='{}')".format(model, framework, endpoint)
            pyfile = open("{}_{}{}.py".format(endpoint, framework, model), "w+")
            pyfile.write("#!/bin/python\n\nimport sys\nsys.path.append('..')\nimport predict\n\n{}".format(comm))
            pyfile.close()
            
            # Sbatch file
            bashfile = open("{}_{}{}.sbatch".format(endpoint, framework, model), "w+")
            lines = ["#!/bin/bash\n",
                     "#SBATCH --job-name={}_{}{}".format(endpoint, abbrev[framework], model),
                     "#SBATCH --output=logs/{}_{}{}.%j.out".format(endpoint, abbrev[framework], model),
                     "#SBATCH --error=logs/{}_{}{}.%j.err".format(endpoint, abbrev[framework], model),
                     "#SBATCH --time=01-00:00:00",
                     "#SBATCH --ntasks=1",
                     "#SBATCH --cpus-per-task=1",
                     "#SBATCH --mem-per-cpu=16G",
                     "#SBATCH --mail-type=FAIL",
                     "#SBATCH --mail-user=ebeam@stanford.edu\n",
                     "module load anaconda/3",
                     "srun python3 {}_{}{}.py".format(endpoint, framework, model)]

            for line in lines:
                bashfile.write(line + "\n")
            bashfile.close()
