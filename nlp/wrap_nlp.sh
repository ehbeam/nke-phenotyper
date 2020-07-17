#!/bin/sh
for FILE in nlp_batch*.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
