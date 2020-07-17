#!/bin/sh
for FILE in mort_*_boot.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done

for FILE in mort_*_null.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
