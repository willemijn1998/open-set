#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=cifarAJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=14:00:00
#SBATCH --mem=32000M
#SBATCH --output=cifar10_output_%A.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
cd $HOME/osr/adapt/
# Activate your environment
source activate cf_osr 

# Run your code
for CF in 777 1234 2731 3925 5432; do
    srun python -u main.py --data_name CIFAR10 --epochs 100 --seed_sampler $SEED --beta_z 6 --threshold_ood 25 | tee -a "$SLURM_SUBMIT_DIR"/output/CIFAR10_$SEED;
done 