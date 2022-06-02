#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=cifarMMDnc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --output=cifar10_mmd_nocontra_run4%A.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
cd $HOME/osr/adapt/
# Activate your environment
source activate cf_osr 

# Run your code

for SEED in 777 1234 2731; do
    srun python -u main.py --data_name CIFAR10 --epochs 100 --seed_sampler $SEED --beta_z 6 --ood_bound 0.21 --mmd_loss --eta 1 --batch_size 64 | tee -a "$SLURM_SUBMIT_DIR"/output/CIFAR10_MMD_$SEED.txt;
done 
