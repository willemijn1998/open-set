#!/bin/bash

#SBATCH --partition=gpu_titanrtx
#SBATCH --job-name=cifarSupCon
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --time=20:00:00
#SBATCH --output=cifar_supcon_nu_%A.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
cd $HOME/osr/adapt/
# Activate your environment
source activate cf_osr 

# Run your code

for THETA in 0.1 0.2 0.3 0.4; do
    srun python -u main.py --data_name CIFAR10 --epochs 100 --seed_sampler 777 --beta_z 6 --ood_bound 0.21 --batch_size 64 --supcon_loss --theta $THETA| tee -a "$SLURM_SUBMIT_DIR"/output/CIFAR10_MMD2_$THETA.txt;
done 