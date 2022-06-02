#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=DebugAd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --output=debug_%A.out


module purge
module load 2021
module load Anaconda3/2021.05
module load R/4.1.0-foss-2021a

# Your job starts in the directory where you call sbatch
cd $HOME/osr/adapt/
# Activate your environment
source activate cf_osr 

# Run your code
for SEED in 777; do
    srun python -u main.py --data_name CIFAR10 --epochs 1 --seed_sampler $SEED --beta_z 6 --ood_bound 0.21 --nu 1 --contra_loss --batch_size 64 --supcon_loss | tee -a "$SLURM_SUBMIT_DIR"/output/CIFAR10_MMD2_$SEED.txt;
done 
