#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=cifarMMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=15:00:00
#SBATCH --mem=64G
#SBATCH --output=cifar_mmd_etas%A.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
cd $HOME/osr/adapt/
# Activate your environment
source activate cf_osr 

# Run your code

for ETA in 2 3 4; do
    srun python -u main.py --data_name CIFAR10 --epochs 100 --seed_sampler 777 --beta_z 6 --ood_bound 0.21 --nu 1 --contra_loss --mmd_loss --eta $ETA --batch_size 64 | tee -a "$SLURM_SUBMIT_DIR"/output/CIFAR10_MMD_777_ETA$ETA.txt;
done 
