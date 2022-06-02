#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=mnistcs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH -o /home/hoopwd/osr/adapt/jobs/output/mnist_output_%j_%a.out

module purge
module load 2021
module load Anaconda3/2021.05
module load R/4.1.0-foss-2021a

# Your job starts in the directory where you call sbatch
cd $HOME/osr/adapt/
# Activate your environment
source activate cf_osr 


# for SEED in 777 1234 2731 3925 5432; do 
#     srun python -u main.py --epochs 50 --seed_sampler $SEED --threshold_ood 25 --nu 20 --temperature 10| tee -a "$SLURM_SUBMIT_DIR"/output/MNIST_$SEED;
# done 

srun python -u main.py --epochs 50 --seed_sampler 777 | tee -a "$SLURM_SUBMIT_DIR"/output/MNIST/adapt_1_777;

