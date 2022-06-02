#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=EvalCifar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
#SBATCH --mem=32000M
#SBATCH -o /home/hoopwd/osr/adapt/jobs/output/evalcifar_output_%j_%a.out

module purge
module load 2021
module load Anaconda3/2021.05
module load R/4.1.0-foss-2021a

# Your job starts in the directory where you call sbatch
cd $HOME/osr/adapt/
# Activate your environment
source activate cf_osr 


srun python -u main.py --data_name CIFAR10 --epochs 100 --seed_sampler 777 --threshold_ood $THRESH | tee -a "$SLURM_SUBMIT_DIR"/output/evalCIFAR_777_$THRESH;