#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=DebugJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=0:10:00
#SBATCH --mem=32000M
#SBATCH --output=debug_output_%j_%a.out

module purge
module load 2021
module load Anaconda3/2021.05
module load R/4.1.0-foss-2021a

# Your job starts in the directory where you call sbatch
cd $HOME/osr/adapt/
# Activate your environment
source activate cf_osr 

conda install pandas
