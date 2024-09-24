#!/bin/bash
#SBATCH --account=small
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/uottawa.o.univ/%u/Morningstar_templates/slurm_scripts/out/kdd_output-%j.out
#SBATCH --error=/home/uottawa.o.univ/%u/Morningstar_templates/slurm_scripts/err/kdd_error-%j.err

# Activate the virtual environment
cd $HOME/
source myenv/bin/activate

cd $HOME/Morningstar_templates/

# Run the Python training script for KDD
python kdd_training.py

# Run the Python testing script for KDD
python kdd_testing.py
