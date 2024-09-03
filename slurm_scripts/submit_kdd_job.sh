#!/bin/bash
#SBATCH --account=research
#SBATCH --gpus=ampere:1
#SBATCH --time=04:00:00
#SBATCH -o $HOME/Morningstar_templates/slurm_scripts/out/kdd_output.out
#SBATCH -e $HOME/Morningstar_templates/slurm_scripts/err/kdd_error.err

# Activate the virtual environment
cd $HOME/
source myenv/bin/activate

cd $HOME/Morningstar_templates/

# Run the Python training script for KDD
python kdd_training.py

# Run the Python testing script for KDD
python kdd_test_model.py
