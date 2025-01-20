#!/bin/bash
#SBATCH --account=small
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/home/uottawa.o.univ/%u/Morningstar_templates/slurm_scripts/out/mnist_output-%j.out
#SBATCH --error=/home/uottawa.o.univ/%u/Morningstar_templates/slurm_scripts/err/mnist_error-%j.err


# Activate the virtual environment
cd $HOME/
source myenv/bin/activate

cd $HOME/Morningstar_templates/

# Run the Python training script
python mnist_training.py

# Run the Python testing script
python mnist_testing.py
