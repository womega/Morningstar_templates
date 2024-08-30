#!/bin/bash
#SBATCH --account=research
#SBATCH --gpus=ampere:1
#SBATCH --time=02:00:00
#SBATCH -o $HOME/Morningstar_templates/slurm_scripts/out/mnist_output.out
#SBATCH -e $HOME/Morningstar_templates/slurm_scripts/err/mnist_error.err

echo "Activating virtual environment for job: $SLURM_JOB_ID"
# Activate the virtual environment
cd $HOME/
source myenv/bin/activate

echo "Virtual environment activated!"
cd $HOME/Morningstar_templates/

echo "Launching training script!"

# Run the Python training script
python mnist_training.py

echo "Training complete, launching testing script!"

# Run the Python testing script
python mnist_test_model.py
