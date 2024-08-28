
#!/bin/bash
#SBATCH --account=research
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH -o $HOME/Morningstar_templates/slurm_scripts/out/mnist_output.out
#SBATCH -e $HOME/Morningstar_templates/slurm_scripts/err/mnist_error.err

# Activate the virtual environment
cd $HOME/
source myenv/bin/activate

cd $HOME/Morningstar_templates/slurm_scripts

# Run the Python training script
python mnist_training.py

# Run the Python testing script
python mnist_test_model.py
