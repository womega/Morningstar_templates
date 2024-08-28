# MNIST Training and Testing on Morning Star Supercomputer


This repository contains a Jupyter notebook for training and testing a RandomForest model on the IRIS dataset [1] as well as scripts for training and testing a neural network model on the MNIST [2] dataset using PyTorch. The training script supports resuming from checkpoints, and the testing script evaluates the trained model and saves the results to a file. Download these files and upload them to MorningStar in a folder named `Morningstar_templates`. Make sure to create the `out` and `err` folders in the `slurm_scripts` folder to save the log messages from SLURM.


## Repository Structure

The repository contains the following files:
- `iris_dataset_analysis.ipynb`: Jupyter notebook to train and test a Random Forest model on the IRIS dataset.
- `mnist_training.py`: Script to train the neural network on the MNIST dataset with checkpointing and the ability to skip training if it is already complete.
- `mnist_test_model.py`: Script to test the trained MNIST model, evaluate its performance, and save the results to a file.
- `submit_mnist_job.sh`: Bash script to submit the training job to the Morning Star supercomputer using SLURM.

## Prerequisites

- Python 3.10.12 or higher
- Virtual environment setup (recommended)

## Running the Code

1. Training the Model

To train the model, use the `mnist_training.py` script. This script will automatically resume from the last checkpoint if one exists and skip training if the model has already been trained for the specified number of epochs.

### Running Locally:

Activate your virtual environment and run the script using the following command:


```
python mnist_training.py
```

### Submitting the Job to SLURM:

You can submit the training job using the `submit_mnist_job.sh` script located in the `slurm_scripts` folder. Ensure that you have set up your environment and SLURM account correctly. In the `.sh` file, replace `myenv` in line 11 (`source myenv/bin/activate`) with the name of the virtual environment you created. After making the necessary corrections, submit the job using the following command:



```
sbatch submit_mnist_job.sh
```


2. Testing the Model

Once the training is complete, you can test the model using the `mnist_test_model.py` script. This script will load the final model from the `model_checkpoints` directory and evaluate its accuracy on the test dataset.


#### Running Locally:

```
python mnist_test_model.py
```

The test results (accuracy and time) will be saved to test_results.txt.

### Submitting the Job to SLURM:

If your testing script requires more than 24 hours, create a new bash script (e.g., `mnist_job_test.sh`), correct the output and error log directories, and update the Python script to be executed. Submit the job with `sbatch`:


```
sbatch submit_mnist_job.sh
```

## Directory Structure

The `model_checkpoints` directory will be created automatically to store the model checkpoints and the final model.

- `model_checkpoints/`: Contains model checkpoints and the final trained model.
- `slurm_scripts/out`: Directory for SLURM output logs.
- `slurm_scripts/err`: Directory for SLURM error logs.

## Notes

- Ensure that the `model_checkpoints` directory is present before running the testing script, as the final model will be loaded from there.
- Modify the number of epochs in the training script (`epochs` variable) to suit your needs.
- If any of the required directories (`model_checkpoints`, `out`, `err`) are missing, create them manually or add a directory creation step in your scripts to avoid errors.

## License

This project is licensed under the MIT License.

## References

[1] R. A. Fisher. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76

[2] Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6), 141–142
