import re
import os
import numpy as np
import torch
import random
import tensorflow as tf

def extract_number(filename):
    """Extract numerical parts from the filename."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


def set_seed(seed=42069):
    # np and random
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Tensorflow
    tf.random.set_seed(seed)

    # hash
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

