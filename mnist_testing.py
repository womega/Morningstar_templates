import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
from utils import extract_number, set_seed

# Define the neural network model (same as in the training script)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Check if GPU is available
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")


print(f"Using device: {device}")

# Load the final model from the checkpoints directory
checkpoint_dir = "mnist_model_checkpoints"

checkpoint_files = sorted(
    [f for f in os.listdir(checkpoint_dir) if not f.startswith('.')], key=extract_number
)
if checkpoint_files:
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    final_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
    print(f"Loading the final model from epoch: {final_epoch}...")
    model = Net().to(device)
    model.load_state_dict(torch.load(latest_checkpoint, weights_only=True)["model_state_dict"])
else:
    raise FileNotFoundError("No checkpoints found. Consider training the model first.")


set_seed()
model.eval()

# Set up data transformations and loaders for the test set
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
testset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Start the timer for testing
start_time = time.time()

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# End the timer
end_time = time.time()
elapsed_time = end_time - start_time

# Calculate accuracy
accuracy = 100 * correct / total

# Print test accuracy
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Testing completed in ⏱️ : {elapsed_time:.2f} seconds")

# Save results to a file
results_path = "mnist_final_test_results.txt"
with open(results_path, "w") as f:
    f.write(f"Final Model Evaluation Results:\n\n\n")
    f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
    f.write(f"Testing completed in: {elapsed_time:.2f} seconds\n\n")

print(f"Results saved to: {results_path}")
