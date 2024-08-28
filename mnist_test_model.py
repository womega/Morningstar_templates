import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os


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
checkpoint_dir = "model_checkpoints"
final_model_path = os.path.join(checkpoint_dir, "mnist_model_final.pth")
if os.path.exists(final_model_path):
    print(f"Loading final model from: {final_model_path}")
    model = Net().to(device)
    model.load_state_dict(torch.load(final_model_path))
else:
    raise FileNotFoundError(f"Final model not found in: {final_model_path}")

model.eval()

# Set up data transformations and loaders for the test set
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
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
results_path = "test_results.txt"
with open(results_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    f.write(f"Testing completed in: {elapsed_time:.2f} seconds\n")

print(f"Results saved to: {results_path}")
