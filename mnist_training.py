import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

# Check if GPU is available
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")


print(f"Using device: {device}")

# Create 'mnist_model_checkpoints' directory if it doesn't exist
checkpoint_dir = "mnist_model_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f"Created '{checkpoint_dir}' directory.")


# Define the neural network model
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


# Initialize the model, loss function, and optimizer
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Total number of epochs for training
epochs = 5

# Load checkpoint if available
start_epoch = 0
checkpoint_files = sorted(
    [f for f in os.listdir(checkpoint_dir) if not f.endswith("final.pth")]
)
if checkpoint_files:
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

# Check if training is already complete
if start_epoch >= epochs:
    print(
        f"Training is already complete for {epochs} epochs. No further training required."
    )
else:
    # Set up data transformations and loaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Start the total timer
    total_start_time = time.time()

    # Train the model

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        print(
            f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}, Time  ⏱️ : {epoch_elapsed_time:.2f} seconds"
        )

        # Save a checkpoint after each epoch
        checkpoint_path = os.path.join(
            checkpoint_dir, f"mnist_model_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    # End the total timer
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    # Print total time and GPU information
    print(f"Training completed in ⏱️ : {total_elapsed_time:.2f} seconds")
    if device.type == "cuda":
        print(f"GPU used: {torch.cuda.get_device_name(0)}")

    # Save the final model in the checkpoints directory
    final_model_path = os.path.join(checkpoint_dir, "mnist_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
