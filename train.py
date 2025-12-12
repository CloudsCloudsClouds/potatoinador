import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_EPOCHS = 10  # Change this thing if things are bad
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "potato_rock_classifier.pth"

# Data loading
print("Setting up data transformations and loaders...")

# Define the transformations for the input images
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # Normalization values are standard for models pre-trained on ImageNet
        # I have no idea why use these values, but it's also in the docs and the IA uses it so...
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the dataset from the organized folder structure
# After running prep_ds.py
comb_ds_path = "data/combined_dataset"
try:
    comb_ds = datasets.ImageFolder(comb_ds_path, transform=transform)
except FileNotFoundError:
    print(f"Error: Dataset not found at '{comb_ds_path}'.")
    print("Please make sure you have run the dataset preparation script first.")
    exit()


# Create a DataLoader to iterate over the dataset in batches
comb_loader = DataLoader(
    comb_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,  # Just put ~~4~~ 2, it's good enough
)

print(f"Number of images in combined dataset: {len(comb_ds)}")
print(f"Classes found: {comb_ds.classes}")
print(f"Class to index mapping: {comb_ds.class_to_idx}")
num_classes = len(comb_ds.classes)


# Transform learning!
print("\nSetting up the model...")

# Set the device to a GPU if available, otherwise use the CPU
# It's cool. Prob shouldn't be used for raspberry pi.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained MobileNetV3 model
# Using the recommended 'weights' parameter instead of 'pretrained'
model = models.mobilenet_v3_large(
    weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
)

# Freeze all the parameters in the pre-trained model
# Also standar for tensorflow version
# This detaches the head that works well and we attach a body later
for param in model.parameters():
    param.requires_grad = False

# Rebuild the classifier head for our specific task.
# This is a robust way to avoid type-checking issues with in-place modification.
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),  # Thinking layer.
    nn.Hardswish(),  # Activator. Adds non linearity. It's a bit weird to explain, but basically "if it's interesting, amplify it, if not, squash it."
    nn.Dropout(
        p=0.2, inplace=True
    ),  # Randomly drop some neurons to prevent overfitting
    nn.Linear(256, num_classes),  # Final decision layer.
    # 256 is good enough here. It's a 3 class clasifier!
)

# Move the model to the selected device
model = model.to(device)

# Optimizer and loss function
print("Defining loss function and optimizer...")
criterion = nn.CrossEntropyLoss()

# Just touch the outer
# If I were to optim the whole model, it would look like
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# But training the body is the important part. The head is good enough.
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)


# --- Training Loop ---
print(f"\nStarting training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(comb_loader):
        # Move data to the selected device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if (i + 1) % 20 == 0:  # Print every 20 batches
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(comb_loader)}], Loss: {running_loss / 20:.4f}"
            )
            running_loss = 0.0

print("\nFinished Training!")


# --- Save the Model ---
print(f"Saving model state to {MODEL_SAVE_PATH}...")
# We save the model's state_dict, which contains all the learned weights and parameters
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved successfully.")

# Now, to export to onnx
torch.onnx.export(
    model,
    (torch.rand(1, 3, 224, 224).to(device)),
    "potato_rock_classifier.onnx",
    export_params=True,
    opset_version=18,
)
