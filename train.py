import os

import kagglehub
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

BATCH_SIZE = 32
IMG_SIZE = (180, 180)

# So far so good...
# Let's get the latest datasets.
rock_path = kagglehub.dataset_download("neelgajare/rocks-dataset")
print("Path to rocks files:", rock_path)
potato_path = kagglehub.dataset_download("misrakahmed/vegetable-image-dataset")
print("Path to vegetables files:", potato_path)

# Here the fun thing. Let's get just the potatoes.

data_dir = os.path.join(potato_path, "Vegetable Images", "train", "Potato")
print(data_dir)

for root, dirs, files in os.walk(data_dir):
    for file in files:
        print(f"Directory: {root}")
        print(f"Subdirectories: {dirs}")
        print(f"Files: {files[:10]}...")

# potato_train_ds = tf.keras.utils.image_dataset_from_directory
# It was easy in tensorflow...
# I wonder what would I do now? I'm using pytorch

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

potato_train_ds = datasets.ImageFolder(root=data_dir, transform=transform)

potato_loader = DataLoader(
    potato_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

print(f"Number of potato images: {len(potato_train_ds)}")
print(f"Classes found: {potato_train_ds.classes}")
print(f"Class to index mapping: {potato_train_ds.class_to_idx}")
