import os

import cv2
import torch

# First, check if potato_rock_classifier.pth exists, if not, create it by running the files in order.

if not os.path.exists("potato_rock_classifier.pth"):
    os.system("uv run prep_ds.py")
    os.system("uv run train.py")

# Load the model.

model = torch.load("potato_rock_classifier.pth")

# Activate webcam
