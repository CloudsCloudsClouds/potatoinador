import torch
import torch.nn as nn
from torchvision import models

# This script is just in case the model is already trained in a pth, but doesn't have a onnx
# It's just legacy, all train.py now also export a onnx
CLASS_NAMES = ["neither", "potato", "rock"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "potato_rock_classifier.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = models.mobilenet_v3_large(weights=None)
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(256, NUM_CLASSES),
)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'.")
    print("Please run train.py to create the model file first.")
    exit()
model = model.to(device)
model.eval()

torch.onnx.export(
    model,
    (torch.rand(1, 3, 224, 224).to(device)),
    "potato_rock_classifier.onnx",
    export_params=True,
    opset_version=18,
)
