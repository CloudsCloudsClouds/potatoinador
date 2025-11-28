# I told gemini to fix some typos and instead went to run everything -_-
# It works. The model doesn't but this does work.
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# --- Constants ---
MODEL_PATH = "potato_rock_classifier.pth"
IMG_SIZE = (224, 224)
# The class names must be in the same order as determined by ImageFolder during training
# You can verify this from the output of `train.py`: `Class to index mapping: {'potato': 0, 'rock': 1}`
CLASS_NAMES = ["potato", "rock"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Model Setup ---
print("Loading the Potato Rock Detector model...")

# Set the device to a GPU if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Initialize the model architecture (must be identical to the training script)
model = models.mobilenet_v3_large(weights=None)  # No pre-trained weights needed here

# Rebuild the classifier head to match the trained model's structure
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 1280),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(1280, NUM_CLASSES),
)

# 2. Load the saved weights (the state_dict)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'.")
    print("Please run train.py to create the model file first.")
    exit()


# 3. Set the model to evaluation mode
# This is crucial as it disables layers like Dropout for inference
model.eval()
model = model.to(device)
print("Model loaded successfully!")

# --- Image Transformations ---
# These must be the same as the transformations used during training
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# --- Webcam Inference ---
print("\nStarting webcam feed... Press 'q' to quit.")
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 1. Pre-process the frame
    # Convert the OpenCV frame (BGR) to a PIL Image (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Apply transformations and add a batch dimension
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 2. Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # 3. Get prediction and confidence
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, cat_id = torch.max(probabilities, 0)
    class_name = CLASS_NAMES[cat_id]
    confidence_score = confidence.item()

    # 4. Display the result on the frame
    display_text = f"{class_name}: {confidence_score:.2f}"
    color = (0, 255, 0)  # Green
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Show the frame in a window
    cv2.imshow("Potato Rock Detector", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
