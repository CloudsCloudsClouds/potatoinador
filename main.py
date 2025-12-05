import time

import cv2
import numpy as np
import onnxruntime as ort
import serial
import torch
from PIL import Image
from torchvision import transforms

# --- Configuration ---

# -- Model and Image Settings --
MODEL_PATH = "potato_rock_classifier.onnx"
IMG_SIZE = (224, 224)
# IMPORTANT: The class names must match the order from your 3-class training.
# Check the `Class to index mapping` output from `train.py`.
# Default order from ImageFolder is alphabetical:
CLASS_NAMES = ["neither", "potato", "rock"]
NUM_CLASSES = len(CLASS_NAMES)

# -- Arduino Serial Communication Settings --
# Your Arduino's serial port on Linux
SERIAL_PORT = "/dev/ttyACM1"
BAUD_RATE = 9600


# --- Serial Connection Setup ---\
ser = None
try:
    print(
        f"Attempting to connect to Arduino on port {SERIAL_PORT} at {BAUD_RATE} baud..."
    )
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    # Wait for the connection to establish, especially for boards that reset on connection
    time.sleep(2)
    print("Arduino connected successfully.")
except serial.SerialException as e:
    print(f"Warning: Could not open serial port {SERIAL_PORT}. {e}")
    print("The program will run without Arduino communication.")
    ser = None


# --- Model Setup ---
print("\nLoading the Potato Rock Detector ONNX model...")

try:
    # Check for GPU provider
    providers = ort.get_available_providers()
    provider = (
        "CUDAExecutionProvider"
        if "CUDAExecutionProvider" in providers
        else "CPUExecutionProvider"
    )
    print(f"Using ONNX Runtime provider: {provider}")
    ort_session = ort.InferenceSession(MODEL_PATH, providers=[provider])
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    print(
        f"Please ensure the model file '{MODEL_PATH}' exists and is a valid ONNX model."
    )
    print("You may need to run export_onnx.py first.")
    exit()

# Get the input name for the model
input_name = ort_session.get_inputs()[0].name

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

# State variable to track the last detected class to avoid spamming the serial port
last_sent_class = None

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
    input_batch = input_tensor.unsqueeze(0).numpy()

    # 2. Perform inference with ONNX Runtime
    ort_outs = ort_session.run(None, {input_name: input_batch})
    output = ort_outs[0]

    # 3. Get prediction and confidence from the output logits
    # The output is a numpy array. We apply softmax to get probabilities.
    exp_scores = np.exp(output[0] - np.max(output[0]))
    probabilities = exp_scores / np.sum(exp_scores, axis=0)

    cat_id = np.argmax(probabilities)
    class_name = CLASS_NAMES[cat_id]
    confidence_score = probabilities[cat_id]

    # 4. Implement State-Change Serial Communication
    if ser:
        # Send 'P' for potato if the state has changed to potato
        if class_name == "potato" and last_sent_class != "potato":
            ser.write(b"P")
            print("Sent 'P' to Arduino for POTATO")
            last_sent_class = "potato"
        # Send 'R' for rock if the state has changed to rock
        elif class_name == "rock" and last_sent_class != "rock":
            ser.write(b"R")
            print("Sent 'R' to Arduino for ROCK")
            last_sent_class = "rock"
        # If it's neither, reset the state so the next potato/rock is detected
        elif class_name == "neither":
            last_sent_class = "neither"  # Clear the last sent state

    # 5. Display the result on the frame
    display_text = f"{class_name.upper()}: {confidence_score:.2f}"
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
if ser:
    ser.close()
    print("Serial port closed.")
