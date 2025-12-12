#!/usr/bin/env python3
# Main. But run in a raspi
# HDMI output not expected
# We:
#   - load the onnx model
#   - find a camera
#   - get the camera working with the model
#   - run the model on the camera frames
#   - move a servo 90 in GPIO 18 either to left or to the right, or to the center
#       - If potato: to the left
#       - If rock: to the right
#       - If either: to the center
#
# It requires the requirements_rpi.txt
# It also needs the onnx model. Obviously.

import select
import sys
import termios
import time
import tty

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

# Try to import RPi.GPIO, but allow the script to run without it for testing
try:
    import RPi.GPIO as GPIO

    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Servo control will be simulated.")
    GPIO_AVAILABLE = False

# -- Model and Image Settings --
MODEL_PATH = "potato_rock_classifier.onnx"
IMG_SIZE = (224, 224)
# Class names must match the order from training
CLASS_NAMES = ["neither", "potato", "rock"]

# -- Servo Settings --
SERVO_PIN = 18  # GPIO pin for servo control (Physical pin 12)
SERVO_FREQUENCY = 50  # 50Hz for standard servos

# Servo angles for each classification
SERVO_ANGLE_POTATO = 0  # Right position for potato
SERVO_ANGLE_ROCK = 180  # Left position for rock
SERVO_ANGLE_NEUTRAL = 90  # Center/neutral position

# -- Detection Settings --
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to trigger servo
DEBOUNCE_TIME = 1.0  # Seconds to wait before next servo action
HOLD_TIME = 0.5  # Seconds to hold servo position

# -- Camera Settings --
CAMERA_INDEX = 0  # Default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# --- Servo Control Functions ---


def setup_servo():
    """Initialize GPIO and servo motor."""
    if not GPIO_AVAILABLE:
        print("Servo setup skipped (GPIO not available)")
        return None

    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQUENCY)
        pwm.start(0)
        print(f"Servo initialized on GPIO {SERVO_PIN}")
        return pwm
    except Exception as e:
        print(f"Error setting up servo: {e}")
        return None


def angle_to_duty_cycle(angle):
    """
    Convert servo angle (0-180) to PWM duty cycle.
    Standard servo: 1ms = 0°, 1.5ms = 90°, 2ms = 180°
    At 50Hz: 1ms = 5% duty, 2ms = 10% duty
    """
    # Clamp angle between 0 and 180
    angle = max(0, min(180, angle))
    # Convert to duty cycle (5% to 10%)
    duty_cycle = 2.5 + (angle / 180.0) * 10.0
    return duty_cycle


def move_servo(pwm, angle):
    """Move servo to specified angle."""
    if pwm is None:
        print(f"[SIMULATED] Servo would move to {angle}°")
        return

    try:
        duty = angle_to_duty_cycle(angle)
        pwm.ChangeDutyCycle(duty)
        print(f"Servo moved to {angle}°")
    except Exception as e:
        print(f"Error moving servo: {e}")


def cleanup_servo(pwm):
    """Clean up GPIO resources."""
    if GPIO_AVAILABLE and pwm:
        pwm.stop()
        GPIO.cleanup()
        print("Servo cleanup completed")


# --- Model Setup ---


def load_model():
    """Load the ONNX model for inference."""
    print("Loading Potato/Rock Classifier ONNX model...")

    try:
        # On Raspberry Pi, we'll use CPU
        providers = ["CPUExecutionProvider"]
        ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
        print("Model loaded successfully (CPU mode)")
        return ort_session
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        print(f"Please ensure '{MODEL_PATH}' exists in the current directory.")
        return None


# --- Image Preprocessing ---

# Transformations matching the training pipeline
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_frame(frame):
    """Convert OpenCV frame to model input tensor."""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Apply transformations
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).numpy()

    return input_batch


def predict(ort_session, input_batch):
    """Run inference and return class name, confidence."""
    # Get input name
    input_name = ort_session.get_inputs()[0].name

    # Run inference
    ort_outs = ort_session.run(None, {input_name: input_batch})
    output = ort_outs[0]

    # Apply softmax to get probabilities
    exp_scores = np.exp(output[0] - np.max(output[0]))
    probabilities = exp_scores / np.sum(exp_scores)

    # Get prediction
    class_id = np.argmax(probabilities)
    class_name = CLASS_NAMES[class_id]
    confidence = probabilities[class_id]

    return class_name, confidence


# --- MAIN LOOP ---
def main():
    # Load model first
    ort_session = load_model()
    if ort_session is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)

    # Setup servo
    pwm = setup_servo()

    # Setup camera
    print(f"\nAttempting to open camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("=" * 50)
        print("ERROR: Could not open camera!")
        print("Please connect a camera and try again.")
        print("=" * 50)
        cleanup_servo(pwm)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"Camera opened successfully ({FRAME_WIDTH}x{FRAME_HEIGHT})")

    # Move servo to neutral position at start
    move_servo(pwm, SERVO_ANGLE_NEUTRAL)
    time.sleep(0.3)

    last_action_time = 0

    # Save terminal settings for restoring later
    old_settings = termios.tcgetattr(sys.stdin)

    print("\n" + "=" * 50)
    print("POTATO-ROCK CLASSIFIER RUNNING")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("  c - Force center position")
    print("  p - Test potato position (left)")
    print("  r - Test rock position (right)")
    print("=" * 50 + "\n")

    try:
        # Set terminal to non-blocking mode
        tty.setcbreak(sys.stdin.fileno())

        while True:
            # Check for keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()

                if key == "q":
                    print("\nQuit requested. Shutting down...")
                    break
                elif key == "c":
                    print("[MANUAL] Moving to center")
                    move_servo(pwm, SERVO_ANGLE_NEUTRAL)
                elif key == "p":
                    print("[MANUAL] Testing potato position (left)")
                    move_servo(pwm, SERVO_ANGLE_POTATO)
                elif key == "r":
                    print("[MANUAL] Testing rock position (right)")
                    move_servo(pwm, SERVO_ANGLE_ROCK)

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame from camera")
                time.sleep(0.1)
                continue

            # Preprocess and predict
            input_batch = preprocess_frame(frame)
            class_name, confidence = predict(ort_session, input_batch)

            current_time = time.time()

            # Display current prediction
            print(
                f"\rDetected: {class_name:8s} ({confidence:5.1%})", end="", flush=True
            )

            # Check debounce and confidence threshold before acting
            if current_time - last_action_time >= DEBOUNCE_TIME:
                if confidence >= CONFIDENCE_THRESHOLD:
                    if class_name == "potato":
                        print(f"\n>>> POTATO! Moving servo LEFT")
                        move_servo(pwm, SERVO_ANGLE_POTATO)
                        time.sleep(HOLD_TIME)
                        move_servo(pwm, SERVO_ANGLE_NEUTRAL)
                        last_action_time = current_time

                    elif class_name == "rock":
                        print(f"\n>>> ROCK! Moving servo RIGHT")
                        move_servo(pwm, SERVO_ANGLE_ROCK)
                        time.sleep(HOLD_TIME)
                        move_servo(pwm, SERVO_ANGLE_NEUTRAL)
                        last_action_time = current_time

                    # "neither" - just stay at center, no action needed

            # Small delay to prevent CPU overload
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl+C")

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Cleanup resources
        print("\nCleaning up...")
        cap.release()
        move_servo(pwm, SERVO_ANGLE_NEUTRAL)  # Return to center
        time.sleep(0.3)
        cleanup_servo(pwm)
        print("Goodbye!")


if __name__ == "__main__":
    main()
