import os
import cv2
import mediapipe as mp
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw
import torch
from collections import deque
import serial
import time

# Disable the NSFW safety checker
def disable_safety_checker(pipe):
    """Completely remove the NSFW safety checker."""
    pipe.safety_checker = None

# Arduino setup
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=0.1)  # Update port as needed

# Global variables for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = None, None
smooth_points = deque(maxlen=5)  # Buffer for smoothing positions

# Smooth positions using a moving average filter
def get_smoothed_position(new_x, new_y):
    """Smooth the finger position using a moving average filter."""
    smooth_points.append((new_x, new_y))  # Add the new position to the buffer
    avg_x = int(np.mean([p[0] for p in smooth_points]))  # Average x-coordinates
    avg_y = int(np.mean([p[1] for p in smooth_points]))  # Average y-coordinates
    return avg_x, avg_y

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load ControlNet and Stable Diffusion
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# Completely disable the NSFW safety checker
disable_safety_checker(pipe)

# Send commands to Arduino
def send_to_arduino(command):
    """Send a command to the Arduino."""
    arduino.write(f"{command}\n".encode())
    time.sleep(0.1)

# Check if the canvas is blank
def is_canvas_blank(canvas):
    """Check if the canvas is completely black (empty)."""
    return not np.any(canvas)

# Preprocess the canvas for Stable Diffusion
def preprocess_canvas(canvas, output_size=(256, 256)):
    """Convert the canvas to grayscale, normalize, and resize for Stable Diffusion."""
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    binary_canvas = gray_canvas  # Simplified preprocessing
    resized_canvas = cv2.resize(binary_canvas, output_size, interpolation=cv2.INTER_AREA)
    pil_image = Image.fromarray(resized_canvas).convert("RGB")
    return pil_image

# Generate an image using Stable Diffusion with ControlNet
def generate_image_with_canvas(prompt, canvas, output_path="generated_image.png"):
    input_image = preprocess_canvas(canvas)

    # Save the preprocessed input for debugging
    debug_path = "debug_preprocessed_canvas.png"
    input_image.save(debug_path)
    print(f"Preprocessed canvas saved as {debug_path}. Please inspect the file.")

    # Generate the image
    try:
        send_to_arduino("green")  # Indicate art generation on Arduino
        result = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=20,
            height=256,
            width=256
        ).images[0]

        # Save the output image
        result.save(output_path)
        print(f"Generated image saved to {output_path}")

        # Automatically open the generated image
        open_image(output_path)

    except Exception as e:
        print(f"Error during generation: {e}")

    finally:
        send_to_arduino("off")  # Turn off LED after completion

def open_image(image_path):
    """Automatically open the generated image."""
    try:
        if os.name == 'nt':
            os.startfile(image_path)
        elif os.name == 'posix':
            os.system(f'xdg-open "{image_path}"')
    except Exception as e:
        print(f"Failed to open image: {e}")

def run_application():
    global canvas, prev_x, prev_y
    cap = cv2.VideoCapture(0)

    while True:
        # Wait for the Arduino button to start
        if arduino.in_waiting > 0:
            message = arduino.readline().decode().strip()
            if message == "start_canvas":
                print("Canvas started!")
                break

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            send_to_arduino("red")  # Indicate tracking on Arduino
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                height, width, _ = frame.shape
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                smoothed_x, smoothed_y = get_smoothed_position(x, y)
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (smoothed_x, smoothed_y), (255, 255, 255), 2)
                prev_x, prev_y = smoothed_x, smoothed_y
        else:
            send_to_arduino("off")  # Turn off LED when not tracking
            prev_x, prev_y = None, None

        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow("Finger Drawing", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        elif key == ord('s'):
            if is_canvas_blank(canvas):
                print("Canvas is blank!")
                continue
            prompt = "A vibrant colorful abstract painting with glowing patterns and bright hues"
            generate_image_with_canvas(prompt, canvas)

    cap.release()
    cv2.destroyAllWindows()

run_application()
