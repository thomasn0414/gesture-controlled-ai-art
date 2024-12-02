# Gesture-Controlled AI Art Generator

This project allows users to create abstract AI-generated artwork by controlling a virtual canvas with their hand movements. The system integrates **MediaPipe** for hand tracking, **Stable Diffusion** for AI art generation, and **Arduino** for interactive feedback using an RGB LED and a button.

---

## Features
1. **Hand Tracking**:
   - Tracks hand gestures via a webcam and translates finger movements into strokes on a virtual canvas.
2. **AI Art Generation**:
   - Generates vibrant abstract artwork from the virtual canvas using Stable Diffusion.
3. **Arduino Integration**:
   - **Button**: Starts the canvas drawing process.
   - **RGB LED**:
     - Red: Indicates the system is tracking hand movements.
     - Green: Indicates that AI art is being generated.
     - Off: Indicates the system is idle or has completed a task.
4. **Keyboard Controls**:
   - `q`: Quit the application.
   - `c`: Clear the canvas.
   - `s`: Save the canvas and generate AI artwork.

---

## Hardware Setup
### Materials Required:
- **Arduino Board** (e.g., Arduino Uno or Nano)
- **RGB LED**
- **Push Button**
- **3 Resistors** (e.g., 220Ω for each LED pin)
- Breadboard and jumper wires
- A computer with Python 3.8+ and Arduino IDE installed

### Wiring Diagram
Refer to the TinkerCAD wiring diagram below for hardware connections:
![TinkerCAD Wiring Diagram](Arduino/TinkerCAD_Wiring.png)

#### Wiring Explanation:
- **RGB LED**:
  - Connect the **red pin** of the RGB LED to `PIN 9` via a 220Ω resistor.
  - Connect the **green pin** to `PIN 10` via a 220Ω resistor.
  - Connect the **blue pin** to `PIN 11` via a 220Ω resistor.
  - Connect the common cathode/anode to **ground** or **5V** depending on the LED type.
- **Push Button**:
  - Connect one side of the button to `PIN 2`.
  - Connect the other side to **ground**.
  - Use a pull-up resistor if your button requires it.

---

## Software Setup
### Prerequisites:
1. **Python 3.8+** installed on your computer.
2. **Arduino IDE** installed for uploading the Arduino sketch.
3. A **webcam** connected to your computer for hand tracking.

---

## Setup Instructions

### Step 1: Arduino Setup
1. Ensure the following Arduino pins are used:
   - **PIN 9**: Red LED pin.
   - **PIN 10**: Green LED pin.
   - **PIN 11**: Blue LED pin.
   - **PIN 2**: Button input.
2. Upload the sketch to your Arduino board.
3. Verify that the Arduino is connected to the correct **COM port**.

---

### Step 2: Python Dependencies
1. Navigate to the `Python/` folder:
   ```bash
   cd Python
2. Install the required Python packages by running: pip install -r requirements.txt
3. The requirements.txt file includes the following dependencies:
  - mediapipe: For hand tracking.
  - diffusers: For Stable Diffusion AI art generation.
  - opencv-python: For webcam integration.
  - torch: For running Stable Diffusion.
  - numpy: For numerical operations.
  - pillow: For image manipulation.

### Step 3: Running the Application
Ensure the Arduino is connected and powered on.

Run the Python application:

bash
Copy code
python gesture_python.py
How it works:\

The program will wait for the button press to start the canvas.\
Once pressed:\
The system begins tracking your hand movements, lighting the RGB LED red.\
Your gestures are drawn on a virtual canvas in real time.\
Use the following keyboard controls:\

q: Quit the program.\
c: Clear the canvas (reset hand-drawn input).\
s: Save the canvas and start generating AI artwork. The LED will turn green during this process.\

## Debugging and Testing
### Common Issues:
No LED Feedback:

Verify the wiring of the RGB LED and resistors.\
Check if the correct pins (9, 10, 11) are defined in the Arduino sketch.\
\
Button Not Working:\
Ensure the button is wired correctly to PIN 2 with a pull-up resistor.\
\
No Hand Tracking:\
Ensure your webcam is properly connected.\
Check if the hand is clearly visible in the webcam feed.\
\
Blank AI Art Generation:\
Inspect the preprocessed canvas saved at:\
Debug/preprocessed_canvas.png\

Verify that the canvas contains meaningful input (not blank or noisy).\
\
Static Canvas Testing:\
Modify the script to use a predefined canvas for debugging:
   ```bash
static_canvas = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(static_canvas)
draw.rectangle([50, 50, 200, 200], fill="black", outline="blue", width=3)
