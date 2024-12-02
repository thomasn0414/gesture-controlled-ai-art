#define RED_PIN 9
#define GREEN_PIN 10
#define BLUE_PIN 11
#define BUTTON_PIN 2

void setup() {
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.begin(9600);
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    if (command == "red") {
      setLED(255, 0, 0);  // Red for tracking
    } else if (command == "green") {
      setLED(0, 255, 0);  // Green for generating
    } else if (command == "off") {
      setLED(0, 0, 0);    // Turn off LED
    }
  }

  // Detect button press to start the canvas
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(200);  // Debounce
    Serial.println("start_canvas");
  }
}

void setLED(int red, int green, int blue) {
  analogWrite(RED_PIN, red);
  analogWrite(GREEN_PIN, green);
  analogWrite(BLUE_PIN, blue);
}

