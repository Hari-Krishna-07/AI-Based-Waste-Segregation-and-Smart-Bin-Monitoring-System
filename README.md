# AI-Based-Waste-Segregation-and-Smart-Bin-Monitoring-System

An AI and embedded-systems-based smart waste segregation system that automatically identifies waste as **Plastic, Paper, Glass, or Metal** and controls a servo-based segregation mechanism using an **ESP32**.

The system combines **YOLOv8, SVM, OpenCV, Python, USB Webcam, ultrasonic sensing, and servo motors** to automate waste classification and bin management.

---

## 📌 Project Overview

Manual waste segregation is time-consuming and can lead to incorrect classification of recyclable materials. This project proposes an automated waste segregation system in which a camera captures the waste item, an AI-based classification pipeline identifies its category, and an ESP32 controls the mechanical segregation process.

The system is designed to:

- Detect the presence of a waste object
- Capture the object using a USB webcam
- Classify the waste using YOLOv8 and SVM
- Identify the waste as Plastic, Paper, Glass, or Metal
- Send the final classification to the ESP32 through USB serial communication
- Rotate the waste bin using a servo motor
- Monitor the selected bin level using an ultrasonic sensor
- Prevent waste from entering a full bin
- Operate the trapdoor using a second servo motor

---

## 🎯 Problem Statement

Improper waste segregation makes recycling difficult and increases the amount of recyclable material sent to landfills.

A smart automated system is therefore required to:

- Identify different types of waste automatically
- Reduce manual segregation
- Improve sorting consistency
- Monitor bin capacity
- Prevent overflow with buzzer
- Automate the mechanical segregation process

---

## 💡 Proposed Solution

The proposed system uses a **USB webcam** to observe the waste placement area.

When a waste object is detected, multiple frames are captured. YOLOv8 is used to generate feature embeddings from the waste image. These features are scaled and passed to an **SVM classifier**, which produces the final waste classification.

The detected class is accepted only when the classification satisfies the programmed confidence and majority-voting conditions.

The final valid classification is sent from the Python program to the ESP32 through **USB Serial communication at 115200 baud**.

The ESP32 then:

1. Receives the waste category.
2. Rotates the bin to the corresponding compartment.
3. Checks the selected bin level using the ultrasonic sensor.
4. If the bin is full, prevents the trapdoor from opening and the buzzer alerts the user.
5. If the bin is not full, opens the trapdoor.
6. Allows the waste to fall into the selected compartment.
7. Closes the trapdoor and prepares for the next waste item.

---

## 🧠 AI/ML Methodology

### YOLOv8

The project uses the YOLOv8 classification model:

```text
yolov8n-cls.pt
```

The trained YOLO model is used to obtain feature embeddings from the detected waste object.

The current detection program uses the trained model to generate a **256-feature embedding** for each object crop.

### SVM

The YOLO-generated features are passed through a scaler and then supplied to the trained SVM model.

The SVM performs the final classification into one of the four waste categories:

- Plastic
- Paper
- Glass
- Metal

### Classification Pipeline

```text
USB Webcam
     ↓
Object Detection / ROI Processing
     ↓
Waste Object Crop
     ↓
YOLOv8 Feature Embedding
     ↓
Feature Scaling
     ↓
SVM Classifier
     ↓
Confidence Check
     ↓
10-Frame Majority Voting
     ↓
Final Waste Class
     ↓
ESP32
```

The implementation loads a trained SVM model and feature scaler using Joblib. The classifier must support probability prediction because the program uses SVM class probabilities for confidence checking. 

---

## 🔄 System Working

### Step 1 — System Initialization

The Python program:

- Connects to the ESP32
- Loads the trained YOLO model
- Loads the SVM model
- Loads the feature scaler
- Opens the USB webcam
- Captures an empty background

### Step 2 — Waste Detection

The webcam continuously monitors a predefined region of interest.

OpenCV performs background subtraction and image processing to detect an object placed in the specified area.

The program uses:

- Grayscale conversion
- Gaussian blur
- Background difference
- Thresholding
- Morphological operations
- Contour detection
- Minimum-area filtering

### Step 3 — Waste Classification

After an object is detected, the object is cropped from the camera frame.

YOLOv8 generates the feature embedding from the crop.

The embedding is then:

```text
YOLO Embedding
      ↓
Scaler
      ↓
SVM
      ↓
Class Probability
```

A classification is accepted only when the SVM confidence reaches the programmed threshold.

### Step 4 — Multi-Frame Decision

The system does not rely on a single camera frame.

It collects **10 valid classification frames**.

The final class is selected using majority voting.

At least **6 out of 10 frames** must agree on the same class. Otherwise, the result is treated as **UNKNOWN**.

This helps reduce unstable frame-to-frame predictions.

### Step 5 — ESP32 Communication

Only a valid waste classification is sent to the ESP32.

The command is transmitted through USB serial communication at:

```text
115200 baud
```

Example commands:

```text
PLASTIC
PAPER
GLASS
METAL
```

Invalid or unknown classifications are blocked and are not sent to the ESP32.

### Step 6 — Bin Rotation

The ESP32 receives the waste category and controls the **bin-rotation servo motor**.

The servo rotates the waste compartment to the required position.

### Step 7 — Bin-Level Monitoring

An ultrasonic sensor is used to monitor the level of the selected bin.

The programmed condition is:

```text
Distance < 7 cm → BIN FULL
```

If the selected bin is full, the system does not release the waste.

### Step 8 — Trapdoor Control

If the selected bin has sufficient space:

```text
Bin available
     ↓
Trapdoor Servo opens
     ↓
Waste falls into selected compartment
     ↓
Trapdoor closes
     ↓
System ready for next waste
```

---

## ⚙️ Decision Logic

```text
START
  │
  ▼
Wait for Waste
  │
  ▼
Object Detected?
  │
  ├── NO ──► Continue Monitoring
  │
  ▼ YES
Capture Multiple Frames
  │
  ▼
YOLOv8 Feature Extraction
  │
  ▼
SVM Classification
  │
  ▼
Confidence Check
  │
  ├── LOW ──► Ignore Frame
  │
  ▼ VALID
10 Valid Frames Completed
  │
  ▼
Majority Voting
  │
  ├── Less than 6/10 agreement
  │          ↓
  │       UNKNOWN
  │          ↓
  │       Do Not Send
  │
  ▼ Valid Result
Send Waste Type to ESP32
  │
  ▼
Rotate Bin Servo
  │
  ▼
Check Bin Level
  │
  ├── < 7 cm ──► BIN FULL
  │                  ↓
  │             Do Not Open
  │
  ▼ Bin Available
Open Trapdoor Servo
  │
  ▼
Waste Falls
  │
  ▼
Close Trapdoor
  │
  ▼
Wait for Waste Removal
  │
  ▼
Ready for Next Waste
```

---

## 🔌 Hardware Components

| S.No | Component | Quantity | Purpose |
|---:|---|---:|---|
| 1 | ESP32 | 1 | Main embedded controller |
| 2 | USB Webcam | 1 | Waste image acquisition |
| 3 | Ultrasonic Sensor | 1 | Bin-level monitoring |
| 4 | Servo Motor | 2 | Bin rotation and trapdoor control |
| 5 | Display | 1 | System/status indication |
| 6 | Breadboard | 1 | Circuit prototyping |
| 7 | Jumper Wires | As required | Electrical connections |

---

## 💻 Software and Technologies

### Programming

- Python
- Embedded C/C++ / Arduino programming

### AI/ML

- YOLOv8
- SVM
- Scikit-learn
- Joblib

### Computer Vision

- OpenCV
- NumPy

### Embedded System

- ESP32
- Arduino IDE

### Communication

- USB Serial Communication
- Baud Rate: **115200**

---

## 📂 Repository Structure

A recommended GitHub structure for the project is:

```text
AI-Based-Smart-Waste-Segregation-and-Management-System/
│
├── README.md
│
├── Python/
│   ├── detect.py
│   ├── train_svm.py
│   └── ...
│
├── ESP32/
│   └── smart_waste_management.ino
│
├── Models/
│   ├── yolov8n-cls.pt
│   ├── final_svm_model.pkl
│   └── final_scaler.pkl
│
├── Dataset/
│   ├── plastic/
│   ├── paper/
│   ├── glass/
│   └── metal/
│
├── Images/
│   ├── circuit_diagram.png
│   ├── system_setup.png
│   ├── plastic_detection.png
│   ├── paper_detection.png
│   ├── glass_detection.png
│   ├── metal_detection.png
│   └── bin_full.png
│
└── requirements.txt
```

> The exact filenames and folders should be adjusted to match the final files uploaded to the repository.

---

## 📋 Waste Classes

| Class | Description |
|---|---|
| 🟦 Plastic | Plastic waste |
| 📄 Paper | Paper-based waste |
| 🔩 Metal | Metallic waste |
| 🪟 Glass | Glass waste |

---

## 🛡️ Safety and Validation Logic

The software includes several checks before a classification is sent to the ESP32.

### Valid Class Check

Only these commands are accepted:

```text
PAPER
PLASTIC
METAL
GLASS
```

Any other classification is blocked.

### Confidence Check

The SVM confidence threshold is configured as:

```text
75%
```

Frames below this confidence are not counted toward the final decision.

### Majority Voting

The final result requires:

```text
Minimum 6 / 10 matching frames
```

If this condition is not satisfied:

```text
FINAL RESULT: UNKNOWN
```

The result is not sent to the ESP32.

### Object Removal

After a waste item is classified, the program waits until the object is removed before processing another item.

This prevents the same object from being repeatedly classified.

---

## 📷 Camera and Detection Configuration

The current Python implementation uses a USB webcam with:

```text
Frame Width  : 640 pixels
Frame Height : 480 pixels
```

The camera processing uses predefined regions for detecting and positioning the waste object.

The object is expected to be placed inside the specified detection/placement area.

---

## 🧪 Testing

The system can be tested using individual waste samples from the four supported categories.

### Test Cases

| Test | Input | Expected Result |
|---|---|---|
| 1 | Plastic | Plastic classification |
| 2 | Paper | Paper classification |
| 3 | Glass | Glass classification |
| 4 | Metal | Metal classification |
| 5 | Low-confidence object | UNKNOWN / no command |
| 6 | Full selected bin | BIN FULL / trapdoor remains closed |
| 7 | Available selected bin | Trapdoor opens |
| 8 | Object removed | System becomes ready for next waste |

---

## 📊 Results

The system is designed to provide the following sequence:

```text
Waste Placement
      ↓
Object Detection
      ↓
YOLOv8 + SVM Classification
      ↓
Waste Category
      ↓
ESP32
      ↓
Bin Rotation
      ↓
Bin-Level Check
      ↓
Trapdoor Control
```

Recommended result screenshots for the GitHub repository include:

- System setup
- Camera detection window
- Plastic classification
- Paper classification
- Glass classification
- Metal classification
- ESP32 serial output
- Full-bin condition
- Trapdoor operation

---

## ⚠️ Challenges Faced

During development, the project involved several practical challenges:

- Stabilizing waste classification across consecutive camera frames
- Integrating YOLOv8 embeddings with the SVM classifier
- Maintaining consistent feature dimensions
- Handling low-confidence predictions
- Preventing invalid classifications from reaching the ESP32
- Detecting the waste object using background subtraction
- Managing the transition between different system states
- Synchronizing Python classification with ESP32 servo control
- Monitoring bin capacity before releasing waste
- Preventing repeated classification of the same waste object

The current software addresses classification instability by using multiple valid frames and majority voting before producing the final result.

---

## ✅ Advantages

- Automated waste segregation
- Reduces manual sorting
- Supports four waste categories
- Combines deep-learning feature extraction with SVM classification
- Multi-frame decision making improves prediction stability
- Automatic bin rotation
- Automatic bin-level monitoring
- Prevents waste from entering a full bin
- Serial communication between AI system and ESP32
- Modular AI and embedded-system architecture
- Suitable for small-scale smart waste management applications

---

## ⚠️ Limitations

- The system currently supports four predefined waste categories.
- Classification performance depends on the quality and diversity of the training dataset.
- Camera lighting and background conditions can affect object detection.
- The system is designed for one waste object at a time.
- The current prototype uses a USB webcam connected to the computer.
- The prototype is intended for demonstration and research purposes and is not an industrial waste-management system.

---

## 🔮 Future Scope

The system can be further improved by:

- Adding more waste categories
- Improving the training dataset with different object orientations and lighting conditions
- Using a more advanced object-detection pipeline
- Adding real-time LCD/status information
- Adding IoT connectivity
- Sending bin-status notifications to a mobile application
- Adding cloud-based waste monitoring
- Recording waste segregation statistics
- Adding multiple ultrasonic sensors for individual compartment monitoring
- Adding a dashboard for monitoring waste collection
- Developing a more compact mechanical design
- Deploying the AI model on an edge-computing device

---

## 🌍 Applications

The proposed system can be adapted for:

- Educational institutions
- Smart classrooms
- Offices
- Laboratories
- Parks
- Small-scale recycling facilities

---

## 🏥 Relevance to Biomedical/Healthcare Engineering

The project demonstrates the application of:

- Embedded systems
- Artificial intelligence
- Computer vision
- Sensors
- Automation
- Microcontrollers
- Healthcare/environmental technology concepts

The same combination of sensing, AI-based decision making, and embedded control can be extended to automated systems used in healthcare and other controlled environments.

---

## 🚀 How to Run the Python System

### 1. Install Python

Install a compatible Python version on the computer.

### 2. Install Required Libraries

```bash
pip install ultralytics opencv-python numpy joblib pyserial scikit-learn
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

### 3. Connect the ESP32

Connect the ESP32 to the computer through USB.

Make sure the correct COM port is selected.

The current `detect.py` configuration uses:

```text
COM5
115200 baud
```

The COM port should be changed if Windows assigns a different port.

### 4. Upload the ESP32 Program

Open the ESP32 Arduino code in Arduino IDE.

Select the correct:

- Board
- COM port

Compile and upload the program.

Close the Arduino Serial Monitor before running Python because the Python program requires access to the serial port.

### 5. Check Model Files

The Python program requires:

```text
YOLO model
SVM model
Scaler
```

The trained model paths should be updated according to the location of the files on the computer.

### 6. Run Detection

From the project directory:

```bash
python detect.py
```

The program will:

1. Connect to ESP32
2. Load the AI models
3. Open the webcam
4. Capture the empty background
5. Wait for waste
6. Detect and classify the waste
7. Send the valid class to ESP32
8. Wait for object removal
9. Continue monitoring

---

## ⌨️ Keyboard Controls

The current detection program supports:

| Key | Function |
|---|---|
| `Q` | Quit the detection program |
| `R` | Reset/capture a new empty background |

Before resetting the background, remove all waste objects from the detection area.

---

## 🔧 Configuration

Important parameters in `detect.py` include:

```python
ESP32_PORT = "COM5"
ESP32_BAUD = 115200

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TOTAL_FRAMES = 10
SVM_CONFIDENCE_THRESHOLD = 0.75

EXPECTED_FEATURES = 256

EMPTY_FRAMES_REQUIRED = 15
```

These values can be adjusted according to the hardware setup and testing environment.

---

## 🔐 Prototype Disclaimer

This project is an academic and research prototype demonstrating AI-based waste classification and embedded automation.

It should not be considered a certified industrial waste-management product. Hardware, mechanical safety, electrical protection, and software reliability should be further validated before deployment in real-world environments.

---

## 🏁 Conclusion

The **AI-Based Smart Waste Segregation and Management System** demonstrates how artificial intelligence, computer vision, machine learning, sensors, and embedded systems can be integrated to automate waste segregation.

The system uses **YOLOv8 for feature extraction and SVM for final classification**, followed by ESP32-based mechanical control. Multi-frame classification and majority voting are incorporated to reduce unstable predictions, while ultrasonic bin-level monitoring prevents waste from being released into a full compartment.

The project provides a foundation for developing a more advanced smart waste-management system with IoT connectivity, additional waste categories, improved AI models, and real-time monitoring.

## ⭐ Project Highlights

```text
YOLOv8 + SVM
      │
      ▼
Waste Classification
      │
      ├── Plastic
      ├── Paper
      ├── Glass
      └── Metal
      │
      ▼
ESP32
      │
      ├── Bin Rotation Servo
      ├── Trapdoor Servo
      └── Ultrasonic Bin Monitoring
      │
      ▼
Smart Waste Segregation
```

---
---

## 👨‍💻 Author

**Hari Krishna R**

**Kajal K**

**Jeeva V**

**Sindhu S**

---

**Built as an academic project integrating Artificial Intelligence, Machine Learning, Computer Vision, and Embedded Systems.**
