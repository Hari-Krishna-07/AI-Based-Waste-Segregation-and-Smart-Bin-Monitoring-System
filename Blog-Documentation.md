# AI-Based Smart Waste Segregation and Management System

> Technical / Project Documentation  
> **Architecture:** YOLOv8 + Deep Feature Extraction + SVM + OpenCV + ESP32  
> **Waste Classes:** Paper, Plastic, Metal, Glass

---

## 1. Project Overview

This project implements an AI-based smart waste segregation system that automatically identifies waste and controls a physical sorting mechanism.

The system combines:

- Computer vision using OpenCV
- YOLOv8 deep feature extraction
- SVM-based classification
- Confidence-based decision making
- Multi-frame majority voting
- Serial communication
- ESP32-based hardware control
- Ultrasonic sensors
- Servo motors
- LCD/status display

The important architectural point is that the current implementation is a **hybrid YOLOv8 + SVM system**.

- **OpenCV** detects/localizes the object in the camera scene.
- **YOLOv8** extracts deep visual features.
- **Scaler** normalizes those features.
- **SVM** performs the final waste classification.
- **ESP32** receives the final class and controls the physical mechanism.

---

## 2. Overall System Architecture

```text
                         WASTE OBJECT
                              |
                              v
                           CAMERA
                              |
                              v
                           OpenCV
                              |
                    Object Presence / ROI
                              |
                              v
                        Object Crop
                              |
                              v
                           YOLOv8
                              |
                     Deep Feature Vector
                         (256-D)
                              |
                              v
                           Scaler
                              |
                              v
                            SVM
                              |
                     10-Frame Voting
                              |
                              v
                    Final Waste Category
                              |
                    +---------+---------+
                    |                   |
                  VALID              UNKNOWN
                    |                   |
                    v                   v
               Serial to ESP32      No Hardware
                    |                Command
                    v
                   ESP32
                    |
             Bin-Level Verification
                    |
                    v
              Servo / Mechanism
                    |
                    v
              Waste Segregation
```

---

## 3. YOLO Version

The project uses **YOLOv8 through the Ultralytics framework**.

The trained model is stored as:

```text
best.pt
```

The model is loaded with:

```python
from ultralytics import YOLO

model = YOLO(MODEL_PATH)
```

### Role of YOLOv8 in this project

YOLOv8 is currently used primarily as a **deep feature extractor**.

The project does not use the YOLO output directly as the final four-class SVM decision.

The pipeline is:

```text
Image
  |
  v
YOLOv8
  |
  v
Deep Embedding
  |
  v
Feature Scaling
  |
  v
SVM
  |
  v
Waste Class
```

---

## 4. Dataset Organization

The feature-extraction program expects the dataset to be organized into training, validation, and testing folders.

```text
dataset/
|
+-- train/
|   +-- glass/
|   +-- metal/
|   +-- paper/
|   +-- plastic/
|
+-- val/
|   +-- glass/
|   +-- metal/
|   +-- paper/
|   +-- plastic/
|
+-- test/
    +-- glass/
    +-- metal/
    +-- paper/
    +-- plastic/
```

The target classes are:

1. Paper
2. Plastic
3. Metal
4. Glass

The exact number of images in each split is determined by the dataset used during the experiment and should be reported from the actual dataset rather than estimated.

---

## 5. Feature Extraction

The project contains an `extract.py` program whose purpose is to convert every dataset image into a numerical representation.

For every image:

```text
Input Image
    |
    v
Trained YOLOv8
    |
    v
YOLO Embedding
    |
    v
Flattened Feature Vector
    |
    v
Feature + Class Label
    |
    v
CSV
```

The feature extractor uses:

```python
embeddings = model.embed(
    source=str(image_path),
    imgsz=224,
    verbose=False
)
```

The resulting embedding is converted to a NumPy array and flattened.

The real-time system expects a **256-dimensional feature vector**.

### HOG is not used

The current feature-extraction implementation does **not** use HOG (Histogram of Oriented Gradients).

The feature type is:

> **YOLOv8 deep embedding features**

Therefore:

```text
Image -> YOLOv8 -> Deep Features -> SVM
```

rather than:

```text
Image -> HOG -> SVM
```

---

## 6. Feature CSV Files

The extraction process generates:

```text
yolo_features_train.csv
yolo_features_val.csv
yolo_features_test.csv
```

Each row represents one image and contains numerical feature values plus the corresponding class label.

Conceptually:

| feature_0 | feature_1 | ... | feature_255 | label |
|---:|---:|---:|---:|---|
| 0.21 | 0.43 | ... | 0.62 | plastic |
| 0.73 | 0.18 | ... | 0.41 | metal |
| 0.32 | 0.67 | ... | 0.73 | paper |
| 0.81 | 0.29 | ... | 0.19 | glass |

The feature CSVs form the input dataset for the SVM stage.

---

## 7. Image Preprocessing

The real-time application uses OpenCV before feature extraction.

The camera is configured at:

```text
640 x 480
```

The system uses a predefined Region of Interest (ROI) to restrict object detection to the expected waste-placement area.

The processing sequence is approximately:

```text
Camera Frame
    |
    v
ROI
    |
    v
Grayscale
    |
    v
Blur
    |
    v
Background Subtraction
    |
    v
Thresholding
    |
    v
Morphological Processing
    |
    v
Contour Detection
    |
    v
Object Bounding Box
```

---

## 8. Background Subtraction

The system first captures the scene without waste.

Multiple empty frames are used to construct a background reference.

Conceptually:

```text
Empty Scene
    |
    v
Background Reference
```

When waste is placed:

```text
Current Frame - Background
          |
          v
     Difference
          |
          v
   Potential Object
```

This allows the system to determine whether a new object has entered the designated region.

---

## 9. Region of Interest (ROI)

The current code defines an ROI approximately as:

```text
ROI_X1 = 220
ROI_Y1 = 120

ROI_X2 = 420
ROI_Y2 = 350
```

An additional object-placement region is used to check whether the detected object's center is in the expected position.

This reduces false triggers from irrelevant areas outside the intended detection region.

---

## 10. Contour-Based Object Localization

After background subtraction, the system processes the resulting image and searches for contours.

A minimum contour-area threshold is used to ignore small noise regions.

The valid object contour is then converted into a bounding box.

Conceptually:

```text
Difference Image
      |
      v
Threshold
      |
      v
Morphology
      |
      v
Contours
      |
      v
Largest Valid Object
      |
      v
Bounding Box
      |
      v
Object Crop
```

This means OpenCV is responsible for locating the object before YOLO feature extraction.

---

## 11. Real-Time YOLOv8 Inference

Once the object has been localized, the object region is cropped.

The cropped object is then passed to YOLOv8 for embedding extraction.

```text
Camera Frame
     |
     v
OpenCV Bounding Box
     |
     v
Object Crop
     |
     v
YOLOv8
     |
     v
Feature Vector
```

The embedding extraction uses an image size of:

```text
224 x 224
```

The resulting feature vector is expected to contain:

```text
256 features
```

---

## 12. Feature Scaling

The project loads the trained scaler:

```text
final_scaler.pkl
```

The extracted YOLO features are transformed before SVM prediction.

```text
YOLO Features
     |
     v
Scaler
     |
     v
Scaled Features
     |
     v
SVM
```

Using the same scaler during training and inference is important because the SVM must receive features in the same representation used during training.

---

## 13. SVM Classification

The final classifier is a trained Support Vector Machine.

The model is stored as:

```text
final_svm_model.pkl
```

The SVM receives the scaled YOLO feature vector.

For example:

```text
PAPER       5%
PLASTIC     8%
METAL      82%
GLASS       5%
```

The highest-probability class is selected:

```text
METAL
```

Therefore:

> YOLOv8 extracts the visual representation, while SVM performs the final waste classification.

---

## 14. Confidence Threshold

The current real-time pipeline uses a confidence threshold of:

```text
75%
```

For example:

```text
METAL = 86%
```

can be accepted.

A prediction such as:

```text
METAL = 61%
```

is treated as insufficiently confident.

This provides a safety layer before a classification is sent to the physical hardware.

---

## 15. Multi-Frame Voting

The system does not depend on a single camera frame.

The current configuration uses:

```text
10 frames per decision
```

For example:

```text
Frame 1  -> METAL
Frame 2  -> METAL
Frame 3  -> GLASS
Frame 4  -> METAL
Frame 5  -> METAL
Frame 6  -> METAL
Frame 7  -> METAL
Frame 8  -> GLASS
Frame 9  -> METAL
Frame 10 -> METAL
```

The result is:

```text
METAL = 8/10
GLASS = 2/10
```

Therefore:

```text
Final Class = METAL
```

---

## 16. Majority-Voting Safety Rule

The current implementation requires at least:

```text
6 out of 10 frames
```

to agree on the final class.

For example:

```text
8/10 -> METAL
```

is accepted.

But:

```text
5/10 -> METAL
5/10 -> GLASS
```

does not satisfy the majority requirement and can result in:

```text
UNKNOWN
```

The system also considers the confidence of accepted predictions.

The purpose is to reduce the effect of a temporary incorrect prediction caused by:

- Motion
- Blur
- Lighting
- Reflections
- Object orientation
- Detection noise

---

## 17. Valid Output Classes

Only the four supported classes are considered valid:

```text
PAPER
PLASTIC
METAL
GLASS
```

An invalid or uncertain result should not trigger a physical sorting command.

---

## 18. ESP32 Communication

After a valid final classification is obtained, the Python application communicates with the ESP32 over serial communication.

The current configuration uses:

```text
Port: COM5
Baud Rate: 115200
```

A valid class is transmitted to the ESP32.

Example:

```text
Python
   |
   | "METAL\n"
   v
ESP32
```

The ESP32 then uses the received category to control the mechanical sorting system.

---

## 19. Hardware Architecture

The hardware side includes:

- ESP32
- Camera
- Ultrasonic sensors
- Servo motors
- LCD/status display
- Waste compartments
- Mechanical trapdoor/bin mechanism

The computer handles the AI processing while the ESP32 handles hardware control.

This is a hybrid:

```text
Computer / AI
     |
     v
Serial Communication
     |
     v
ESP32 / Hardware
```

---

## 20. Ultrasonic Sensors

The hardware design uses ultrasonic sensing for object/bin-state monitoring.

### Sensor 1

Used for object-presence detection in the relevant hardware stage.

Conceptually:

```text
Waste enters
     |
     v
Ultrasonic Sensor
     |
     v
Object Present
```

### Sensor 2

Used for bin-level monitoring by measuring distance to the waste inside the selected bin.

Conceptually:

```text
Ultrasonic Sensor
      |
      v
Distance Measurement
      |
      v
Fill-Level Decision
      |
      v
Bin Full / Available
```

The exact sensor thresholds and hardware control logic depend on the ESP32 firmware.

---

## 21. Servo Motors

The physical system uses servo motors for mechanical control.

### Servo 1 â€” Trapdoor

Controls opening and closing of the waste-release mechanism.

```text
Bin Available
     |
     v
Open Trapdoor
     |
     v
Waste Falls
     |
     v
Close Trapdoor
```

### Servo 2 â€” Bin Rotation

Moves the sorting mechanism to the appropriate waste compartment.

Conceptually:

```text
PAPER   -> Paper position
PLASTIC -> Plastic position
METAL   -> Metal position
GLASS   -> Glass position
```

The exact servo angles are defined by the ESP32 firmware.

---

## 22. Bin-Full Protection

Before releasing waste, the selected compartment should be checked.

```text
Final AI Class
      |
      v
Selected Bin
      |
      v
Check Fill Level
      |
   +--+--+
   |     |
 FULL  AVAILABLE
   |     |
   v     v
Stop   Rotate
       |
       v
   Open Trapdoor
       |
       v
   Waste Released
       |
       v
   Close Trapdoor
```

If the bin is full, the system can prevent additional waste from being released into that compartment.

---

## 23. FPS / Real-Time Performance

The current code establishes the camera resolution and inference pipeline, but it does **not provide a verified benchmark FPS value**.

The camera is configured for:

```text
640 x 480
```

However, camera resolution is not the same as processing FPS.

Actual FPS depends on:

- CPU/GPU hardware
- YOLO model configuration
- OpenCV processing time
- Feature extraction time
- SVM prediction time
- Camera capture rate
- Serial communication
- Hardware response time

### Recommended benchmark

Measure:

```text
FPS = Number of processed frames / Total processing time
```

For the final report, insert the measured result:

```text
Measured FPS: [XX FPS]
Deployment hardware: [CPU/GPU/device]
```

Do not report an estimated FPS as a measured result.

---

## 24. Accuracy, Precision, Recall and F1-Score

The final system should be evaluated on the test dataset.

Recommended metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The final report should use the actual measured values from the project's test results.

```text
Test Accuracy: [XX%]
Precision:     [XX%]
Recall:        [XX%]
F1-score:      [XX%]
```

These values should not be invented if they have not yet been measured.

---

## 25. mAP

mAP (mean Average Precision) is primarily associated with object-detection and instance-segmentation evaluation.

The current real-time pipeline uses:

```text
OpenCV -> object localization
YOLOv8 -> feature extraction
SVM -> final classification
```

Therefore, a YOLO object-detection mAP value should **not** be claimed for the current pipeline unless an object-detection model was separately trained and evaluated using detection annotations.

If the YOLO training experiment produced classification metrics instead, those metrics should be reported instead of presenting them as mAP.

---

## 26. Current Technical Specifications

| Specification | Current Project |
|---|---|
| YOLO | YOLOv8 |
| YOLO role | Deep feature extraction |
| Final classifier | SVM |
| Feature type | YOLO deep embeddings |
| Expected feature size | 256 |
| Feature extraction input size | 224 x 224 |
| Camera resolution | 640 x 480 |
| Object localization | OpenCV |
| Background method | Background subtraction |
| Classification frames | 10 |
| Minimum agreeing frames | 6/10 |
| Confidence threshold | 75% |
| Valid classes | Paper, Plastic, Metal, Glass |
| SVM model | `final_svm_model.pkl` |
| Scaler | `final_scaler.pkl` |
| YOLO model | `best.pt` |
| Serial port | COM5 |
| Serial baud rate | 115200 |
| FPS | To be experimentally measured |
| Accuracy | To be experimentally measured/reported |
| mAP | Not established for the supplied pipeline |

---

## 27. Limitations

### 27.1 Background sensitivity

The OpenCV localization stage depends on background subtraction.

Changes in:

- Lighting
- Shadows
- Camera position
- Background appearance
- Reflections

can affect object detection.

### 27.2 Controlled placement

The object is expected to enter the predefined detection/placement area.

Objects placed outside this region may not be detected correctly.

### 27.3 Multiple objects

The current contour-based approach is designed around identifying the relevant object region. Multiple simultaneous objects may not be separated reliably.

### 27.4 Dataset dependency

Classification performance depends on how representative the training data is.

Useful variation includes:

- Different object shapes
- Different sizes
- Different orientations
- Different lighting
- Different backgrounds
- Crumpled/damaged waste
- Partially visible objects

### 27.5 Similar materials

Some waste materials are visually similar.

Examples include:

- Transparent glass vs transparent plastic
- Reflective metal vs reflective packaging
- Paper-coated plastic
- Plastic-coated paper

### 27.6 Hardware latency

The complete response time includes:

```text
Camera
  -> OpenCV
  -> YOLOv8
  -> Scaler
  -> SVM
  -> Voting
  -> Serial Communication
  -> ESP32
  -> Servo
```

Therefore, total system latency is not determined by YOLO inference time alone.

---

## 28. Deployment Approach

The project uses a hybrid deployment approach.

### AI computer

Runs:

```text
Python
 |
 +-- OpenCV
 +-- Ultralytics YOLOv8
 +-- NumPy
 +-- Pandas
 +-- SVM / Scikit-learn
 +-- Joblib
 +-- Serial communication
```

### Embedded controller

The ESP32 handles:

```text
ESP32
 |
 +-- Ultrasonic sensors
 +-- Servo motors
 +-- LCD
 +-- Mechanical sorting system
```

The architecture avoids requiring the ESP32 to run the complete YOLO inference pipeline.

---

## 29. Model and Deployment Files

The main AI artifacts are:

```text
best.pt
final_svm_model.pkl
final_scaler.pkl
```

Feature datasets are:

```text
yolo_features_train.csv
yolo_features_val.csv
yolo_features_test.csv
```

The main real-time application is:

```text
detect.py
```

The feature-generation program is:

```text
extract.py
```

---

## 30. End-to-End Example

Consider a plastic bottle.

### Step 1 â€” Object enters

The camera captures the scene.

### Step 2 â€” OpenCV

Background subtraction and contour processing detect the object.

### Step 3 â€” Crop

The object bounding box is extracted.

### Step 4 â€” YOLOv8

The crop is passed through YOLOv8.

```text
Object Image
     |
     v
YOLOv8
     |
     v
256-D Feature Vector
```

### Step 5 â€” Scaling

The feature vector is transformed using the trained scaler.

### Step 6 â€” SVM

The SVM predicts the waste category.

Example:

```text
PLASTIC = 91%
```

### Step 7 â€” Multi-frame decision

After 10 frames:

```text
PLASTIC = 9/10
```

The result satisfies the majority requirement.

### Step 8 â€” Final classification

```text
FINAL = PLASTIC
```

### Step 9 â€” ESP32

Python sends:

```text
PLASTIC
```

### Step 10 â€” Mechanical sorting

The ESP32 checks the selected bin and controls the servo mechanism.

---

## 31. Why Use YOLOv8 + SVM?

The project separates feature extraction from classification.

### YOLOv8

Provides a learned visual representation of the waste image.

### SVM

Uses the extracted feature representation to make the final class decision.

This produces a hybrid architecture:

```text
Deep Learning
      +
Machine Learning
      |
      v
Hybrid Waste Classifier
```

The approach also makes it possible to generate reusable feature datasets and train/evaluate the SVM independently.

---

## 32. Why Use Multi-Frame Voting?

A physical sorting mechanism should not immediately react to one uncertain prediction.

The system therefore evaluates several consecutive frames:

```text
Frame 1  \
Frame 2   \
Frame 3    \
Frame 4     \
Frame 5      >  Majority Voting
Frame 6     /
Frame 7    /
Frame 8   /
Frame 9  /
Frame 10/
```

This provides temporal stability and reduces the effect of occasional incorrect predictions.

---

## 33. Future Improvements

Potential improvements include:

### YOLO-based object localization

A future version could use a dedicated YOLO object-detection model for localization instead of relying on background subtraction.

Current:

```text
Camera
  -> OpenCV
  -> Object Crop
  -> YOLO Embedding
  -> SVM
```

Possible future version:

```text
Camera
  -> YOLO Detection
  -> Object Crop
  -> Feature Extraction
  -> SVM
```

This could make the system less dependent on a fixed background.

### More waste categories

Additional categories could be introduced, such as:

- Organic
- E-waste
- Cardboard
- Textile
- Hazardous waste

### Edge deployment

The AI pipeline could eventually be moved from a desktop/laptop to an edge-computing platform with suitable AI acceleration.

### Dataset expansion

Real-world images collected during deployment could be reviewed and added to the training dataset to improve generalization.

---

## 34. Conclusion

The AI-Based Smart Waste Segregation and Management System combines computer vision, deep learning, machine learning, and embedded hardware into an automated waste-sorting workflow.

The current architecture uses **YOLOv8 for deep feature extraction**, **OpenCV for object localization**, and **SVM for final classification** into Paper, Plastic, Metal, or Glass.

A scaler normalizes the YOLO feature vector before SVM prediction. The system then evaluates multiple frames and applies majority voting with a minimum **6/10 agreement** and a **75% confidence threshold** before sending a valid classification to the ESP32.

The ESP32 acts as the hardware controller and integrates ultrasonic sensing, bin-level monitoring, servo control, and the physical waste-segregation mechanism.

The complete pipeline is:

```text
                    SMART WASTE SYSTEM

                         CAMERA
                            |
                            v
                         OpenCV
                            |
                     Object Localization
                            |
                            v
                       Object Crop
                            |
                            v
                         YOLOv8
                            |
                       Deep Features
                          (256-D)
                            |
                            v
                          Scaler
                            |
                            v
                           SVM
                            |
                   10-Frame Voting
                            |
                            v
                Final Waste Classification
                            |
                            v
                       Serial / USB
                            |
                            v
                          ESP32
                            |
                    Bin-Level Verification
                            |
                            v
                     Servo Mechanism
                            |
                            v
                   AUTOMATED SEGREGATION
```

---

