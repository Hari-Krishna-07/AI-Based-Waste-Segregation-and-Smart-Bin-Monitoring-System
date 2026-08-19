# AI-Based-Waste-Segregation-and-Smart-Bin-Monitoring-System
# AI-Based Smart Waste Segregation and Management System

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
