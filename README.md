# ✋ Sign-Language-Detection-using-YOLO
**Output**: Real-time Gesture Detection + Voice Feedback

---

## 🧠 Overview

This project implements **real-time detection of American Sign Language (ASL) gestures** using the **YOLOv8** object detection framework developed by Ultralytics. It combines state-of-the-art computer vision and deep learning techniques to enable interactive hand gesture recognition from live webcam input.

The system detects and classifies ASL hand signs in real time, then **speaks the predicted sign aloud** using a text-to-speech engine. This project showcases the integration of **custom object detection**, **gesture classification**, and **speech feedback**, designed to assist communication for hearing or speech-impaired individuals.

---

## 🚀 Features

- **ASL Gesture Detection**: Real-time hand sign recognition using YOLOv8 trained on the ASL alphabet.
- **Text-to-Speech**: Detected sign is immediately spoken aloud using `pyttsx3`.
- **Custom YOLOv8 Model**: Trained on a curated ASL dataset (A–Z) with 5,000+ samples.
- **Webcam Inference**: Frame-by-frame gesture detection using OpenCV.
- **Lightweight & Fast**: Uses YOLOv8 Nano variant for smooth performance even on CPU systems.

---

## ⚙️ Prerequisites

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- OpenCV (`pip install opencv-python`)
- pyttsx3 (`pip install pyttsx3`)
- A webcam-enabled system
- (Optional) CUDA-enabled GPU for faster training/inference

---

## 🔬 Technical Details

### 🎓 Model Training

This project fine-tunes YOLOv8 on a **converted object detection version of the ASL alphabet dataset**. Each hand gesture image is labeled to occupy the full frame (since each image contains a single centered sign). A reduced subset of 5,000 images is used for efficient model convergence.

- **Model**: YOLOv8 Nano (`yolov8n.pt`)
- **Dataset**: [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Epochs**: 10  
- **Image Size**: 416 × 416  
- **Output Weights**: `sign_language_project/yolov8n_asl_5k/weights/best.pt`

### 🧪 Real-Time Inference

- Captures live webcam input using OpenCV
- Detects hand gesture in every frame
- Draws bounding box + class label on screen
- Uses TTS engine to speak the label aloud
- Press `'q'` to quit the live session

---

## 🧹 Challenges Addressed

- 📷 Handling webcam access and frame errors across platforms
- ⚡ Ensuring fast inference on non-GPU machines with the Nano model
- 🔊 Syncing voice feedback without blocking frame updates
- 📊 Reducing training time while maintaining model accuracy

---

_Contact or raise an issue to contribute or extend this project further!_
