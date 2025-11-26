# Handspeak Lite: A Simple Sign Language Detector

## 📘 Overview

Handspeak Lite is a lightweight and real-time American Sign Language (ASL) alphabet recognition system built using MediaPipe for hand tracking and a Random Forest machine learning classifier for gesture prediction. Designed as a Data Engineering Capstone project for **CSCI 6991**, this system focuses on efficiency, modularity, and simplicity—allowing real-time ASL letter recognition without the need for a GPU.

This repository contains the complete code, model training pipeline, data collection utilities, and presentation materials.

---

## 🎥 Demo Video

You can watch the full project presentation here:
🔗 **[https://youtu.be/3iHxRUfecgg](https://youtu.be/3iHxRUfecgg)**

---

## 📑 Presentation Slides

Your final presentation slides are available in the repository:
📄 **Handspeak_Lite_Presentation1.pptx**

---

## 🚀 Features

* Real-time ASL alphabet detection using a standard laptop webcam
* 21-point hand landmark extraction using MediaPipe
* Lightweight, CPU-friendly Random Forest classifier
* Custom data collection pipeline
* Data normalization for distance and orientation consistency
* Modular codebase: easy to extend with new gestures or datasets
* Webcam probing utility for device compatibility

---

## 🧠 System Architecture

```
Webcam → MediaPipe Hand Tracking → Landmark Extraction → Normalization →
Random Forest Classifier → Real-Time Predictions
```

---

## 📂 Project Structure

```
sign-language-detector/
│
├── collect_data.py           # Script to collect labeled hand landmark data
├── train_model.py            # Trains Random Forest classifier
├── detect_debug.py           # Real-time ASL detection and visualization
├── probe_cameras.py          # Tests webcam availability and settings
│
├── raw_landmarks.csv         # Collected dataset (if included)
├── model.joblib              # Trained ML model
│
├── Handspeak_Lite_Presentation1.pptx  # Final presentation slides
├── README.md                 # Project documentation
```

---

## ⚙️ Installation

### **1. Clone the repository**

```bash
git clone https://github.com/Planetharun/sign-language-detector.git
cd sign-language-detector
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

Ensure Python 3.10+ is installed.

---

## 📝 Usage Guide

### **1. Probe available cameras**

```bash
python probe_cameras.py
```

This helps determine the correct camera index.

### **2. Collect data samples**

```bash
python collect_data.py
```

Press the letter key (A–Z) to record hand landmarks.

### **3. Train the model**

```bash
python train_model.py
```

This will:

* Load the dataset
* Normalize landmark geometry
* Train a Random Forest classifier
* Save the model as `model.joblib`

### **4. Run live ASL detection**

```bash
python detect_debug.py
```

Your prediction and confidence scores appear on screen in real time.

---

## 📊 Model Details

* **Model Type:** Random Forest Classifier
* **Number of Trees:** 300
* **Input Dimensions:** 63 landmark coordinates (x, y, z)
* **Normalization:** Wrist-centered scaling
* **Dataset:** Custom-collected through webcam

---

## 🧪 Experiments & Results

* Achieved strong accuracy across most ASL alphabet signs
* Real-time performance maintained at smooth FPS on CPU-only systems
* Best performance on letters with distinct hand shapes: A, B, C, L
* Confusion occurs between visually similar letters (e.g., M, N, S)

---

## 🗣️ Team Contributions

**Tharun Guntupalli**

* Full pipeline engineering (data collection → training → detection)
* Script development and debugging
* Model tuning and normalization logic
* Presentation video creation

**Lakshmi Narayana Naraboiena**

* Literature review and research
* Dataset planning and documentation
* Experimentation support
* Presentation preparation

---

## 📚 Technologies Used

* **Python 3**
* **MediaPipe Hands** (landmark tracking)
* **OpenCV** (video processing)
* **Scikit-learn** (Random Forest model)
* **Joblib** (model persistence)

---

## 🙏 Acknowledgments

Special thanks to:

* **Dr. Robert Gilliland**, faculty advisor
* Developers of **MediaPipe**, **OpenCV**, and **Scikit-learn**
* ASL educational resources and open-source contributors

---

## 📬 Contact

For any questions or improvements, feel free to open an issue or contact:
📧 **Tharun Guntupalli**

---

### ✔ Your README is ready to copy into GitHub! Let me know if you want a PDF or want to refine the formatting.
