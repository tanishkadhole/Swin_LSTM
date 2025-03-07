# **Deepfake Detection Model - Training & Inference Guide**

This repository contains the implementation of a **Deepfake Detection Model** using **Swin Transformer** for feature extraction and **LSTM** for classification.

## **Prerequisites**

Ensure you have Python **3.8+** installed. Then, install the required dependencies:

```bash
pip install torch torchvision timm facenet-pytorch opencv-python pillow numpy
```

### **Check GPU Availability (Optional but Recommended)**

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

---

## **Project Structure**

```
Deepfake-Detection/
├── data/
│   ├── videos/                  # Raw video files (real & fake videos)
│   │   ├── real/
│   │   │   ├── real_video_1.mp4
│   │   │   ├── real_video_2.mp4
│   │   ├── fake/
│   │   │   ├── fake_video_1.mp4
│   │   │   ├── fake_video_2.mp4
│   ├── extracted_frames/        # Frames extracted from videos
│   │   ├── real/
│   │   ├── fake/
│   ├── extracted_faces/         # Faces extracted from frames
│   │   ├── real/
│   │   ├── fake/
│   ├── train_faces/             # Training dataset (real/fake faces)
│   │   ├── real/
│   │   ├── fake/
│   ├── test_faces/              # Testing dataset (real/fake faces)
│   │   ├── real/
│   │   ├── fake/
├── dataset/
│   ├── extracted_features/      # Features extracted using Swin Transformer
│   │   ├── real.pt              # All real face features stored in a single file
│   │   ├── fake.pt              # All fake face features stored in a single file
│   ├── test_features/           # Features for evaluation
│   │   ├── real.pt
│   │   ├── fake.pt
├── models/
│   ├── swin_model_custom.pth    # Trained Swin Transformer model
│   ├── lstm_model_custom.pth    # Trained LSTM model
├── extract_frames.py            # Extract frames from video
├── extract_faces.py             # Detect and extract faces
├── swin_feature_extraction.py   # Feature extraction using Swin Transformer
├── train_swim.py                # Train the Swin Transformer model
├── train_lstm.py                # Train the LSTM model
├── detect.py                    # Deepfake detection script
├── evaluate.py                  # Model evaluation script
├── README.md                    # Project documentation
```
---

## **Step 1: Prepare the Dataset**

Before training, you need to extract **frames** and **faces** from videos.

### **1. Extract Frames from Videos**

- Place your videos inside `data/videos/`.
- Extract frames using:

```bash
python extract_frames.py --input data/videos/ --output data/extracted_frames/
```

- Extracted frames will be stored in `data/extracted_frames/`.

### **2. Extract Faces from Frames**

- Detect and crop faces from frames:

```bash
python extract_faces.py --input data/extracted_frames/ --output data/extracted_faces/
```

- Cropped faces will be saved in `data/extracted_faces/`.

---

## **Step 2: Train the Swin Transformer Model**

The **Swin Transformer** extracts features from the detected faces.

### **1. Ensure Dataset is Ready**

Organize `data/train_faces/` as follows:

```
data/train_faces/
├── real/
│   ├── face_1.jpg
│   ├── face_2.jpg
│   └── ...
├── fake/
│   ├── face_1.jpg
│   ├── face_2.jpg
│   └── ...
```

### **2. Train the Swin Model**

Run the training script:

```bash
python train_swin.py
```

This will train the Swin Transformer and save the model weights to:

```
models/swin_model_custom.pth
```

---



## **Step 3: Extract Features Using Swin Transformer**

Extract features from the trained Swin model:

```bash
python swin_feature_extraction.py --input data/extracted_faces/ --output dataset/extracted_features/
```

- This will generate two files:
  - `dataset/extracted_features/real.pt` → Stores all real face features.
  - `dataset/extracted_features/fake.pt` → Stores all fake face features.

---

## **Step 4: Train the LSTM Model**

The **LSTM model** takes extracted features and classifies faces as **real** or **deepfake**.

### **1. Ensure Extracted Features Exist**

`dataset/extracted_features/real.pt` and `dataset/extracted_features/fake.pt` should be available.

### **2. Train the LSTM Model**

Run the training script:

```bash
python train_lstm.py
```

This will train the LSTM model and save it as:

```
models/lstm_model_custom.pth
```

---

## **Step 5: Test and Inference**

After training, you can test the model on a new video.

### **1. Prepare a New Test Video**
- Place the video inside `data/videos/test_video.mp4`.

### **2. Extract Frames from the Test Video**
```bash
python extract_frames.py --input data/videos/test/ --output data/test_frames/
```

### **3. Extract Faces from Frames**
```bash
python extract_faces.py --input data/test_frames/ --output data/test_faces/
```

### **4. Run Deepfake Detection**
```bash
python detect.py --input data/test_faces/
```

### **5. Get Video-Level Classification**
- The `detect.py` script will classify each face as **"Real"** or **"Deepfake"**.
- If the majority of detected faces are classified as **"Deepfake"**, the entire video is labeled as deepfake.

Example:
```python
import os
from detect import detect_deepfake

test_faces_folder = "data/test_faces/"
deepfake_count = sum(1 for face in os.listdir(test_faces_folder) if detect_deepfake(os.path.join(test_faces_folder, face)) == "Deepfake")
total_faces = len(os.listdir(test_faces_folder))

detection_ratio = deepfake_count / total_faces if total_faces > 0 else 0
video_label = "Deepfake" if detection_ratio > 0.5 else "Real"
print(f"Video Classification: {video_label}")
``` 

---
Step 6: Model Evaluation

To assess the performance of the trained deepfake detection model, you need to evaluate it on a labeled test dataset.

1. Prepare a Test Dataset

Ensure the test dataset contains both real and deepfake faces.

The extracted features for the test dataset should be in:
```
dataset/test_features/
├── real.pt     # Pre-extracted real face features
├── fake.pt     # Pre-extracted deepfake face features

```
---

2. Run Model Evaluation

Run the evaluation script:
```bash
python evaluate.py
```

This will compute the model's accuracy, precision, recall, and F1-score and display the results.

### **License**
This project is open-source and free to use.

