# Intuitive Human-Computer Interaction via Hand Gesture Recognition

This project implements a real-time hand gesture recognition system for HCI (Human-Computer Interaction). It utilizes a custom-trained **Convolutional Neural Network (CNN)** on the LeapGestureRecog dataset and compares it with a geometric **MediaPipe Landmark** approach.

## 🚀 Features
* **Dual Approach:** Includes both Deep Learning (CNN) and Geometric (MediaPipe) detection.
* **Real-time Control:** Map gestures to system commands like Volume Up/Down and Play/Pause.
* **High Accuracy:** CNN trained on 20,000 near-infrared images.

## 📁 File Structure
- `data/`: Contains the LeapGestureRecog dataset.
- `models/`: Stores the trained `.h5` model.
- `src/train.py`: Script for data preprocessing and CNN training.
- `src/app_cnn.py`: Real-time gesture control using the trained CNN.
- `src/app_mediapipe.py`: Gesture control using MediaPipe landmarks.

## 🛠️ Installation & Setup
1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt