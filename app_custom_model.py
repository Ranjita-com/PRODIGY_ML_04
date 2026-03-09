import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
import screen_brightness_control as sbc
import pyautogui
import time

# --- Setup Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Note: Ensure these paths match where train_custom.py saved your files
MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'gesture_model_custom.h5')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'scaler.pkl')
LABELS_PATH = os.path.join(PROJECT_ROOT, 'src', 'gesture_labels.csv')

# --- Load Resources ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print(f"Error: Model or Scaler not found.")
    print(f"Looking for: {MODEL_PATH}")
    print("Run train_custom.py first.")
    exit()

print("Loading model and scaler...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)#converts the bytes to python object

# Load Labels
gesture_names = []
if os.path.exists(LABELS_PATH):#if exists the gesture_labels.csv file, then read the labels from it
    with open(LABELS_PATH, 'r') as f:
        gesture_names = [line.strip() for line in f.readlines()]
else:
    print("Warning: Labels file not found. Using indices.")

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Helper Function (Must match collect_data.py) ---
def normalize_landmarks(hand_landmarks):
    """Converts 21 landmarks into a normalized 1D array (42 values)."""
    landmark_list = []
    # Extract coordinates
    for landmark in hand_landmarks.landmark:
        landmark_list.append(landmark.x)
        landmark_list.append(landmark.y)
    
    # Normalize by the wrist (first landmark)
    base_x = landmark_list[0]
    base_y = landmark_list[1]
    
    normalized_landmarks = []
    for i in range(0, len(landmark_list), 2):
        normalized_landmarks.append(landmark_list[i] - base_x)
        normalized_landmarks.append(landmark_list[i+1] - base_y)
        
    return normalized_landmarks

# --- Main Loop ---
cap = cv2.VideoCapture(0)
last_action_time = 0
action_label = "Waiting..."

print("Starting Gesture Control...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip and process
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = "Unknown"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 1. Extract and Normalize
            features = normalize_landmarks(hand_landmarks)
            
            # 2. Scale (using the scaler from training)
            # Reshape to (1, 42) because scaler expects 2D array
            features_array = np.array([features], dtype=np.float32)
            features_scaled = scaler.transform(features_array)
            
            # 3. Predict
            prediction = model.predict(features_scaled, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]

            if confidence > 0.8: # Threshold
                if class_id < len(gesture_names):
                    current_gesture = gesture_names[class_id]
                else:
                    current_gesture = f"ID: {class_id}"

                # --- ACTION MAPPING ---
                current_time = time.time()
                if current_time - last_action_time > 0.5: # Cooldown
                    if current_gesture == "Pointing_Up":
                        # Action: Brightness UP
                        try:
                            current_brightness = sbc.get_brightness()
                            if current_brightness:
                                sbc.set_brightness(min(current_brightness[0] + 10, 100))
                                action_label = "Action: BRIGHTNESS UP"
                                last_action_time = current_time
                        except Exception:
                            pass
                    elif current_gesture == "Thumbs_Down":
                        # Action: Brightness DOWN
                        try:
                            current_brightness = sbc.get_brightness()
                            if current_brightness:
                                sbc.set_brightness(max(current_brightness[0] - 10, 0))
                                action_label = "Action: BRIGHTNESS DOWN"
                                last_action_time = current_time
                        except Exception:
                            pass
                    # ✋ Show Desktop
                    elif current_gesture == "Open_Palm":
                        pyautogui.hotkey('win', 'd')
                        action_label = "Show Desktop"
                
                    # ✌ Volume UP
                    elif current_gesture == "Victory":
                        pyautogui.press("volumeup")
                        action_label = "Volume UP"
                
                    # 🤏 Volume DOWN
                    elif current_gesture == "Closed_Fist":
                        pyautogui.press("volumedown")
                        action_label = "Volume DOWN"
                
                    last_action_time = current_time

    # Display Info
    cv2.putText(frame, f"Gesture: {current_gesture} ({confidence:.2f})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, action_label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Custom Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
