import cv2
import numpy as np
import csv
import os
import time

# Import MediaPipe with error handling
try:
    import mediapipe as mp
    mp_solutions = mp.solutions
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except (ImportError, AttributeError) as e:
    print(f"\nCRITICAL ERROR DETAILS: {e}")
    print("Error: Failed to import MediaPipe solutions.")
    print("This is likely due to a version conflict with TensorFlow, especially the 'protobuf' library.")
    print("\nPlease fix your environment by running these commands in your activated '(tfenv)' terminal:")
    print("1. python -m pip uninstall -y protobuf mediapipe tensorflow")
    print("2. python -m pip install protobuf==3.20.3")
    print("3. python -m pip install tensorflow")
    print("4. python -m pip install mediapipe==0.10.9")
    exit(1)

# --- Configuration ---
# The CSV file where the collected landmark data will be saved
CSV_FILE = 'keypoint_data.csv' 
NUM_SAMPLES = 100 # Number of frames to record for each gesture
GESTURE_LABELS = [
    "Open_Palm", "Closed_Fist", "Pointing_Up", 
    "Thumbs_Up", "Thumbs_Down", "Victory",
    "Peace", "Shaka", "Rock"
]
# Create a dictionary to map key presses to class IDs (0, 1, 2, ...)
KEY_TO_ID = {str(i): i for i in range(len(GESTURE_LABELS))}

# Generate the label file automatically to ensure consistency
with open('gesture_labels.csv', 'w', newline='', encoding='utf-8') as f:
    f.write('\n'.join(GESTURE_LABELS))

# Initialize MediaPipe Hands from the imported module
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Helper Function (Same as in app.py) ---
def normalize_landmarks(hand_landmarks):
    """Converts 21 landmarks into a normalized 1D array (42 values)."""
    landmark_list = []
    if not hand_landmarks:
        return None
    
    # Extract coordinates
    for landmark in hand_landmarks.landmark:
        # We only use X and Y, making it scale and location invariant relative to the wrist
        landmark_list.append(landmark.x)
        landmark_list.append(landmark.y)
    
    # Normalize by the wrist (first landmark - index 0 and 1) coordinates
    base_x = landmark_list[0]
    base_y = landmark_list[1]
    
    normalized_landmarks = []
    for i in range(0, len(landmark_list), 2):
        # Subtract wrist coordinates from all other coordinates
        normalized_landmarks.append(landmark_list[i] - base_x)
        normalized_landmarks.append(landmark_list[i+1] - base_y)
        
    return normalized_landmarks

# --- Main Data Collection Logic ---
def collect_data():
    """Captures video, detects landmarks, and saves them to a CSV file."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not detected")
        return    
    # Check if CSV file exists and write header if it's new
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header: 'class_id', 'x1', 'y1', 'x2', 'y2', ...
            header = ['class_id'] + [f'L{i}_{coord}' for i in range(21) for coord in ['x', 'y']]
            writer.writerow(header)
        print(f"Created new data file: {CSV_FILE}")

    mode = 'Waiting'
    class_id = -1
    sample_count = 0
    
    print("\n--- DATA COLLECTION MODE ---")
    print("Press a number key (0-8) to start recording a gesture.")
    for key, label in enumerate(GESTURE_LABELS):
        print(f"  [{key}] : {label}")
    print("Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Display Current Mode and Instructions
        cv2.putText(image, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Handle Hand Detection and Data Logging
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                normalized_features = normalize_landmarks(hand_landmarks)

                if mode == 'Recording' and normalized_features is not None:
                    # 1. Prepare data row: [class_id, x1, y1, x2, y2, ...]
                    data_row = [class_id] + normalized_features
                    
                    # 2. Append to CSV
                    with open(CSV_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data_row)
                    
                    sample_count += 1
                    
                    # 3. Update status on screen
                    current_label = GESTURE_LABELS[class_id]
                    status_text = f"Recording {current_label}: {sample_count}/{NUM_SAMPLES}"
                    cv2.putText(image, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    if sample_count >= NUM_SAMPLES:
                        print(f"\n--- DONE: Collected {NUM_SAMPLES} samples for {current_label} (ID {class_id}) ---")
                        mode = 'Waiting'
                        class_id = -1
                        sample_count = 0
                        time.sleep(1) # Pause to let user reset hand
        
        cv2.imshow('Gesture Data Collector', image)

        key = cv2.waitKey(1) & 0xFF
        
        # Check for 'q' to quit
        if key == ord('q'):
            break

        # Check for number keys to start recording
        if key != 255:
            try:
                if chr(key).isdigit() and chr(key) in KEY_TO_ID:
                    new_id = KEY_TO_ID[chr(key)]
                    if new_id < len(GESTURE_LABELS):
                        class_id = new_id
                        mode = 'Recording'
                        sample_count = 0
                        print(f"-> STARTING COLLECTION for ID {class_id}: {GESTURE_LABELS[class_id]}")
            except ValueError:
                pass
    
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_data()
