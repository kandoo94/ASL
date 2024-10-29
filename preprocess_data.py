# preprocess_data.py

import cv2
import mediapipe as mp
import os
import numpy as np
import pickle

def preprocess_image(img_path):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
        else:
            print(f"No hand landmarks detected in image: {img_path}")  # Debugging line
    return None

def preprocess_dataset(data_path='dataset/asl_alphabet_train'):
    data, labels = [], []
    
    # Traverse each subdirectory in the dataset path
    for class_name in os.listdir(data_path):
        class_dir = os.path.join(data_path, class_name)
        
        # Ensure we're working with directories
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith(".jpg") or img_file.endswith(".png"):  # Adjust for your image format
                    img_path = os.path.join(class_dir, img_file)
                    features = preprocess_image(img_path)
                    if features is not None:
                        data.append(features)
                        labels.append(class_name)
                    else:
                        print(f"Skipped image due to missing features: {img_path}")  # Debugging line
    
    # Save the preprocessed data
    data = np.array(data)
    labels = np.array(labels)
    with open("dataset/preprocessed_data.pkl", "wb") as f:
        pickle.dump((data, labels), f)
    print(f"Data preprocessing complete. Processed {len(data)} samples.")

if __name__ == "__main__":
    preprocess_dataset()