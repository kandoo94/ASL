# real_time_predict.py

import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
import time

# Load trained model
with open("models/asl_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands

# Initialize the running text variable and timer
detected_text = ""
last_detection_time = 0  # Track the time of the last letter detection
DETECTION_INTERVAL = 5  # Time interval between detections in seconds

def get_hand_landmarks(image):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

def predict_class(frame):
    landmarks = get_hand_landmarks(frame)
    if landmarks is not None:
        return model.predict([landmarks])[0]
    return None

def update_video():
    global detected_text, last_detection_time
    
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_video)
        return
    
    # Predict the letter based on the hand landmarks
    current_time = time.time()
    if current_time - last_detection_time >= DETECTION_INTERVAL:
        predicted_letter = predict_class(frame)
        if predicted_letter:
            # Append the detected letter to the running text if a prediction is made
            detected_text += predicted_letter
            last_detection_time = current_time  # Reset the timer
            if len(detected_text) > 30:  # Limit length to keep it manageable on screen
                detected_text = detected_text[-30:]

    # Display the detected text on the frame
    cv2.putText(frame, f"Detected Text: {detected_text}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame in OpenCV window
    cv2.imshow("ASL Translator", frame)

    # Update Tkinter label with the detected text
    text_label.config(text=f"Detected Text: {detected_text}")

    # Schedule the next update
    root.after(10, update_video)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up Tkinter GUI
root = tk.Tk()
root.title("ASL Real-Time Translator")

# Label to show detected text in the GUI
text_label = tk.Label(root, text="Detected Text:", font=("Arial", 20))
text_label.pack()

# Start updating the video feed
root.after(10, update_video)

# Close everything on exit
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), cv2.destroyAllWindows(), root.destroy()))
root.mainloop()
