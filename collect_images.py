import cv2
import os

# Define paths for saving images
save_path = 'dataset/asl_alphabet_train'
os.makedirs(save_path, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)
class_name = input("Enter the label/class name for this capture session: ")

# Set image count and limit
image_count = 0
image_limit = 200  # Number of images to capture per class

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display instructions
    cv2.putText(frame, f"Class: {class_name} | Count: {image_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("ASL Data Collection", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s') and image_count < image_limit:
        # Save the current frame as an image file
        filename = os.path.join(save_path, f"{class_name}_{image_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        image_count += 1
    elif key & 0xFF == ord('q') or image_count >= image_limit:
        break

cap.release()
cv2.destroyAllWindows()
