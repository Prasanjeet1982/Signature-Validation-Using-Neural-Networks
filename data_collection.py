import os
import cv2  # Using OpenCV for image processing
import shutil  # For file operations
import numpy as np

# Define paths
data_folder = "signature_data"
genuine_folder = os.path.join(data_folder, "genuine")
forged_folder = os.path.join(data_folder, "forged")

# Create folders if they don't exist
os.makedirs(genuine_folder, exist_ok=True)
os.makedirs(forged_folder, exist_ok=True)

# Function to capture signature images
def capture_signature(camera_id, output_folder, num_images):
    cap = cv2.VideoCapture(camera_id)
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        cv2.imshow("Capture Signature", frame)
        key = cv2.waitKey(1)
        if key == ord("s"):
            img_name = f"signature_{count}.png"
            img_path = os.path.join(output_folder, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Signature {count + 1}/{num_images} saved as {img_name}")
            count += 1
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# Collect genuine signatures (assuming camera ID 0 and capturing 10 images)
print("Capture genuine signatures. Press 's' to save, 'q' to quit.")
capture_signature(0, genuine_folder, 10)

# Collect forged signatures (assuming camera ID 1 and capturing 10 images)
print("Capture forged signatures. Press 's' to save, 'q' to quit.")
capture_signature(1, forged_folder, 10)

# Optionally, perform additional preprocessing or checks on the collected data
# For example, you might want to resize images, convert to grayscale, etc.
