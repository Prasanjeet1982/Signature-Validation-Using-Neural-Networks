import os
import cv2
import numpy as np

def resize_images(input_folder, output_folder, target_size=(128, 128)):
    os.makedirs(output_folder, exist_ok=True)
    images = os.listdir(input_folder)
    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, target_size)
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, resized_img)

def normalize_images(input_folder):
    images = os.listdir(input_folder)
    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path).astype(np.float32)
        img /= 255.0  # Normalize pixel values between 0 and 1
        cv2.imwrite(img_path, img * 255)  # Save the normalized image

# Example usage:
if __name__ == "__main__":
    input_genuine_folder = "signature_data/genuine"
    output_genuine_folder_resized = "preprocessed_data/genuine_resized"
    output_genuine_folder_normalized = "preprocessed_data/genuine_normalized"

    input_forged_folder = "signature_data/forged"
    output_forged_folder_resized = "preprocessed_data/forged_resized"
    output_forged_folder_normalized = "preprocessed_data/forged_normalized"

    # Resize images
    resize_images(input_genuine_folder, output_genuine_folder_resized)
    resize_images(input_forged_folder, output_forged_folder_resized)

    # Normalize images
    normalize_images(output_genuine_folder_resized)
    normalize_images(output_forged_folder_resized)
