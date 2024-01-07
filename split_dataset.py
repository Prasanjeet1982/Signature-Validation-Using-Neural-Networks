import os
import shutil
import random

def split_dataset(input_folder, output_folder, train_split=0.7, val_split=0.15, test_split=0.15):
    # Create output folders if they don't exist
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all image filenames in the input folder
    images = os.listdir(input_folder)
    random.shuffle(images)  # Shuffle the images randomly

    num_images = len(images)
    num_train = int(train_split * num_images)
    num_val = int(val_split * num_images)

    # Split images into train, validation, and test sets
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    # Copy images to respective folders
    copy_images(input_folder, train_images, train_folder)
    copy_images(input_folder, val_images, val_folder)
    copy_images(input_folder, test_images, test_folder)

def copy_images(input_folder, image_list, output_folder):
    for img_name in image_list:
        img_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)
        shutil.copy(img_path, output_path)

# Example usage:
if __name__ == "__main__":
    input_dataset_folder = "signature_data"
    output_split_folder = "split_dataset"

    split_dataset(input_dataset_folder, output_split_folder)
