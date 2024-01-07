import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

def extract_features(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Load pre-trained VGG16 model without top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Create a new model by taking the output of the last convolutional layer
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
    
    images = os.listdir(input_folder)
    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Resize image to VGG input size
        img = img.astype(np.float32) / 255.0  # Normalize pixel values
        
        # Expand dimensions to match model input shape (batch size of 1)
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        features = feature_extractor.predict(img)
        
        # Save extracted features to a file
        features_path = os.path.join(output_folder, img_name.replace('.png', '.npy'))
        np.save(features_path, features)

# Example usage:
if __name__ == "__main__":
    input_images_folder = "preprocessed_data"
    output_features_folder = "extracted_features"

    extract_features(input_images_folder, output_features_folder)
