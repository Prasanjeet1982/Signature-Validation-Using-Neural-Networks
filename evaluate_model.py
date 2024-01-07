import numpy as np
from tensorflow.keras.models import load_model
from utils import evaluate_model

# Load the preprocessed validation data
x_val = np.load('preprocessed_data/val_images.npy')
y_val = np.load('preprocessed_data/val_labels.npy')

# Load the trained model
model = load_model('saved_models/signature_validation_model.h5')

# Evaluate the model on the validation set
evaluate_model(model, (x_val, y_val))
