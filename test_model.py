import numpy as np
from tensorflow.keras.models import load_model
from utils import evaluate_model

# Load the preprocessed test data
x_test = np.load('preprocessed_data/test_images.npy')
y_test = np.load('preprocessed_data/test_labels.npy')

# Load the trained model
model = load_model('saved_models/signature_validation_model.h5')

# Evaluate the model on the test set
evaluate_model(model, (x_test, y_test))
