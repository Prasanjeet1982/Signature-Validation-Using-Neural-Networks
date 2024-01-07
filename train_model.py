import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import plot_training_history, evaluate_model

def train_model(model, train_data, val_data, epochs=20, batch_size=32, model_name='signature_validation_model.h5'):
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # Define callbacks for model training
    checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        callbacks=[checkpoint, early_stopping])

    # Plot training history
    plot_training_history(history)

    # Evaluate the trained model on validation set
    evaluate_model(model, (x_val, y_val))

# Example usage:
if __name__ == "__main__":
    # Load preprocessed training and validation data
    x_train = np.load('preprocessed_data/train_images.npy')
    y_train = np.load('preprocessed_data/train_labels.npy')
    x_val = np.load('preprocessed_data/val_images.npy')
    y_val = np.load('preprocessed_data/val_labels.npy')

    # Create or load the model
    # model = create_signature_validation_model(input_shape)
    # Or load an existing model
    # model = load_model('saved_models/existing_model.h5')

    # Train the model
    train_model(model, (x_train, y_train), (x_val, y_val), epochs=20, batch_size=32, model_name='signature_validation_model.h5')
