import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import plot_training_history, evaluate_model

def create_improved_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),  # Additional Dense layer
        Dropout(0.3),  # Additional Dropout layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load preprocessed training and validation data
    x_train = np.load('preprocessed_data/train_images.npy')
    y_train = np.load('preprocessed_data/train_labels.npy')
    x_val = np.load('preprocessed_data/val_images.npy')
    y_val = np.load('preprocessed_data/val_labels.npy')

    input_shape = (128, 128, 3)  # Input shape based on preprocessed image size

    # Create an improved model
    improved_model = create_improved_model(input_shape)

    # Define callbacks for model training
    checkpoint = ModelCheckpoint('improved_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the improved model
    history = improved_model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val),
                                 callbacks=[checkpoint, early_stopping])

    # Plot training history
    plot_training_history(history)

    # Load the best saved model and evaluate on the validation set
    best_model = load_model('improved_model.h5')
    evaluate_model(best_model, (x_val, y_val))
