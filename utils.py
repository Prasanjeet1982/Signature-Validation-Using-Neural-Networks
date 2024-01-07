import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_history(history):
    # Plot training and validation accuracy over epochs
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(model, test_data):
    x_test, y_test = test_data
    
    # Evaluate model on test data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Make predictions
    y_pred = model.predict_classes(x_test)

    # Generate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    cr = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(cr)

# Other utility functions can be added based on specific needs
