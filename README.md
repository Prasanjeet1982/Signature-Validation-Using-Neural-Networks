# Signature Validation Using Neural Networks

This project aims to develop a signature validation system using neural networks in Python. The system distinguishes between genuine and forged signatures by analyzing signature images.

## Overview

The project includes several components:

- **Data Collection**: Gathered a diverse dataset of signature images (genuine and forged).
- **Preprocessing**: Resized, normalized, and prepared the images for model training.
- **Model Development**: Constructed a Convolutional Neural Network (CNN) architecture for signature validation.
- **Training and Evaluation**: Trained the model on the training set and evaluated its performance on validation and test sets.
- **Model Deployment**: Created a simple API for signature validation using Flask.

## File Structure

- `data_collection.py`: Script to collect signature images.
- `preprocessing.py`: File containing image preprocessing techniques.
- `split_dataset.py`: Script to split the dataset into training, validation, and test sets.
- `model.py`: Python file defining the CNN architecture for signature validation.
- `train_model.py`: Script for training the neural network on the training set.
- `evaluate_model.py`: File to assess the model's performance on the validation set.
- `test_model.py`: Script to test the final model on the test set.
- `signature_validation.py`: Python file implementing the signature validation API using Flask.
- `augmentation.py`: File containing data augmentation techniques.
- `improved_model.py`: Script for enhancing the existing model architecture.
- `utils.py`: Helper functions used in model training and evaluation.
- `README.md`: This file, providing an overview of the project.

## Usage

1. **Data Collection**: Run `data_collection.py` to gather signature images.
2. **Preprocessing**: Utilize `preprocessing.py` to preprocess the collected images.
3. **Model Development**: Use `model.py` to define the neural network architecture.
4. **Training**: Execute `train_model.py` to train the model on the training set.
5. **Evaluation**: Run `evaluate_model.py` to assess the model's performance on the validation set.
6. **Testing**: Use `test_model.py` to test the final model on the test set.
7. **Deployment**: Execute `signature_validation.py` to launch the signature validation API.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- Flask
- NumPy
- OpenCV (for image processing)

Ensure the required dependencies are installed before running the scripts or deploying the API.

## Notes

- Adjust hyperparameters, model architecture, or file paths as needed.
- Additional improvements, data augmentation, or model fine-tuning can be explored for better performance.
- Ensure ethical considerations and privacy guidelines are followed when handling signature data.
