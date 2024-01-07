from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
model = load_model('saved_models/signature_validation_model.h5')

# Define a route for signature validation
@app.route('/validate_signature', methods=['POST'])
def validate_signature():
    # Get the image file from the request
    file = request.files['file']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image (resize, normalize, etc.)
    img = cv2.resize(img, (128, 128))  # Adjust the size to match the model input
    img = img.astype(np.float32) / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))
    
    # Define response based on the prediction
    if prediction[0][0] > 0.5:
        result = "Genuine Signature"
    else:
        result = "Forged Signature"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
