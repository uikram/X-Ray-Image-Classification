from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = r"E:\Deployment\Trained Model\xray.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Modify this according to your model)
CLASS_LABELS = ["COVID19", "NORMAL", "PNEUMONIA"]  # Example labels

# Image preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 channels (RGB)
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        # Read and process the uploaded image
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)

        # Perform prediction
        predictions = model.predict(image)
        class_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))

        # Return JSON response
        return jsonify({
            "prediction": CLASS_LABELS[class_idx],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
