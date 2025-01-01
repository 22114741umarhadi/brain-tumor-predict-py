import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Load the model when the application starts
model = load_model('BrainTumorDetectionModel.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No tumor detected"
    elif classNo == 1:
        return "Tumor detected"

def process_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((64, 64))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Expand dimensions
    input_img = np.expand_dims(image_array, axis=0)
    
    # Get prediction
    predict_x = model.predict(input_img)
    result = np.argmax(predict_x, axis=1)
    
    return result[0]

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Brain tumor detection API is running'
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'message': 'Please provide an image file in the request'
            }), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        
        if not image_bytes:
            return jsonify({
                'error': 'Empty image file',
                'message': 'The provided image file is empty'
            }), 400

        # Process the image and get prediction
        prediction = process_image(image_bytes)
        result = get_className(prediction)
        
        # Return prediction result
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence_class': int(prediction),
            'message': 'Image processed successfully'
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Processing error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)