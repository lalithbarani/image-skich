

from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import io

# Create a Flask application
app = Flask(__name__)
CORS(app)  

def pencil_sketch_high_quality(img):
    try:
        if img is None:
            print("Error: No image data")
            return None
        
        # Convert image to grayscale
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert grayscale image
        inverted_img = 255 - grayscale_img
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted_img, (21, 21), 0)
        
        # Invert blurred image
        inverted_blurred = 255 - blurred
        
        # Create pencil sketch
        pencil_sketch = cv2.divide(grayscale_img, inverted_blurred, scale=256.0)
        
        # Adding pencil-like strokes
        height, width = pencil_sketch.shape[:2]
        pencil_marks = np.zeros_like(img)

        for i in range(0, height, 10):
            for j in range(0, width, 10):
                cv2.line(pencil_marks, (j, i), (j + 5, i + 5), (50, 50, 50), 2)

        # Resize pencil_marks to match pencil_sketch if needed
        pencil_marks_resized = cv2.resize(pencil_marks, (width, height))
        
        # Ensure both images are in the same color format (grayscale)
        pencil_marks_resized = cv2.cvtColor(pencil_marks_resized, cv2.COLOR_BGR2GRAY)

        # Blend pencil marks with pencil sketch
        enhanced_sketch = cv2.addWeighted(pencil_sketch, 0.8, pencil_marks_resized, 0.2, 0)
        
        return enhanced_sketch
    
    except Exception as e:
        print(f"Error: {e}")
        return None


@app.route('/process', methods=['POST'])
@cross_origin()
def process_image():
    try:
        if 'image' not in request.files:
            return 'No image uploaded', 400
        
        image_file = request.files['image']
        
        img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
        processed_image = pencil_sketch_high_quality(img)
        
        if processed_image is None:
            return 'Error processing image', 500
        
     
        _, img_encoded = cv2.imencode('.jpg', processed_image)
        img_bytes = io.BytesIO(img_encoded)
        
   
        return send_file(img_bytes, mimetype='image/jpeg')
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return str(e), 500
    
# Run the application
if __name__ == '__main__':
    app.run()
