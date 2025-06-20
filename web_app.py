import os
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import base64
import io
from jaxtyping import Float

app = Flask(__name__)

# Define the same model architecture
class digitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.final_layer = nn.Linear(512, 10)
    
    def forward(self, images: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        x = self.first_layer(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final_layer(x)
        return x

# Load the trained model
model = digitClassifier()
model.load_state_dict(torch.load("digit_classifier_model.pth", map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image_data):
    """Process the canvas drawing for prediction"""
    # Decode base64 image
    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # No need to invert colors - canvas already draws white on black like MNIST
    
    # Normalize
    img_array = img_array.astype(np.float32) / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    
    # Convert to tensor
    img_tensor = torch.tensor(img_array).unsqueeze(0).view(1, 784)
    
    return img_tensor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Preprocess the image
        img_tensor = preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(img_tensor)
            probabilities = torch.softmax(prediction, dim=1)
            predicted_digit = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        # Convert probabilities to list for JSON
        probs_list = probabilities[0].numpy().tolist()
        
        return jsonify({
            'predicted_digit': predicted_digit,
            'confidence': float(confidence),
            'probabilities': probs_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 