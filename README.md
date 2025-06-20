# 🤖 Neural Network Digit Classifier

A web application that uses a PyTorch neural network to classify handwritten digits (0-9) in real-time.

## Features

- **Interactive Drawing Canvas**: Draw digits with your mouse or finger
- **Real-time Predictions**: AI predicts your drawings instantly
- **Probability Visualization**: See confidence levels for all digits 0-9
- **Mobile-Friendly**: Works on phones and tablets
- **Pre-trained Model**: Uses a neural network trained on the MNIST dataset

## How It Works

1. **Neural Network**: A 3-layer neural network trained on 60,000 handwritten digits
2. **Web Interface**: Flask backend with HTML5 Canvas for drawing
3. **Real-time Processing**: Converts your drawing to 28x28 grayscale and makes predictions

## Local Development

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training script: `python main.py` (creates the model file)
4. Start the web app: `python web_app.py`
5. Open `http://localhost:5000` in your browser

## Deploy to Render

1. Push this code to GitHub
2. Connect your GitHub repo to [Render](https://render.com)
3. Create a new Web Service
4. Render will automatically detect the Flask app and deploy it!

## Files

- `main.py` - Training script and model definition
- `web_app.py` - Flask web application
- `templates/index.html` - Web interface with drawing canvas
- `digit_classifier_model.pth` - Trained model weights
- `requirements.txt` - Python dependencies

## Model Performance

The neural network achieves ~95% accuracy on the MNIST test dataset and works well on hand-drawn digits with proper preprocessing.

## Technologies Used

- **PyTorch** - Neural network framework
- **Flask** - Web framework
- **HTML5 Canvas** - Drawing interface
- **JavaScript** - Client-side interactions 