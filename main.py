import torch
import torch.nn as nn
from jaxtyping import Float
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

torch.manual_seed(0)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class digitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.final_layer = nn.Linear(512, 10)
        # Remove sigmoid - CrossEntropyLoss expects raw logits
    
    def forward(self, images: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        x = self.first_layer(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final_layer(x)
        return x  # Return raw logits for CrossEntropyLoss

# Initialize model, loss function, and optimizer
model = digitClassifier()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Model file path
model_path = "digit_classifier_model.pth"

# Check if trained model exists
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!")
    skip_training = True
else:
    print("No pre-trained model found. Starting training...")
    skip_training = False

# Training loop (only if no saved model exists)
if not skip_training:
    epochs = 5
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.view(images.shape[0], 784)
            
            # TRAINING BODY
            model_prediction = model(images)
            optimizer.zero_grad()
            loss = loss_function(model_prediction, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_dataloader):.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Evaluation
model.eval()
with torch.no_grad():  # Disable gradient computation for evaluation
    for images, labels in test_dataloader:
        images = images.view(images.shape[0], 784)
        
        model_prediction = model(images)
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(model_prediction, dim=1)
        max_probs, predicted_indices = torch.max(probabilities, dim=1)
        
        # Show first few examples
        for i in range(min(5, len(images))):
            plt.figure(figsize=(8, 4))
            
            # Plot image
            plt.subplot(1, 2, 1)
            plt.imshow(images[i].view(28, 28), cmap='gray')
            plt.title(f'True Label: {labels[i].item()}')
            plt.axis('off')
            
            # Plot probabilities
            plt.subplot(1, 2, 2)
            plt.bar(range(10), probabilities[i].numpy())
            plt.xlabel('Digit')
            plt.ylabel('Probability')
            plt.title(f'Predicted: {predicted_indices[i].item()} (Prob: {max_probs[i].item():.3f})')
            plt.xticks(range(10))
            
            plt.tight_layout()
            plt.show()
            
            print(f"True label: {labels[i].item()}, Predicted: {predicted_indices[i].item()}")
            print(f"Probabilities: {probabilities[i].numpy().round(3)}")
            print("-" * 50)
        
        break  # Only show first batch

# Function to test with your own image
def predict_custom_image(image_path, invert_colors=False):
    """
    Load and predict a custom image of a handwritten digit.
    
    Args:
        image_path (str): Path to your image file
        invert_colors (bool): Set to True if your image has dark digits on white background
        
    The image should be:
    - A handwritten digit (0-9)
    - Preferably on a white background with dark digit
    - Any size (will be resized to 28x28)
    """
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        
        # Convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Store original for display
        original_array = img_array.copy()
        
        # Invert colors if needed (MNIST has white digits on black background)
        if invert_colors:
            img_array = 255 - img_array
            print("Colors inverted: dark digits on white -> white digits on dark")
        
        # Normalize to 0-1 range
        img_array = img_array.astype(np.float32) / 255.0
        
        # Apply same normalization as training data
        img_array = (img_array - 0.1307) / 0.3081
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.tensor(img_array).unsqueeze(0).view(1, 784)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(img_tensor)
            probabilities = torch.softmax(prediction, dim=1)
            predicted_digit = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        # Display results
        plt.figure(figsize=(12, 4))
        
        # Show original image
        plt.subplot(1, 4, 1)
        plt.imshow(original_array, cmap='gray')
        plt.title('Your Original Image')
        plt.axis('off')
        
        # Show after color inversion (if applied)
        if invert_colors:
            plt.subplot(1, 4, 2)
            plt.imshow(255 - original_array, cmap='gray')
            plt.title('After Color Inversion')
            plt.axis('off')
        else:
            plt.subplot(1, 4, 2)
            plt.imshow(original_array, cmap='gray')
            plt.title('No Color Inversion')
            plt.axis('off')
        
        # Show processed image (how the model sees it)
        plt.subplot(1, 4, 3)
        # Denormalize for display
        display_array = (img_array * 0.3081) + 0.1307
        plt.imshow(display_array, cmap='gray')
        plt.title('Model Input (28x28)')
        plt.axis('off')
        
        # Show prediction probabilities
        plt.subplot(1, 4, 4)
        plt.bar(range(10), probabilities[0].numpy())
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title(f'Prediction: {predicted_digit} (Conf: {confidence:.3f})')
        plt.xticks(range(10))
        
        plt.tight_layout()
        plt.show()
        
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.3f}")
        print(f"All probabilities: {probabilities[0].numpy().round(3)}")
        
        # Show top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        print(f"\nTop 3 predictions:")
        for i in range(3):
            print(f"  {top3_indices[i].item()}: {top3_probs[i].item():.3f}")
        
        return predicted_digit, confidence, probabilities[0].numpy()
        
    except Exception as e:
        print(f"Error processing image: {e}")
        print("Make sure the image file exists and is a valid image format (PNG, JPG, etc.)")
        return None, None, None

# Example usage (uncomment and modify the path to test with your own image):
print("Testing your image WITHOUT color inversion:")
predict_custom_image("test1.png", invert_colors=False)

print("\n" + "="*60)
print("Testing your image WITH color inversion:")
predict_custom_image("test1.png", invert_colors=True)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("To test with your own image, use:")
print('predict_custom_image("path/to/your/image.png")')
print("\nTips for best results:")
print("- Draw a single digit (0-9) clearly")
print("- Use a white background with dark digit")
print("- Save as PNG, JPG, or other common image format")
print("- Any size is fine (will be resized to 28x28)")
print("="*60)








