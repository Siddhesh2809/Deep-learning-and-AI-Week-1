import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define 3-Layer Feedforward Neural Network
class DigitRecognizerNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),  # Input layer (28x28 images flattened)
            nn.ReLU(),             # ReLU activation function
            nn.Linear(128, 64),   # Hidden layer
            nn.ReLU(),             # ReLU activation function
            nn.Linear(64, 10)     # Output layer (10 classes: digits 0-9)
        )

    def forward(self, x):
        return self.network(x)

# 2. Instantiate Model, Loss Function, and Optimizer
model = DigitRecognizerNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model Architecture Successfully Created:")
print(model)
