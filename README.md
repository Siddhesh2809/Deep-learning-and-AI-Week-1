# Deep Learning - Week 1 Assignment & Mini Project

## Assignment 1: Understanding and Training a Neural Network

### Q1: Theoretical Concepts
* **Neuron (Perceptron):** The fundamental unit of a neural network that accepts inputs, applies weights and bias, and produces an output via an activation function.
* **Weights:** Adjustable parameters that determine the importance/strength of each input feature.
* **Biases:** An extra trainable parameter added to shift the activation function for better data fitting.
* **Epochs:** One complete pass of the entire training dataset through the neural network.
* **Batches:** A small subset of the training dataset processed together in one iteration.
* **Learning Rate:** A hyperparameter controlling the step size of weight updates during optimization.

---

---

### Q2: Tensor Operations
```python
import torch

# Tensor Creation & Reshaping
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print("Reshaped X:\n", x.reshape(4, 1))
print("Matrix Multiplication:\n", torch.matmul(x, y))
```

---

### Q3: Train a Simple Model on Synthetic Regression Data
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data
torch.manual_seed(42)
X = torch.randn(100, 1)
y = 2 * X + 3 + torch.randn(100, 1) * 0.2

# 2. Simple Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Training function
def train_model(learning_rate, epochs=100):
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for epoch in range(epochs):
        predictions = model(X)
        loss = criterion(predictions, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
    return loss_history

# Train with base learning rate
losses = train_model(learning_rate=0.01)
print("Training completed. Final Loss:", losses[-1])
```
### Q4: Plot Training and Validation Loss
```python
import matplotlib.pyplot as plt

# Plotting Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
```
---

### Q5: Compare Results Using Two Learning Rates
```python
# Train with two different learning rates
losses_lr_001 = train_model(learning_rate=0.01)
losses_lr_01 = train_model(learning_rate=0.1)

# Plot Comparison
plt.figure(figsize=(8, 5))
plt.plot(losses_lr_001, label='Learning Rate = 0.01')
plt.plot(losses_lr_01, label='Learning Rate = 0.1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve Comparison Across Learning Rates')
plt.legend()
plt.grid(True)
plt.show()
```
**Key Observations (Q5 Comparison):**

* Higher Learning Rate (0.1): Convergence is faster, reducing the loss in fewer epochs.

* Lower Learning Rate (0.01): Loss decreases more gradually and smoothly over epochs.
