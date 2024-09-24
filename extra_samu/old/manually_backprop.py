import torch
import torch.nn as nn

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(3, 5)  # Input layer to hidden layer (3 -> 5)
        self.fc2 = nn.Linear(5, 2)  # Hidden layer to output layer (5 -> 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after fc1
        x = self.fc2(x)  # No activation on final layer
        return x

# Initialize the network
net = SimpleNN()

# Create a random input tensor (batch_size=1, input_dim=3)
input_tensor = torch.randn(1, 3, requires_grad=True)

# Forward pass
output = net(input_tensor)

# Set custom gradients for the output (let's say we want dL/d(output) = [1.0, 0.5] for each output element)
grad_output = torch.tensor([[1.0, 0.5]])

# Perform backward pass using the custom gradients
output.backward(grad_output)

# Extract the input gradients
input_gradients = input_tensor.grad

print("Input Tensor:")
print(input_tensor)

print("\nOutput Tensor:")
print(output)

print("\nInput Gradients (dL/d(input)):")
print(input_gradients)
