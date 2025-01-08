import torch
import torch.nn as nn


class SimpleFCNN(nn.Module):
    def __init__(self, input_size=3, output_size=200, hidden_size=512):
        super(SimpleFCNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Hidden to output layer
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()                          # Activation function

    def forward(self, x):
        # Apply layers with activation function
        x = self.fc1(x)         # Hidden layer
        # x = nn.Dropout(0.5)(x)
        x = self.relu(x)        # Activation function
        x = self.fc2(x)         # Hidden layer
        # x = nn.Dropout(0.5)(x)
        x = self.relu(x)        # Activation function
        x = self.fc3(x)         # Output layer
        return x
