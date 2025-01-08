import torch
from torch import nn

class CNN_Network(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, output_dim):
        super(CNN_Network, self).__init__()
        
        # CNN layer
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=9, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Fully connected layer
        self.fc1 = nn.Linear(1568, 780)
        self.fc2 = nn.Linear(780, 60)
        self.fc3 = nn.Linear(61, output_dim)
    
    def forward(self, x):
        # Pass through CNN

        speed = x[:,-1]
        x = x[:,:-1].unsqueeze(1)  # Add channel dimension

        x1 = self.cnn1(x)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.cnn2(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = nn.Dropout(0.5)(x1)
        
        x1 = x1.reshape(x1.shape[0], -1)
        speed = speed.unsqueeze(-1)
        
        # Pass through fully connected layer
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)
        x1 = torch.cat((x1, speed),axis=1)
        x1 = self.fc3(x1)
        return x1
