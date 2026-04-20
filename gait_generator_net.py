import torch
import torch.nn as nn

import torch
import torch.nn as nn

class OldSimpleFCNN(nn.Module):
    def __init__(self, input_size=3, output_size=204, hidden_size=512):
        super(OldSimpleFCNN, self).__init__()
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class SimpleFCNN(nn.Module):
    def __init__(self, input_size=3, output_size=204, hidden_size=512):
        super(SimpleFCNN, self).__init__()
        # Increased depth slightly to help map the non-linear relationship at low speeds
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),  # LeakyReLU often trains better for signal regression
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size), # Added one more layer for capacity
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    """Residual block with layer normalization for stable training."""
    def __init__(self, hidden_size):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual


class GaitFFTPredictor(nn.Module):
    """
    Improved architecture with residual connections and layer normalization.
    Better for predicting FFT coefficients which have structured output.
    """
    def __init__(self, input_size=3, output_size=204, hidden_size=512, num_blocks=3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_size, output_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x
