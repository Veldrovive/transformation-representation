"""
In the readme, we call the network defined in this file "eta".
It is meant to embed images into a space where images transformed by the same transformation are close together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTransEmbedder(nn.Module):
    """
    A standard convolutional neural network that takes in an image and outputs a vector.
    """
    def __init__(self):
        super(ConvTransEmbedder, self).__init__()
        # Images start at 224x224x3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # 224x224x32
        self.pool1 = nn.MaxPool2d(2, 2) # 112x112x32

        self.conv2 = nn.Conv2d(32, 64, 5, padding=1) # 112x112x64
        self.pool2 = nn.MaxPool2d(2, 2) # 55x55x64

        self.conv3 = nn.Conv2d(64, 64, 5, padding=1) # 56x56x64
        self.pool3 = nn.MaxPool2d(2, 2) # 26x26x64

        self.fc1 = nn.Linear(26 * 26 * 64, 1024)
        self.fc2 = nn.Linear(1024, 128)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Gamma(nn.Module):
    """
    Gamma takes a vector of size (n, 128) and outputs a vector of size (128) by applying a symmetric function to the input.
    In this case we use the element-wise average function of the input vectors
    """
    def __init__(self):
        super(Gamma, self).__init__()
        self.fc1 = nn.Linear(128, 128)  # Represents the combination rule for the vectors

    def forward(self, x):
        x = self.fc1(x)
        return torch.mean(x, dim=-2)
    
class AttentionGamma(nn.Module):
    """
    Uses a self-attention layer instead of a linear layer to provide the combination rule for the vectors.
    Still averages the outputs of the attention to ensure a symmetric function.
    """
    def __init__(self):
        super(AttentionGamma, self).__init__()
        self.fc1 = nn.Linear(128, 128)  # Represents the combination rule for the vectors
        self.attn = nn.MultiheadAttention(128, 1)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.fc1(x)
        residual = x
        x = x.permute(1, 0, 2)
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)
        # Concatenate the original input with the attention output
        x = torch.cat((x, residual), dim=-1)
        x = self.fc2(x)
        return torch.mean(x, dim=-2)
    
class AvgGamma(nn.Module):
    """
    This gamma expects input to be (n, 128) and just returns the average of the vectors
    """
    def __init__(self):
        super(AvgGamma, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=-2)

class IdentityGamma(nn.Module):
    """
    This gamma expects input to be (1, 128) and just returns the first element
    """
    def __init__(self):
        super(IdentityGamma, self).__init__()

    def forward(self, x):
        return x[0]