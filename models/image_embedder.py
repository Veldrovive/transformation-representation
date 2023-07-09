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
    
class SimpleImgClassifierLateFusion(nn.Module):
    """
    Takes a transformation embedding and an image and outputs a set of probabilities over all classes.
    Use use a simple convolution network for the backbone and use a late fusion approach to combine the transformation embedding
    and the image embedding deep in the network.
    """
    def __init__(self, num_classes, trans_embedding_size=128):
        super(SimpleImgClassifierLateFusion, self).__init__()
        # images start at 3x224x224
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1) # 32x112x112
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # 64x56x56
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 128x28x28
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 128x14x14
        self.conv5 = nn.Conv2d(128, 128, 3, stride=2, padding=1) # 128x7x7
        self.conv6 = nn.Conv2d(128, 128, 3, stride=2, padding=1) # 128x4x4

        self.fc1 = nn.Linear(4 * 4 * 128, 1024)
        concat_size = 1024 + trans_embedding_size
        self.fc2 = nn.Linear(concat_size, num_classes)
    
    def forward(self, x, y):
        x = F.relu(self.conv1(x)) # 32x112x112
        x = F.relu(self.conv2(x)) # 64x56x56
        x = F.relu(self.conv3(x)) # 128x28x28
        x = F.relu(self.conv4(x)) # 128x14x14
        x = F.relu(self.conv5(x)) # 128x7x7
        x = F.relu(self.conv6(x)) # 128x4x4

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)) # 1024
        x = torch.cat((x, y), dim=1) # 1024 + 128
        x = self.fc2(x) # num_classes
        
        # Return the logits
        return x
    
if __name__ == "__main__":
    print("Testing SimpleImgClassifierLateFusion")
    simp_img_classifier = SimpleImgClassifierLateFusion(10)
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_context = torch.randn(1, 128)
    output = simp_img_classifier(dummy_img, dummy_context)
    print(output.shape)

class ResNetImgClassifierLateFusion(nn.Module):
    """
    Takes a transformation embedding and an image and outputs a set of probabilities over all classes.
    We model the convolutional part after a resnet and use a late fusion approach to combine the transformation embedding
    and the image embedding deep in the network.
    """
    def __init__(self, num_classes, trans_embedding_size=128):
        super(ResNetImgClassifierLateFusion, self).__init__()
        self.num_classes = num_classes
        self.trans_embedding_size = trans_embedding_size

        # Images start at 224x224x3
        raise NotImplementedError

class ResNetClassifierAttentionFusion(nn.Module):
    """
    Takes a transformation embedding and an image and outputs a set of probabilities over all classes.

    The convolutional part of the network is modeled after a resnet.
    For fusion, we introduce the context vector at multiple layers of the network and have an attention layer
    to combine the context vector with the image embedding.

    For some brainstorming with gpt, look here https://chat.openai.com/share/a4bf92a2-09a6-4d8a-89d3-7cf8fbef88b4
    and here https://chat.openai.com/share/3f6eadc6-f4e9-4d74-aeae-e87b8a1e9749
    """
    def __init__(self, num_classes, trans_embedding_size=128):
        super(ResNetClassifierAttentionFusion, self).__init__()
        raise NotImplementedError
    
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
        self.fc1 = nn.Linear(128, 256)  # Represents the combination rule for the vectors
        self.attn = nn.MultiheadAttention(256, 1)
        self.fc2 = nn.Linear(512, 128)
        # self.attn2 = nn.MultiheadAttention(256, 1)
        # self.fc3 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
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