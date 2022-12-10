""" 
"Probabilistic Face Embeddings" to compute mean and variance of each Gaussian.
"""

import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class UncertaintyHead(nn.Module):
    def __init__(self, in_channels=512):
        super(UncertaintyHead, self).__init__()

        self.fc1 = nn.Linear(in_channels, in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels, in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels, affine=True)

    def forward(self, x):  # (b h w) c
        x = self.fc1(x)
        X = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)

        return x


class PredictionHead(nn.Module):
    def __init__(self, inplanes, planes):
        super(PredictionHead, self).__init__()

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, inplanes)

    def forward(self):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x
