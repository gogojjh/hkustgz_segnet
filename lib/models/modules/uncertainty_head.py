""" 
"Probabilistic Face Embeddings" to compute mean and variance of each Gaussian.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.modules.contrast import momentum_update, l2_normalize, ProjectionHead


class UncertaintyHead(nn.Module):
    def __init__(self, in_channels=512):
        super(UncertaintyHead, self).__init__()

        self.fc1 = torch.nn.Parameter(
            torch.Tensor(in_channels, in_channels)).cuda()
        self.bn1 = nn.BatchNorm1d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Parameter(
            torch.Tensor(in_channels, in_channels)).cuda()
        self.bn2 = nn.BatchNorm1d(in_channels, affine=True)
        self.gamma = torch.nn.Parameter(torch.Tensor([1.0])).cuda()
        self.beta = torch.nn.Parameter(torch.Tensor([0.0])).cuda()

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

    def forward(self, x):  # (b h w) c
        x = self.relu(self.bn1(F.linear(x, F.normalize(self.fc1))))
        x = self.bn2(F.linear(x, F.normalize(self.fc2)))
        # scale and shift
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        x = torch.sigmoid(x)  # scale log(var) into [0, 1]

        return x


# class UncertaintyHead(nn.Module):
#     def __init__(self, in_channels=512):
#         super(UncertaintyHead, self).__init__()

#         self.fc1 = nn.Linear(in_channels, in_channels)
#         self.bn1 = nn.BatchNorm1d(in_channels, affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(in_channels, in_channels)
#         self.bn2 = nn.BatchNorm1d(in_channels, affine=True)

#     def forward(self, x):  # (b h w) c
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.bn2(x)

#         return x
