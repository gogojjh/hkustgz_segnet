""" 
"Probabilistic Face Embeddings" to compute mean and variance of each Gaussian.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class UncertaintyHead(nn.Module):   # feature -> log(sigma^2)
    def __init__(self, in_feat=256, out_feat=256):
        super(UncertaintyHead, self).__init__()
        self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
        self.bn1 = nn.BatchNorm2d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm2d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

    def forward(self, x):  # [b c h w]
        x = x.permute(0, 2, 3, 1)  # [b h w c]
        x = F.linear(x, F.normalize(self.fc1, dim=-1))  # [b h w c]
        x = x.permute(0, 3, 1, 2)  # [b h w c]-> [b c h w]
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc2, dim=-1))
        x = x.permute(0, 3, 1, 2)
        x = self.bn2(x)
        x = self.gamma * x + self.beta
        x = torch.log(torch.exp(x) + 1e-6)
        x = torch.sigmoid(x)  # ! log^sigma

        return x  # [b c h w]

# class UncertaintyHead(nn.Module):   # feature -> log(sigma^2)
#     def __init__(self, in_feat=256, out_feat=256):
#         super(UncertaintyHead, self).__init__()
        
#         self.bn1 =  nn.BatchNorm2d(in_feat, affine=True)
#         self.drop = nn.Dropout(0.4)
#         self.var_layer = nn.Linear(in_feat, out_feat)
#         self.bn2 = nn.BatchNorm2d(out_feat)

#         # nn.init.xavier_uniform_(self.var_layer.weight)
#         # nn.init.constant_(self.var_layer.bias, 0)

#     def forward(self, x):  # [b c h w]
#         # x = self.bn1(x)
#         # x = self.drop(x)
#         x = x.permute(0, 2, 3, 1)  # [b h w c]
#         x = self.var_layer(x)
#         x = x.permute(0, 3, 1, 2)  # [b h w c]-> [b c h w]
#         # x = self.bn2(x)

#         return x  # [b c h w]
    
    
# class MeanHead(nn.Module):   # feature -> log(sigma^2)
#     def __init__(self, in_feat=256, out_feat=256):
#         super(MeanHead, self).__init__()
        
#         self.bn1 =  nn.BatchNorm2d(in_feat, affine=True)
#         self.drop = nn.Dropout(0.4)
#         self.mean_layer = nn.Linear(in_feat, out_feat)
#         self.bn2 = nn.BatchNorm2d(out_feat)

#         # nn.init.xavier_uniform_(self.mean_layer.weight)
#         # nn.init.constant_(self.mean_layer.bias, 0)

#     def forward(self, x):  # [b c h w]
#         # x = self.bn1(x)
#         # x = self.drop(x)
#         x = x.permute(0, 2, 3, 1)  # [b h w c]
#         x = self.mean_layer(x)
#         x = x.permute(0, 3, 1, 2)  # [b h w c]-> [b c h w]
#         # x = self.bn2(x)

#         return x  # [b c h w]
