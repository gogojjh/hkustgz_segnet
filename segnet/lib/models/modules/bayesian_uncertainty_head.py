import torch.nn as nn


class BayesianUncertaintyHead(nn.Module):
    ''' 
    "Uncertainty-Guided Transformer Reasoning for Camouflaged Object Detection"
    '''

    def __init__(self, configer):
        super(BayesianUncertaintyHead, self).__init__()
        self.configer = configer

        self.proj_dim = self.configer.get('protoseg', 'proj_dim')

        self.mean_layer = nn.Linear(self.proj_dim, self.proj_dim)
        self.var_layer = nn.Linear(self.proj_dim, self.proj_dim)

    def init_weights(self):
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.constant_(self.var_layer.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [b h w c]
        mean = self.mean_layer(x)
        logvar = self.var_layer(x)  # [b k h w]
        mean = mean.permute(0, 3, 1, 2)  # [b h w c]-> [b c h w]
        logvar = logvar.permute(0, 3, 1, 2)  # [b h w c]-> [b c h w]

        return mean, logvar
