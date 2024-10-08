import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class PMMs(nn.Module):
    '''Prototype Mixture Models
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, configer):
        super(PMMs, self).__init__()
        self.configer = configer

        self.stage_num = self.configer.get('pmm', 'stage_num')
        k = self.configer.get('pmm', 'pmm_k')
        self.num_pro = k
        c = self.configer.get('protoseg', 'proj_dim')
        mu = torch.Tensor(1, c, k)  # .cuda()
        mu.normal_(0, math.sqrt(2. / k))  # Init mu
        self.mu = self._l2norm(mu, dim=1)
        self.kappa = self.configer.get('pmm', 'kappa')
        # self.register_buffer('mu', mu)

    def forward(self, support_feature):
        # [b num_proto proj_dim], [b (h w) num_proto]
        prototypes, z_ = self.generate_prototype(support_feature)
        # Prob_map, P = self.discriminative_model(query_feature, mu_f, mu_b)

        return prototypes, z_  # , Prob_map

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def EM(self, x):
        '''
        EM method
        :param x: feauture  b * c * n
        :return: mu
        '''
        b = x.shape[0]
        mu = self.mu.repeat(b, 1, 1).to(x.device)  # b * c * k
        z_ = None
        with torch.no_grad():
            for i in range(self.stage_num):
                # E STEP:
                # x: [b proj_dim (h w)] mu: [b proj_dim num_proto]
                z = self.Kernel(x, mu)  # [b (h w) num_proto]
                z = F.softmax(z, dim=2)  # b * n * k
                # M STEP:
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))  # [b (h w) num_proto]
                mu = torch.bmm(x, z_)  # b * c * k [b proj_dim num_proto]

                mu = self._l2norm(mu, dim=1)

        mu = mu.permute(0, 2, 1)  # b * k * c # [b num_proto proj_dim]

        return mu, z_

    def Kernel(self, x, mu):
        x_t = x.permute(0, 2, 1)  # b * n * c
        z = self.kappa * torch.bmm(x_t, mu)  # b * n * k

        return z

    def get_prototype(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n #! self-attention
        mu, z_ = self.EM(x)  # b * k * c [b num_proto proj_dim], [b (h w) num_proto]

        return mu, z_

    def generate_prototype(self, z):
        mu_f, z_ = self.get_prototype(z)  # [b num_proto proj_dim], [b (h w) num_proto]

        mu_ = []
        for i in range(self.num_pro):
            mu_.append(mu_f[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        return mu_, z_

    def discriminative_model(self, query_feature, mu_f, mu_b):

        mu = torch.cat([mu_f, mu_b], dim=1)
        mu = mu.permute(0, 2, 1)

        b, c, h, w = query_feature.size()
        x = query_feature.view(b, c, h * w)  # b * c * n
        with torch.no_grad():

            x_t = x.permute(0, 2, 1)  # b * n * c
            z = torch.bmm(x_t, mu)  # b * n * k

            z = F.softmax(z, dim=2)  # b * n * k

        P = z.permute(0, 2, 1)

        P = P.view(b, self.num_pro * 2, h, w)  # b * k * w * h  probability map
        P_f = torch.sum(P[:, 0:self.num_pro], dim=1).unsqueeze(dim=1)  # foreground
        P_b = torch.sum(P[:, self.num_pro:], dim=1).unsqueeze(dim=1)  # background

        Prob_map = torch.cat([P_b, P_f], dim=1)

        return Prob_map, P
