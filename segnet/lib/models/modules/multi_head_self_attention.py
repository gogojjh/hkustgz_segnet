import numpy as np
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, configer):
        super(MultiHeadSelfAttention, self).__init__()
        self.configer = configer
        
        n_head = 1
        d_in = self.configer.get('protoseg', 'proj_dim')
        d_hidden = d_in

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x))) # [(b h w) 8*8 proj_dim]
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x) 
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output