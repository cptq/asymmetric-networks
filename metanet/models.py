''' 
    models that take in B x D vectors, where D is the max param size,
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSSeqReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.pre_pool = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.post_pool = nn.Sequential(*layers)

    def forward(self, x):
        """ 
            x is B x D x N
            N is number of params
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = x.transpose(2,1) # B x N x D
        x = self.pre_pool(x)
        pooled = x.mean(1)
        return self.post_pool(pooled)

class DMCCNN(nn.Module):
    ''' Similar model to Eilertsen et al. 2020. https://arxiv.org/pdf/2002.05688.pdf
        1D CNN
        hidden dim should be at least 32
        if hidden dim == 64, then num filters are:
            in_dim -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 64 ...
    '''
    def __init__(self, in_dim, hidden_dim, out_dim, conv_num_layers):
        super().__init__()
        assert conv_num_layers >= 6
        self.convs = nn.ModuleList()
        conv = nn.Sequential(nn.Conv1d(in_dim, hidden_dim//(2**5), 5), nn.ReLU(), nn.BatchNorm1d(hidden_dim//(2**5)), nn.MaxPool1d(2))
        self.convs.append(conv)
        for k in (5, 4, 3, 2, 1):
            conv = nn.Sequential(nn.Conv1d(hidden_dim//(2**k), hidden_dim//(2**(k-1)), 5), nn.ReLU(), nn.BatchNorm1d(hidden_dim//2**(k-1)), nn.MaxPool1d(2))
            self.convs.append(conv)

        for i in range(conv_num_layers - 6):
            if i % 2 == 1:
                conv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 5), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.MaxPool1d(2))
            else:
                conv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 5), nn.ReLU(), nn.BatchNorm1d(hidden_dim))
            self.convs.append(conv)
        self.readout = DSSeqReadout(hidden_dim, hidden_dim, out_dim, num_layers=2)

    def forward(self, x, *args):
        ''' x is B x D x N or B x N
            where D is number of input channels
            N is number of channels
        '''
        if x.ndim == 2:
            x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
        x = self.readout(x)
        return x
