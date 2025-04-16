from torch import nn
import torch.nn.functional as F
from layer import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfea, n_in, n_lf, n_class, dropout=True):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, 1, nfea, n_lf)
        self.dp1 = nn.Dropout(dropout)
        self.fc = nn.Linear(nfea, n_class)
        self.dropout = dropout

    def forward(self, input, fadj):
        x = self.gc1(input, fadj)
        x = F.elu(x)
        x = self.dp1(x)
        x = self.fc(x)

        return x
