import torch
import math
from torch import nn
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, infeas, outfeas, nfeas, n_lf, bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = infeas
        self.out_features = outfeas
        self.n_lf = n_lf
        self.weight = Parameter(torch.FloatTensor(infeas, n_lf))
        self.weight1 = Parameter(torch.FloatTensor(nfeas, nfeas))
        self.weight2 = Parameter(torch.FloatTensor(n_lf, outfeas))
        if bias:
            self.bias = Parameter(torch.FloatTensor(nfeas))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)


    def forward(self, x, fadj):
        x1 = torch.matmul(x, self.weight)
        x1 = torch.matmul(x1, self.weight2)
        x1 = x1.squeeze()
        output = torch.matmul(x1,(self.weight1*fadj))

        if self.bias is not None:
            return output + self.bias
        else:
            return output


