import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import utils
import torch.optim as optim
# 路径
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

    def forward(self, x, adj):
        h1 = F.relu(self.gc1(x, adj))
        x = self.gc2(h1, adj)
        return F.log_softmax(x, dim=1)
    #CrossEntropyLoss就是把以上Softmax–Log–NLLLoss





