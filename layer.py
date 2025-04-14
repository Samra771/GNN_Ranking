import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GNN_Layer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias if self.bias is not None else output

class GNN_Layer_Init(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer_Init, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        support = self.weight
        output = torch.spmm(adj, support)
        return output + self.bias if self.bias is not None else output

class MLP(Module):
    def __init__(self, nhid, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(nhid, 2 * nhid)
        self.linear2 = torch.nn.Linear(2 * nhid, 2 * nhid)
        self.linear3 = torch.nn.Linear(2 * nhid, 1)

    def forward(self, input_vec):
        x = F.relu(self.linear1(input_vec))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear3(x)
        return x
