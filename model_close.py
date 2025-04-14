import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer, GNN_Layer_Init, MLP

class GNN_Close(nn.Module):
    def __init__(self, ninput, nhid, dropout):
        super(GNN_Close, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput, nhid)
        self.gc2 = GNN_Layer(nhid, nhid)
        self.gc3 = GNN_Layer(nhid, nhid)
        self.gc4 = GNN_Layer(nhid, nhid)
        self.gc5 = GNN_Layer(nhid, nhid)
        self.gc6 = GNN_Layer(nhid, nhid)
        self.gc7 = GNN_Layer(nhid, nhid)

        self.dropout = dropout
        self.score_layer = MLP(nhid, self.dropout)

    def forward(self, adj1, adj2):
        # Aggregation layers
        x1 = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        x2 = F.normalize(F.relu(self.gc2(x1, adj2)), p=2, dim=1)
        x3 = F.normalize(F.relu(self.gc3(x2, adj2)), p=2, dim=1)
        x4 = F.normalize(F.relu(self.gc4(x3, adj2)), p=2, dim=1)
        x5 = F.normalize(F.relu(self.gc5(x4, adj2)), p=2, dim=1)
        x6 = F.normalize(F.relu(self.gc6(x5, adj2)), p=2, dim=1)
        x7 = F.relu(self.gc7(x6, adj2))  # Final layer without normalization

        # Score layers (no need to pass dropout; handled inside MLP)
        score = (
            self.score_layer(x1) +
            self.score_layer(x2) +
            self.score_layer(x3) +
            self.score_layer(x4) +
            self.score_layer(x5) +
            self.score_layer(x6) +
            self.score_layer(x7)
        )

        return score
