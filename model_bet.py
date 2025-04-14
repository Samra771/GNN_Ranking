import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer, GNN_Layer_Init, MLP
import torch

class GNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout):
        super(GNN_Bet, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput, nhid)
        self.gc2 = GNN_Layer(nhid, nhid)
        self.gc3 = GNN_Layer(nhid, nhid)
        self.gc4 = GNN_Layer(nhid, nhid)
        self.gc5 = GNN_Layer(nhid, nhid)

        self.dropout = dropout
        self.score_layer = MLP(nhid, self.dropout)

    def forward(self, adj1, adj2):
        x_1 = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        x2_1 = F.normalize(F.relu(self.gc1(adj2)), p=2, dim=1)

        x_2 = F.normalize(F.relu(self.gc2(x_1, adj1)), p=2, dim=1)
        x2_2 = F.normalize(F.relu(self.gc2(x2_1, adj2)), p=2, dim=1)

        x_3 = F.normalize(F.relu(self.gc3(x_2, adj1)), p=2, dim=1)
        x2_3 = F.normalize(F.relu(self.gc3(x2_2, adj2)), p=2, dim=1)

        x_4 = F.normalize(F.relu(self.gc4(x_3, adj1)), p=2, dim=1)
        x2_4 = F.normalize(F.relu(self.gc4(x2_3, adj2)), p=2, dim=1)

        x_5 = F.relu(self.gc5(x_4, adj1))
        x2_5 = F.relu(self.gc5(x2_4, adj2))

        score1 = sum([
            self.score_layer(x_1),
            self.score_layer(x_2),
            self.score_layer(x_3),
            self.score_layer(x_4),
            self.score_layer(x_5)
        ])
        score2 = sum([
            self.score_layer(x2_1),
            self.score_layer(x2_2),
            self.score_layer(x2_3),
            self.score_layer(x2_4),
            self.score_layer(x2_5)
        ])

        return torch.mul(score1, score2)
