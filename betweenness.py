 import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet
import argparse

torch.manual_seed(20)

# --------------------
# ARGUMENT PARSING
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--g", default="SF")
args = parser.parse_args()
gtype = args.g

# --------------------
# DATA PATH SETUP
# --------------------
if gtype == "SF":
    data_path = "./datasets/data_splits/SF/betweenness/"
    print("Scale-free graphs selected.")
elif gtype == "ER":
    data_path = "./datasets/data_splits/ER/betweenness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP/betweenness/"
    print("Gaussian Random Partition graphs selected.")
else:
    raise ValueError(f"Unknown graph type: {gtype}")

# --------------------
# LOAD DATA
# --------------------
print("Loading data...")
with open(data_path + "training.pickle", "rb") as f:
    list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train = pickle.load(f)

with open(data_path + "test.pickle", "rb") as f:
    list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test = pickle.load(f)

model_size = 10000

# --------------------
# GRAPH TO ADJACENCY
# --------------------
print("Converting graphs to adjacency matrices...")
list_adj_train, list_adj_t_train = graph_to_adj_bet(
    list_graph_train, list_n_seq_train, list_num_node_train, model_size
)
list_adj_test, list_adj_t_test = graph_to_adj_bet(
    list_graph_test, list_n_seq_test, list_num_node_test, model_size
)

# --------------------
# MODEL INIT
# --------------------
hidden = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GNN_Bet(ninput=model_size, nhid=hidden, dropout=0.6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epoch = 15


# --------------------
# TRAIN FUNCTION
# --------------------
def train(adj_list, adj_t_list, node_counts, targets):
    model.train()
    loss_train = 0
    for i in range(len(adj_list)):
        adj = adj_list[i].to(device)
        adj_t = adj_t_list[i].to(device)

        optimizer.zero_grad()
        pred = model(adj, adj_t)

        target = torch.from_numpy(targets[:, i]).float().to(device)
        loss = loss_cal(pred, target, node_counts[i], device, model_size)

        loss.backward()
        optimizer.step()
        loss_train += float(loss)
    return loss_train / len(adj_list)


# --------------------
# TEST FUNCTION
# --------------------
def test(adj_list, adj_t_list, node_counts, targets):
    model.eval()
    kt_scores = []

    with torch.no_grad():
        for i in range(len(adj_list)):
            adj = adj_list[i].to(device)
            adj_t = adj_t_list[i].to(device)
            target = torch.from_numpy(targets[:, i]).float().to(device)

            pred = model(adj, adj_t)
            kt = ranking_correlation(pred, target, node_counts[i], model_size)
            kt_scores.append(kt)

    avg_kt = np.mean(kt_scores)
    std_kt = np.std(kt_scores)
    print(f"   Average KT score on test graphs: {avg_kt:.4f}, Std: {std_kt:.4f}")


# --------------------
# TRAINING LOOP
# --------------------
print("Training started")
print(f"Total Number of Epochs: {num_epoch}")
for epoch in range(num_epoch):
    print(f"\nEpoch {epoch + 1}/{num_epoch}")
    train_loss = train(list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train)
    print(f"Train Loss: {train_loss:.4f}")

    test(list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test)
