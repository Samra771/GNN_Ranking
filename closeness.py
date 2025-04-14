 import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_close import GNN_Close
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
    data_path = "./datasets/data_splits/SF/closeness/"
    print("Scale-free graphs selected.")
elif gtype == "ER":
    data_path = "./datasets/data_splits/ER/closeness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP/closeness/"
    print("Gaussian Random Partition graphs selected.")
else:
    raise ValueError(f"Unknown graph type: {gtype}")

# --------------------
# LOAD DATA
# --------------------
print("Loading data...")
with open(data_path + "training.pickle", "rb") as f:
    list_graph_train, list_n_seq_train, list_num_node_train, cc_mat_train = pickle.load(f)

with open(data_path + "test.pickle", "rb") as f:
    list_graph_test, list_n_seq_test, list_num_node_test, cc_mat_test = pickle.load(f)

model_size = 10000

# --------------------
# GRAPH TO ADJACENCY
# --------------------
print("Converting graphs to adjacency matrices...")
list_adj_train, list_adj_mod_train = graph_to_adj_close(
    list_graph_train, list_n_seq_train, list_num_node_train, model_size
)
list_adj_test, list_adj_mod_test = graph_to_adj_close(
    list_graph_test, list_n_seq_test, list_num_node_test, model_size
)

# --------------------
# MODEL INIT
# --------------------
hidden = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GNN_Close(ninput=model_size, nhid=hidden, dropout=0.6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epoch = 15


# --------------------
# TRAIN FUNCTION
# --------------------
def train(adj_list, adj_mod_list, node_counts, targets):
    model.train()
    total_loss = 0

    for i in range(len(adj_list)):
        adj = adj_list[i].to(device)
        adj_mod = adj_mod_list[i].to(device)
        target = torch.from_numpy(targets[:, i]).float().to(device)

        optimizer.zero_grad()
        pred = model(adj, adj_mod)
        loss = loss_cal(pred, target, node_counts[i], device, model_size)

        loss.backward()
        optimizer.step()
        total_loss += float(loss)

    return total_loss / len(adj_list)


# --------------------
# TEST FUNCTION
# --------------------
def test(adj_list, adj_mod_list, node_counts, targets):
    model.eval()
    kt_scores = []

    with torch.no_grad():
        for i in range(len(adj_list)):
            adj = adj_list[i].to(device)
            adj_mod = adj_mod_list[i].to(device)
            target = torch.from_numpy(targets[:, i]).float().to(device)

            pred = model(adj, adj_mod)
            kt = ranking_correlation(pred, target, node_counts[i], model_size)
            kt_scores.append(kt)

    avg_kt = np.mean(kt_scores)
    std_kt = np.std(kt_scores)
    print(f"    Average KT score on test graphs: {avg_kt:.4f}, Std: {std_kt:.4f}")


# --------------------
# TRAINING LOOP
# --------------------
print("Training started")
print(f"Number of epochs: {num_epoch}")
for epoch in range(num_epoch):
    print(f"\nEpoch {epoch + 1}/{num_epoch}")
    train_loss = train(list_adj_train, list_adj_mod_train, list_num_node_train, cc_mat_train)
    print(f"Train Loss: {train_loss:.4f}")

    test(list_adj_test, list_adj_mod_test, list_num_node_test, cc_mat_test)

