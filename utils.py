import networkx as nx
from networkit import *
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import scipy.sparse as sp
import numpy as np
import torch

# -------- Edge Extraction Helpers --------

def nkit_inedges(u, v, weight, edgeid):
    all_in_dict[u].add(v)

def nkit_outedges(u, v, weight, edgeid):
    all_out_dict[u].add(v)

def get_out_edges(g_nkit, node_sequence):
    global all_out_dict
    all_out_dict = {n: set() for n in node_sequence}
    for n in node_sequence:
        g_nkit.forEdgesOf(n, nkit_outedges)
    return all_out_dict

def get_in_edges(g_nkit, node_sequence):
    global all_in_dict
    all_in_dict = {n: set() for n in node_sequence}
    for n in node_sequence:
        g_nkit.forInEdgesOf(n, nkit_inedges)
    return all_in_dict

def nx2nkit(g_nx):
    g_nkit = Graph(directed=True)
    for _ in range(g_nx.number_of_nodes()):
        g_nkit.addNode()
    for u, v in g_nx.edges():
        g_nkit.addEdge(u, v)
    return g_nkit

def clique_check(index, node_sequence, all_out_dict, all_in_dict):
    node = node_sequence[index]
    in_nodes = all_in_dict[node]
    out_nodes = all_out_dict[node]
    for in_n in in_nodes:
        tmp_out_nodes = set(out_nodes)
        tmp_out_nodes.discard(in_n)
        if not tmp_out_nodes.issubset(all_out_dict[in_n]):
            return False
    return True

# -------- Graph Processing --------

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def graph_to_adj_bet(list_graph, list_n_sequence, list_node_num, model_size):
    list_adjacency = []
    list_adjacency_t = []

    for i in range(len(list_graph)):
        print(f"Processing graphs: {i + 1}/{len(list_graph)}", end='\r')
        graph = nx.MultiDiGraph()
        graph.add_edges_from(list(list_graph[i].edges()))
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))

        node_sequence = list_n_sequence[i]
        node_num = list_node_num[i]

        adj_temp = nx.adjacency_matrix(graph, nodelist=node_sequence)
        adj_temp_t = adj_temp.transpose()

        degree_mask = np.multiply(adj_temp.sum(axis=1), adj_temp_t.sum(axis=1))
        degree_mask = np.where(degree_mask > 0, 1.0, 0.0)

        non_zero_ind = np.nonzero(degree_mask.flatten())[0]
        g_nkit = nx2nkit(graph)
        in_n_seq = [node_sequence[nz] for nz in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit, node_sequence)
        all_in_dict = get_in_edges(g_nkit, in_n_seq)

        for index in non_zero_ind:
            if clique_check(index, node_sequence, all_out_dict, all_in_dict):
                degree_mask[index, 0] = 0.0

        adj_temp = adj_temp.multiply(csr_matrix(degree_mask))
        adj_temp_t = adj_temp_t.multiply(csr_matrix(degree_mask))

        padding_top = csr_matrix((0, 0))
        padding_bottom = csr_matrix((model_size - node_num, model_size - node_num))

        adj_mat = sp.block_diag((padding_top, adj_temp, padding_bottom))
        adj_mat_t = sp.block_diag((padding_top, adj_temp_t, padding_bottom))

        list_adjacency.append(sparse_mx_to_torch_sparse_tensor(adj_mat))
        list_adjacency_t.append(sparse_mx_to_torch_sparse_tensor(adj_mat_t))

    print("")
    return list_adjacency, list_adjacency_t

def graph_to_adj_close(list_graph, list_n_sequence, list_node_num, model_size, print_time=False):
    list_adjacency = []
    list_adjacency_mod = []

    for i in range(len(list_graph)):
        print(f"Processing graphs: {i + 1}/{len(list_graph)}", end='\r')
        graph = nx.MultiDiGraph()
        graph.add_edges_from(list(list_graph[i].edges()))
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))

        node_sequence = list_n_sequence[i]
        node_num = list_node_num[i]

        adj_temp = nx.adjacency_matrix(graph, nodelist=node_sequence)
        adj_temp_t = adj_temp.transpose()

        degree_mask = np.multiply(adj_temp.sum(axis=1), adj_temp_t.sum(axis=1))
        degree_mask = np.where(degree_mask > 0, 1.0, 0.0)

        non_zero_ind = np.nonzero(degree_mask.flatten())[0]
        g_nkit = nx2nkit(graph)
        in_n_seq = [node_sequence[nz] for nz in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit, node_sequence)
        all_in_dict = get_in_edges(g_nkit, in_n_seq)

        for index in non_zero_ind:
            if clique_check(index, node_sequence, all_out_dict, all_in_dict):
                degree_mask[index, 0] = 0.0

        degree_mask = degree_mask.reshape(1, node_num)
        adj_temp_mod = adj_temp.multiply(csr_matrix(degree_mask))

        padding_top = csr_matrix((0, 0))
        padding_bottom = csr_matrix((model_size - node_num, model_size - node_num))

        adj_mat = sp.block_diag((padding_top, adj_temp, padding_bottom))
        adj_mat_mod = sp.block_diag((padding_top, adj_temp_mod, padding_bottom))

        list_adjacency.append(sparse_mx_to_torch_sparse_tensor(adj_mat))
        list_adjacency_mod.append(sparse_mx_to_torch_sparse_tensor(adj_mat_mod))

    print("")
    return list_adjacency, list_adjacency_mod

# -------- Evaluation & Loss --------

def ranking_correlation(y_out, true_val, node_num, model_size):
    y_out = y_out.view(model_size)
    true_val = true_val.view(model_size)

    predict_arr = y_out[:node_num].cpu().detach().numpy()
    true_arr = true_val[:node_num].cpu().detach().numpy()
    kt, _ = kendalltau(predict_arr, true_arr)
    return kt

def loss_cal(y_out, true_val, num_nodes, device, model_size):
    y_out = y_out.view(model_size)
    true_val = true_val.view(model_size)

    _, order_y_true = torch.sort(-true_val[:num_nodes])
    sample_num = num_nodes * 20

    ind_1 = torch.randint(0, num_nodes, (sample_num,), device=device)
    ind_2 = torch.randint(0, num_nodes, (sample_num,), device=device)

    rank_measure = torch.sign(ind_2 - ind_1).float()

    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]]
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]]

    loss_fn = torch.nn.MarginRankingLoss(margin=1.0)
    loss_rank = loss_fn(input_arr1, input_arr2, rank_measure)

    return loss_rank
