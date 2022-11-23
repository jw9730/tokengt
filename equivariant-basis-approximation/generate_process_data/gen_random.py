import torch
from torch_geometric.data import Data
# from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.utils import remove_self_loops, to_undirected
from random import randint
import numpy as np


def barabasi_albert_graph(num_nodes, num_edges, seed=0):
    assert 0 < num_edges < num_nodes
    row = torch.arange(num_edges)
    np.random.seed(seed)
    col = np.random.permutation(num_edges)
    col = torch.tensor(col)

    for i in range(num_edges, num_nodes):
        row = torch.cat([row, torch.full((num_edges,), i, dtype=torch.long)])
        np.random.seed(seed)
        choice = np.random.choice(torch.cat([row, col]).numpy(), num_edges)
        col = torch.cat([col, torch.from_numpy(choice)])
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index


def generate_random_graph(num_nodes, num_edges_attached, graph_seed=0):
    edge_index = barabasi_albert_graph(num_nodes, num_edges_attached, graph_seed)
    return Data(num_nodes=num_nodes, edge_index=edge_index)


def generate_multiple_graphs(num_graphs, min_num_nodes, max_num_nodes, min_num_edges_attached, max_num_edges_attached,
                             seed=0):
    list_graphs = []
    np.random.seed(seed)
    list_num_nodes = np.random.randint(min_num_nodes, max_num_nodes + 1, num_graphs)
    np.random.seed(seed)
    list_seeds = np.random.randint(0, 10000, 3 * num_graphs)
    np.random.seed(seed)
    list_num_edges_attached = np.random.randint(min_num_edges_attached, max_num_edges_attached + 1, num_graphs)
    for i in range(num_graphs):
        graph_seed_i = list_seeds[3 * i]
        node_seed_i = list_seeds[3 * i + 1]
        edge_seed_i = list_seeds[3 * i + 2]

        num_nodes_i = list_num_nodes[i]
        num_edges_attached_i = list_num_edges_attached[i]
        graph = generate_random_graph(num_nodes_i, num_edges_attached_i, graph_seed=graph_seed_i)
        list_graphs.append(graph)
    return list_graphs
