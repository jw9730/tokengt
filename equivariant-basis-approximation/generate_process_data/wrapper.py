import torch
import torch_geometric.datasets
from torch_geometric.data import Data
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
from torch_geometric.data import Dataset
from .gen_random import generate_multiple_graphs
from random import randint


def preprocess_item(item):
    # Data(num_nodes=num_nodes, edge_index=edge_index)
    N = item.num_nodes
    sparse_edge_index = item.edge_index

    # create the dense edge index 
    arr1 = torch.arange(N).unsqueeze(-1).repeat(1, N)  # [N, N]
    arr2 = torch.arange(N).unsqueeze(0).repeat(N, 1)  # [N, N]
    arr3 = torch.cat((arr1.unsqueeze(-1), arr2.unsqueeze(-1)), dim=-1)  # [N, N, 2]
    arr4 = ~torch.eye(N, dtype=torch.bool)
    dense_edge_index = arr3[arr4].transpose(0, 1)  # [2, N*(N-1)]

    item.dense_edge_index = dense_edge_index
    item.sparse_edge_index = sparse_edge_index

    return item


class SyntheticDataset(Dataset):
    def __init__(self, num_graphs=10, min_num_nodes=5, max_num_nodes=20, min_num_edges_attached=4,
                 max_num_edges_attached=4, seed=0):
        super().__init__()
        self.list_graphs = generate_multiple_graphs(num_graphs, min_num_nodes, max_num_nodes, min_num_edges_attached,
                                                    max_num_edges_attached, seed=seed)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.list_graphs)

    def get(self, idx):
        item = self.list_graphs[idx]
        item.idx = idx
        return preprocess_item(item)
