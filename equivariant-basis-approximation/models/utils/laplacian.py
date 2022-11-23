"""https://github.com/graphdeeplearning/graphtransformer/blob/c9cd49368eed4507f9ae92a137d90a7a9d7efc3a/data/SBMs.py#L145"""
import torch
import numpy as np
from scipy import sparse as sp


def lap_eig(dense_adj, number_of_nodes, in_degree):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
    """
    dense_adj = dense_adj.detach().cpu().float().numpy()
    in_degree = in_degree.detach().cpu().float().numpy()

    # Laplacian
    A = dense_adj
    N = np.diag(in_degree.clip(1) ** -0.5)
    L = np.eye(number_of_nodes) - N @ A @ N

    # (sorted) eigenvectors with numpy
    EigVal, EigVec = np.linalg.eigh(L)

    # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    eigvec = torch.from_numpy(EigVec).float()  # [N, N (channels)]
    eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float()  # [N (channels),]
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def get_pe_2d(edge2_indices: torch.LongTensor, edge12_indices: torch.LongTensor, n_node: int, n_edge12: int,
              half_pos_enc_dim=128):
    assert n_node <= half_pos_enc_dim
    device = edge12_indices.device
    # edge2_indices: [2,?]: the indices for the 2-edge of the graph
    # edge12_indices: [|E|, 2] the indices for all 1-edge and 2-edge of the graph from the Batch data object 

    dense_adj = torch.zeros([n_node, n_node], dtype=torch.bool, device=device)
    dense_adj[edge2_indices[0, :], edge2_indices[1, :]] = True
    in_degree = dense_adj.long().sum(dim=1).view(-1)

    EigVec, EigVal = lap_eig(dense_adj, n_node, in_degree)  # EigVec: [N, N] 
    node_pe = torch.zeros(n_node, half_pos_enc_dim).to(device)  # [N, half_pos_enc_dim]
    node_pe[:, :n_node] = EigVec
    E = edge12_indices.shape[0]
    all_edges_pe = torch.zeros([E, 2 * half_pos_enc_dim]).to(device)
    all_edges_pe[:n_edge12, :half_pos_enc_dim] = torch.index_select(node_pe, 0, edge12_indices[:n_edge12, 0])
    all_edges_pe[:n_edge12, half_pos_enc_dim:] = torch.index_select(node_pe, 0, edge12_indices[:n_edge12, 1])

    return all_edges_pe.unsqueeze(0)  # [1, E, 2*half_pos_enc_dim]
