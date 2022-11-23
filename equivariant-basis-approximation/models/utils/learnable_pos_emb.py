import torch
import numpy as np
from scipy import sparse as sp


def get_pe(indices: torch.LongTensor, n_node, pad_size, pos_enc_dim=512):
    i = indices.detach().cpu().numpy()
    A = sp.coo_matrix((np.ones(i.shape[1]), (i[0], i[1])), shape=(n_node, n_node))
    N = sp.diags(np.asarray(A.sum(1)).squeeze(1).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n_node) - N * A * N
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    EigVec = torch.from_numpy(EigVec[:, :pos_enc_dim + 1]).float().to(indices.device)  # [N, D]
    pe = torch.zeros(pad_size, pos_enc_dim).to(indices.device)
    pe[:n_node, :min(pos_enc_dim + 1, n_node)] = EigVec
    return pe.unsqueeze(0)


def get_pe_2d(edge2_indices: torch.LongTensor, edge12_indices: torch.LongTensor, n_node: int, n_edge12: int,
              half_pos_enc_dim=512):
    # edge2_indices: [2,?]: the indices for the 2-edge of the graph
    # edge12_indices: [|E|, 2] the indices for all 1-edge and 2-edge of the graph from the Batch data object 
    # n_node: number of nodes (1-edge) of the graph
    i = edge2_indices.detach().cpu().numpy()

    A = sp.coo_matrix((np.ones(i.shape[1]), (i[0], i[1])), shape=(n_node, n_node))
    N = sp.diags(np.asarray(A.sum(1)).squeeze(1).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n_node) - N * A * N  # L is Laplacian matrix; L: [n_node, n_node]
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())  # EigVal: (n_node,)   EigVec: (n_node, n_node)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])  # EigVal: (n_node,)  EigVec: (n_node, n_node)

    EigVec = torch.from_numpy(EigVec[:, :half_pos_enc_dim]).float().to(
        edge2_indices.device)  # [n_node, min(n_node, half_pos_enc_dim)]
    node_pe = torch.zeros(n_node, half_pos_enc_dim).to(edge2_indices.device)  # [n_node, half_pos_enc_dim]
    node_pe[:, :min(n_node, half_pos_enc_dim)] = EigVec  # [n_node, half_pos_enc_dim]

    E = edge12_indices.shape[0]
    all_edges_pe = torch.zeros([E + 1, 2 * half_pos_enc_dim]).to(edge2_indices.device)  # [E+1, 2*half_pos_enc_dim]
    for j in range(n_edge12):
        node1_idx, node2_idx = edge12_indices[j][0], edge12_indices[j][1]
        all_edges_pe[j] = torch.cat((node_pe[node1_idx], node_pe[node2_idx]), dim=-1)
    return all_edges_pe.unsqueeze(0)  # [1, E+1, 2*half_pos_enc_dim]
