import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.functional import normalize


@torch.no_grad()
def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.t()  # [cols, cols]


@torch.no_grad()
def gaussian_orthonormal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    """create 2D Gaussian orthonormal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    result = torch.diag(multiplier) @ final_matrix
    result = normalize(result, p=2.0, dim=1)

    return result


def get_random_orthogonal_pe(edge12_indices: torch.LongTensor, n_node: int, n_edge12: int, half_ortho_dim: int,
                             device=None):
    assert half_ortho_dim >= n_node
    # edge12_indices: [|E|, 2] the indices for all 1-edge and 2-edge of the graph from the Batch data object 
    # n_node: number of nodes (1-edge) of the graph
    # n_edge12: number of nodes (1-edge) and edges (2-edge) of graph 
    E = edge12_indices.shape[0]
    random_orthonormal_matrix = gaussian_orthonormal_random_matrix(n_node, n_node).to(device)  # [N, N]

    node_ortho_pe = torch.zeros(n_node, half_ortho_dim, dtype=torch.float, device=device)
    node_ortho_pe[:, :n_node] = random_orthonormal_matrix

    all_edges_ortho_pe = torch.zeros(E, 2 * half_ortho_dim, dtype=torch.float, device=device)  # [E, 2*half_ortho_dim]
    all_edges_ortho_pe[:n_edge12, :half_ortho_dim] = torch.index_select(input=node_ortho_pe, dim=0,
                                                                        index=edge12_indices[:n_edge12, 0])
    all_edges_ortho_pe[:n_edge12, half_ortho_dim:] = torch.index_select(input=node_ortho_pe, dim=0,
                                                                        index=edge12_indices[:n_edge12, 1])

    return all_edges_ortho_pe.unsqueeze(0)  # [1, E, 2*half_ortho_dim]
