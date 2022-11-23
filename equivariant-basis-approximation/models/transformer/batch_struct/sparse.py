from typing import Union, Callable

import torch
import numpy as np

from ...utils.set import get_mask, masked_fill


class Batch(object):
    indices: Union[None, torch.LongTensor]
    values: torch.Tensor
    n_nodes: list
    n_edges: Union[None, list]
    device: torch.device
    mask: torch.BoolTensor
    node_mask: torch.BoolTensor
    null_node: bool

    def __init__(self, indices: Union[None, torch.LongTensor], values: torch.Tensor,
                 n_nodes: list, n_edges: Union[None, list], mask: torch.BoolTensor = None, skip_masking: bool = False,
                 node_mask: torch.BoolTensor = None, null_node=False):
        """a mini-batch of sparse (hyper)graphs
        :param indices: LongTensor([B, |E|, k])
        :param values: Tensor([B, |E|, D])
        :param n_nodes: List([n1, ..., nb])
        :param n_edges: List([|E1|, ..., |Eb|])  Number of 1-edges + 2-edges
        :param mask: BoolTensor([B, |E|])
        :param skip_masking:
        :parem node_mask 
        :param: null_node
        """
        # caution: to reduce overhead, we assume a specific organization of indices: see comment in get_diag()
        # we also assume that indices are already well-masked (invalid entries are zero): see comment in self.apply_mask()
        self.indices = indices  # [B, |E|, k] or None
        self.values = values  # [B, |E|, D]
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.device = values.device
        self.order = 1 if indices is None else indices.size(-1)
        assert self.order in (1, 2)
        self.node_mask = get_mask(torch.tensor(n_nodes, dtype=torch.long, device=self.device),
                                  max(n_nodes)) if node_mask is None else node_mask  # [B, N]
        if self.order == 1:
            self.mask = self.node_mask
        else:
            self.mask = get_mask(torch.tensor(n_edges, dtype=torch.long, device=self.device),
                                 max(n_edges)) if mask is None else mask  # [B, |E|]
        if not skip_masking:
            # set invalid values to 0
            self.apply_mask(0)

        self.null_node = null_node

    def __repr__(self):
        return f"Batch(indices {list(self.indices.size())}, values {list(self.values.size())}"

    def to(self, device: Union[str, torch.device]) -> 'Batch':
        if self.indices is not None:
            self.indices = self.indices.to(device)
        self.values = self.values.to(device)
        self.mask = self.mask.to(device)
        self.node_mask = self.node_mask.to(device)
        self.mask2d = self.mask2d.to(device)
        self.device = self.values.device
        return self

    def apply_mask(self, value=0.) -> None:
        # mask out invalid tensor elements
        self.values = masked_fill(self.values, self.mask, value)


def get_mask2d(G: Batch, node_null=False):
    # mask: [bsize, max(n+e)] 
    # we want the mask : [bsize, max(n+e), max(n+e)]
    # n_edges: List([|E1|, ..., |Eb|])
    bsize, E = G.mask.shape
    if node_null == False:
        mask2d = torch.zeros((bsize, E, E), dtype=torch.bool, device=G.device)
        for i in range(bsize):
            n_edge_i = G.n_edges[i]
            mask2d[i, :n_edge_i, :n_edge_i] = torch.ones((n_edge_i, n_edge_i), dtype=torch.bool, device=G.device)
    else:
        mask2d = torch.zeros((bsize, E, E + 1), dtype=torch.bool, device=G.device)
        for i in range(bsize):
            n_edge_i = G.n_edges[i]
            mask2d[i, :n_edge_i, :n_edge_i + 1] = torch.ones((n_edge_i, n_edge_i + 1), dtype=torch.bool,
                                                             device=G.device)
    return mask2d


def batch_like(G: Batch, values: torch.Tensor, skip_masking=False) -> Batch:
    return Batch(G.indices, values, G.n_nodes, G.n_edges, G.mask, skip_masking, G.node_mask, G.null_node)


def apply(G: Union[torch.Tensor, Batch], f: Callable[[torch.Tensor], torch.Tensor], skip_masking=False) -> Union[
    torch.Tensor, Batch]:
    if isinstance(G, torch.Tensor):
        return f(G)
    return batch_like(G, f(G.values), skip_masking)


def add_batch(G1: Union[torch.Tensor, Batch], G2: Union[torch.Tensor, Batch]) -> Union[torch.Tensor, Batch]:
    # add features of two batched graphs with identical edge structures
    if isinstance(G1, Batch) and isinstance(G2, Batch):
        assert G1.order == G2.order
        assert G1.n_nodes == G2.n_nodes
        assert G1.n_edges == G2.n_edges
        return batch_like(G1, G1.values + G2.values, skip_masking=True)
    else:
        assert isinstance(G1, torch.Tensor) and isinstance(G2, torch.Tensor)
        assert G1.size() == G2.size()
        return G1 + G2


def make_batch_concatenated(node_feature: torch.Tensor, edge_index: torch.LongTensor, edge_feature: torch.Tensor,
                            n_nodes: list, n_edges: list, null_params: dict) -> Batch:
    """
    :param node_feature: Tensor([sum(n), Dv])
    :param edge_index: LongTensor([2, sum(e)])
    :param edge_feature: Tensor([sum(e), De])
    :param n_nodes: list
    :param n_edges: list
    :parem null_params: dict
    """
    assert len(node_feature.size()) == len(edge_index.size()) == len(edge_feature.size()) == 2
    use_null_node = null_params['use_null_node']
    null_feat = null_params['null_feature']  # [1, shared_dim]

    bsize = len(n_nodes)
    node_dim = node_feature.size(-1)
    edge_dim = edge_feature.size(-1)
    assert node_dim == edge_dim
    shared_dim = node_dim
    device = node_feature.device
    dtype = node_feature.dtype
    n = node_feature.size(0)  # sum(n)
    e = edge_feature.size(0)  # sum(e)
    # unpack nodes
    idx = torch.arange(max(n_nodes), device=device)
    idx = idx[None, :].expand(bsize, max(n_nodes))  # [B, N]
    node_index = torch.arange(max(n_nodes), device=device, dtype=torch.long)
    node_index = node_index[None, :, None].expand(bsize, max(n_nodes), 2)  # [B, N, 2]
    node_num_vec = torch.tensor(n_nodes, device=device)[:, None]  # [B, 1]
    unpacked_node_index = node_index[idx < node_num_vec]  # [N, 2]
    unpacked_node_feature = node_feature  # [sum(n), Dv]
    # unpack edges
    edge_num_vec = torch.tensor(n_edges, device=device)[:, None]  # [B, 1]
    unpacked_edge_index = edge_index.t()  # [|E|, 2]
    unpacked_edge_feature = edge_feature  # [sum(e), De]

    if not use_null_node:
        # compose tensor
        n_edges_ = [n + e for n, e in zip(n_nodes, n_edges)]
        max_size = max(n_edges_)
        edge_index_ = torch.zeros(bsize, max_size, 2, device=device, dtype=torch.long)  # [B, N + |E|, 2]
        edge_feature_ = torch.zeros(bsize, max_size, shared_dim, device=device, dtype=dtype)  # [B, N + |E|, shared_dim]
        full_index = torch.arange(max_size, device=device)[None, :].expand(bsize, max_size)  # [B, N + |E|]

        node_mask = full_index < node_num_vec  # [B, N + |E|]
        edge_mask = (node_num_vec <= full_index) & (full_index < node_num_vec + edge_num_vec)  # [B, N + |E|]
        edge_index_[node_mask] = unpacked_node_index
        edge_index_[edge_mask] = unpacked_edge_index
        edge_feature_[node_mask] = unpacked_node_feature
        edge_feature_[edge_mask] = unpacked_edge_feature
        # setup batch
        return Batch(edge_index_, edge_feature_, n_nodes, n_edges_, null_node=False)
    else:
        # compose tensor
        n_edges_ = [n + e + 1 for n, e in zip(n_nodes, n_edges)]
        total_edges_num_vec = torch.tensor(n_edges_, device=device)[:, None]  # [B, 1]
        new_n_nodes = [n + 1 for n in n_nodes]

        unpacked_null_index = []
        for i in range(bsize):
            unpacked_null_index.append([n_nodes[i], n_nodes[i]])
        unpacked_null_index = torch.tensor(unpacked_null_index, device=device)  # [B, 2]

        max_size = max(n_edges_)  # N + |E| + 1
        edge_index_ = torch.zeros(bsize, max_size, 2, device=device, dtype=torch.long)  # [B, N+|E|+1, 2]
        edge_feature_ = torch.zeros(bsize, max_size, shared_dim, device=device, dtype=dtype)  # [B, N+|E|+1, D]
        full_index = torch.arange(max_size, device=device)[None, :].expand(bsize, max_size)  # [B, N+|E|+1]

        # node_num_vec: [B, 1]
        node_mask = full_index < node_num_vec  # [B, N+|E|+1]
        edge_mask = (node_num_vec <= full_index) & (full_index < node_num_vec + edge_num_vec)  # [B, N+|E|+1]
        null_mask = (node_num_vec + edge_num_vec <= full_index) & (full_index < total_edges_num_vec)  # [B, N+|E|+1]

        # unpacked_node_index: [?, 2] 
        edge_index_[node_mask] = unpacked_node_index
        edge_index_[edge_mask] = unpacked_edge_index
        edge_index_[null_mask] = unpacked_null_index
        # 
        unpacked_null_feature = null_feat.repeat(bsize, 1)
        edge_feature_[node_mask] = unpacked_node_feature
        edge_feature_[edge_mask] = unpacked_edge_feature
        edge_feature_[null_mask] = unpacked_null_feature
        # we let full 0 for the feature of null nodes 
        return Batch(edge_index_, edge_feature_, new_n_nodes, n_edges_, null_node=True)


def add_null_token(G: Batch, null_feat: torch.tensor):
    # null_feat: [1, shared_dim]
    indices, values, n_nodes, n_edges12 = G.indices, G.values, G.n_nodes, G.n_edges
    # mask: [B, E];  indices: [B, |E|, 2];   values: [B, |E|, D];   
    B, E, D = G.values.shape
    device = G.device

    n_edges2 = [n_edges12[i] - n_nodes[i] for i in range(B)]

    new_n_nodes = [x + 1 for x in n_nodes]  # number of 1-edges
    new_n_edges12 = [x + 1 for x in n_edges12]  # number of 1-edges + number of 2-edges

    unpack_null_index = torch.zeros(B, 2, dtype=torch.long, device=device)  # [B, 2]
    unpack_null_index[:, 0] = torch.tensor(n_nodes)
    unpack_null_index[:, 1] = torch.tensor(n_nodes)

    full_index = torch.arange(E + 1, device=device)[None, :].expand(B, E + 1)  # [B, E+1]
    node_num_vec = torch.tensor(n_nodes, device=device)[:, None]
    edge_num_vec = torch.tensor(n_edges2, device=device)[:, None]
    null_num_vec = torch.ones(B, device=device)[:, None]
    null_mask = (node_num_vec + edge_num_vec <= full_index) & (full_index < node_num_vec + edge_num_vec + null_num_vec)

    new_indices = torch.zeros(B, E + 1, 2, device=device, dtype=torch.long)  # [B, E+1, 2]
    new_indices[:, :E, :] = indices
    new_indices[null_mask] = unpack_null_index

    new_values = torch.zeros(B, E + 1, D, device=device, dtype=values.dtype)  # [B, E+1, D]
    new_values[:, :E, :] = values
    null_feat = null_feat.repeat(B, 1)  # [B, D] 
    new_values[null_mask] = null_feat

    return Batch(new_indices, new_values, new_n_nodes, new_n_edges12, null_node=True, skip_masking=True)
