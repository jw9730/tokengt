import torch


class Batch:
    def __init__(self, idx, sparse_edge_index, dense_edge_index, node_num, sparse_edge_num, dense_edge_num):
        super(Batch, self).__init__()
        self.idx = idx
        self.sparse_edge_index = sparse_edge_index
        self.dense_edge_index = dense_edge_index
        self.node_num = node_num
        self.sparse_edge_num = sparse_edge_num
        self.dense_edge_num = dense_edge_num

    def to(self, device):
        self.idx = self.idx.to(device)
        self.sparse_edge_index = self.sparse_edge_index.to(device)
        self.dense_edge_index = self.dense_edge_index.to(device)
        return self

    def __len__(self):
        return len(self.node_num)


def collator(items, max_node=512):
    #  Data(num_nodes=num_nodes, edge_index=edge_index)
    items = [item for item in items if item is not None and item.num_nodes <= max_node]
    node_num = [item.num_nodes for item in items]
    sparse_edge_num = [item.sparse_edge_index.shape[1] for item in items]
    dense_edge_num = [item.dense_edge_index.shape[1] for item in items]
    idxs = tuple([item.idx for item in items])
    # sparse
    sparse_edge_indices = tuple([item.sparse_edge_index for item in items])
    sparse_edge_index = torch.cat(sparse_edge_indices, dim=1)  # [2, E]
    # dense
    dense_edge_indices = tuple([item.dense_edge_index for item in items])
    dense_edge_index = torch.cat(dense_edge_indices, dim=1)  # [2, dense_E]
    return Batch(idx=torch.LongTensor(idxs),
                 sparse_edge_index=sparse_edge_index, dense_edge_index=dense_edge_index,
                 node_num=node_num, sparse_edge_num=sparse_edge_num, dense_edge_num=dense_edge_num)
