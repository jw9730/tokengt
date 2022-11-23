"""
Utilities for handling varying-sized sets
Batching is based on padding and masking for efficient parallelization
Masking convention: Data == True, Padding == False
"""
from typing import Tuple
import torch

MASK = 0


@torch.no_grad()
def test_valid(x: torch.Tensor) -> None:
    has_inf = torch.isinf(x).any()
    has_nan = torch.isnan(x).any()
    assert not (has_inf or has_nan), f"tensor of shape [{x.shape}] has inf:{has_inf} or has nan:{has_nan}"


@torch.no_grad()
def test_valid_mask(x: torch.Tensor, mask: torch.BoolTensor) -> None:
    [test_valid(x_[m_]) for x_, m_ in zip(x, mask)]


def to_masked_batch(x: torch.Tensor, sizes: list, value: float = MASK) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """Construct masked batch from node features of torch_geometric data format (concatenated)
    :param x: Tensor([n1 + ... + nb, D])
    :param sizes: list([n1, ..., nb])
    :param value: float
    :return: Tensor([B, N, D]), BoolTensor([B, N])
    """
    mask = get_mask(torch.tensor(sizes, dtype=torch.long, device=x.device))
    new_x = to_batch(x, sizes, mask, value)
    return new_x, mask


def get_mask(sizes: torch.LongTensor, max_sizes:int) -> torch.BoolTensor:
    idx = torch.arange(max_sizes, device=sizes.device)
    return idx[None, :] < sizes[:, None]  # [B, N]
    

def to_batch(x: torch.Tensor, sizes: list, mask: torch.BoolTensor, value: float = MASK) -> torch.Tensor:
    new_x = torch.zeros(len(sizes), max(sizes), x.size(-1), device=x.device, dtype=x.dtype).fill_(value)
    new_x[mask] = x
    return new_x


def masked_fill(x: torch.Tensor, mask: torch.BoolTensor, value: float) -> torch.Tensor:
    return x.clone().masked_fill_(~mask.unsqueeze(-1), value)