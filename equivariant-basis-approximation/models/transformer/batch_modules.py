from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch_struct.sparse import Batch as B, apply, add_batch


class Apply(nn.Module):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], skip_masking=False):
        super().__init__()
        self.f = f
        self.skip_masking = skip_masking

    def forward(self, G: Union[torch.Tensor, B]) -> Union[torch.Tensor, B]:
        return apply(G, self.f, self.skip_masking)


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(G1: Union[torch.Tensor, B], G2: Union[torch.Tensor, B]) -> Union[torch.Tensor, B]:
        return add_batch(G1, G2)
