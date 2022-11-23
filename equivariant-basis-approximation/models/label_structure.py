from .transformer.batch_struct.sparse import Batch as B
import torch
import torch.nn as nn

struct_string_map = {1: "(i, i, i, i)", 2: "(i, i, j, i)", 3: "(i, i, i, j)", 4: "(i, j, j, j)", 5: "(i, j, i, i)",
                     6: "(i, i, j, j)", 7: "(i, j, i, j)", 8: "(i, j, j, i)", 9: "(i, j, k, k)", 10: "(i, i, j, k)",
                     11: "(i, j, k, j)", 12: "(i, j, i, k)", 13: "(i, j, j, k)", 14: "(i, j, k, i)", 15: "(i, j, k, l)"}


def get_structure_label(G: B, mask2d_G_null: torch.Tensor, mask2d_G: torch.Tensor, struct_idx=1):
    # mask2d_G : tensor((bsize, |E|, |E|, dtype=bool) = [128, 110, 110]
    assert 1 <= struct_idx <= 15
    # G.values: [B, |E|, D] 
    # G.indices: [B, |E|, 2] 
    # G.mask: [Bsize, |E|] 
    Bsize, E, _ = G.indices.shape
    arr1 = G.indices[:, None, :, :]  # [Bsize, 1, E, 2] 
    arr2 = G.indices[:, :, None, :]  # [Bsize, E, 1, 2] 
    arr3 = arr1.repeat(1, E, 1, 1)
    arr4 = arr2.repeat(1, 1, E, 1)
    arr5 = torch.cat((arr3, arr4), dim=-1)  # [Bsize, E, E, 4]  # arr5[i][j][k] = [ek, ej]

    arr6_01 = torch.eq(arr5[..., 0], arr5[..., 1])  # [Bsize, E, E]
    arr6_02 = torch.eq(arr5[..., 0], arr5[..., 2])
    arr6_03 = torch.eq(arr5[..., 0], arr5[..., 3])
    arr6_12 = torch.eq(arr5[..., 1], arr5[..., 2])
    arr6_13 = torch.eq(arr5[..., 1], arr5[..., 3])
    arr6_23 = torch.eq(arr5[..., 2], arr5[..., 3])

    if struct_idx == 1:  # (i,i,i,i)
        mask_u = torch.logical_and(arr6_01, arr6_02)
        mask_u = torch.logical_and(mask_u, arr6_03)
    elif struct_idx == 2:  # (i, i, j, i)
        mask_u = torch.logical_and(arr6_01, arr6_03)
        mask_u = torch.logical_and(mask_u, ~arr6_02)

    elif struct_idx == 3:  # (i, i, i, j)
        mask_u = torch.logical_and(arr6_01, arr6_02)
        mask_u = torch.logical_and(mask_u, ~arr6_03)

    elif struct_idx == 4:  # (i, j, j, j)
        mask_u = torch.logical_and(arr6_12, arr6_13)
        mask_u = torch.logical_and(mask_u, ~arr6_01)

    elif struct_idx == 5:  # (i, j, i, i)
        mask_u = torch.logical_and(arr6_02, arr6_03)
        mask_u = torch.logical_and(mask_u, ~arr6_01)

    elif struct_idx == 6:  # (i, i, j, j)
        mask_u = torch.logical_and(arr6_01, arr6_23)
        mask_u = torch.logical_and(mask_u, ~arr6_02)

    elif struct_idx == 7:  # (i, j, i, j)
        mask_u = torch.logical_and(arr6_02, arr6_13)
        mask_u = torch.logical_and(mask_u, ~arr6_01)

    elif struct_idx == 8:  # (i, j, j, i)
        mask_u = torch.logical_and(arr6_03, arr6_12)
        mask_u = torch.logical_and(mask_u, ~arr6_01)

    elif struct_idx == 9:  # (i, j, k, k)
        mask_tmp1 = torch.logical_and(~arr6_01, ~arr6_02)
        mask_tmp2 = torch.logical_and(~arr6_12, arr6_23)
        mask_u = torch.logical_and(mask_tmp1, mask_tmp2)

    elif struct_idx == 10:  # (i, i, j, k)
        mask_tmp1 = torch.logical_and(arr6_01, ~arr6_12)
        mask_tmp2 = torch.logical_and(~arr6_13, ~arr6_23)
        mask_u = torch.logical_and(mask_tmp1, mask_tmp2)

    elif struct_idx == 11:  # (i, j, k, j)
        mask_tmp1 = torch.logical_and(~arr6_01, ~arr6_02)
        mask_tmp2 = torch.logical_and(~arr6_12, arr6_13)
        mask_u = torch.logical_and(mask_tmp1, mask_tmp2)

    elif struct_idx == 12:  # (i, j, i, k)
        mask_tmp1 = torch.logical_and(~arr6_01, ~arr6_03)
        mask_tmp2 = torch.logical_and(~arr6_13, arr6_02)
        mask_u = torch.logical_and(mask_tmp1, mask_tmp2)

    elif struct_idx == 13:  # (i, j, j, k)
        mask_tmp1 = torch.logical_and(~arr6_01, ~arr6_03)
        mask_tmp2 = torch.logical_and(~arr6_13, arr6_12)
        mask_u = torch.logical_and(mask_tmp1, mask_tmp2)

    elif struct_idx == 14:  # (i, j, k, i)
        mask_tmp1 = torch.logical_and(~arr6_01, ~arr6_02)
        mask_tmp2 = torch.logical_and(~arr6_12, arr6_03)
        mask_u = torch.logical_and(mask_tmp1, mask_tmp2)

    elif struct_idx == 15:  # (i, j, k, l)
        mask_tmp1 = torch.logical_and(~arr6_01, ~arr6_02)
        mask_tmp2 = torch.logical_and(~arr6_03, ~arr6_12)
        mask_tmp3 = torch.logical_and(~arr6_13, ~arr6_23)
        mask_u = torch.logical_and(mask_tmp1, mask_tmp2)
        mask_u = torch.logical_and(mask_u, mask_tmp3)

    label_structure = torch.ones(size=(Bsize, E, E), dtype=torch.float, device=G.indices.device)  # [Bsize, E, E]
    label_structure = label_structure.masked_fill(mask_u == False, 0)
    label_structure = label_structure.masked_fill(mask2d_G == False, 0)

    null_label_structure = torch.zeros(size=(Bsize, E, E + 1), dtype=torch.float,
                                       device=G.indices.device)  # [Bsize, E, E+1]
    null_label_structure[:, :, :E] = label_structure
    tmp_arr = torch.sum(null_label_structure, -1)  # [Bsize, E]
    null_arr = torch.zeros(size=(Bsize, E), dtype=torch.float, device=G.indices.device)
    null_arr = null_arr.masked_fill(tmp_arr == 0, 1).flatten()  # [Bsize, E]

    list_num_edges = G.n_edges
    num_edge_vec = torch.tensor(list_num_edges)[:, None, None]  # [Bsize, 1, 1]
    full_index = torch.arange(E + 1)[None, None, :].repeat(Bsize, E, 1)  # (Bsize, E, E+1)
    mask_last_edge = full_index == num_edge_vec  # (Bsize, E, E+1)
    null_label_structure[mask_last_edge] = null_arr
    null_label_structure = null_label_structure / null_label_structure.sum(-1, keepdim=True)  # [Bsize, E, E+1]
    null_label_structure = null_label_structure.masked_fill(mask2d_G_null == False, 0)  # [Bsize, E, E+1]
    return null_label_structure  # [Bsize, E, E+1]


def get_multiple_struct_label(G: B, mask2d_G_null: torch.Tensor, mask2d_G: torch.Tensor, list_struct_idx: list):
    list_struct_label = []
    for i in range(len(list_struct_idx)):
        struct_label_i = get_structure_label(G, mask2d_G_null, mask2d_G, list_struct_idx[i])  # [Bsize, E, E+1]
        struct_label_i = struct_label_i.unsqueeze(1)  # [bsize, 1, E, E+1]
        list_struct_label.append(struct_label_i)  # list([bsize, 1, E, E+1])
    return torch.cat(list_struct_label, dim=1)  # [bsize, len(list_struct_idx), E, E+1]
