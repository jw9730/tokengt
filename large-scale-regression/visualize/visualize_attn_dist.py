import torch
import torch.nn as nn
from torch.nn.functional import normalize
import numpy as np
from fairseq import utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import os
import time
import sys
from os import path

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from algos import algos_spd

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import logging


def get_shortest_distance_matrix(N, edge2_indices):
    assert edge2_indices.shape[0] == 2
    dense_adj = torch.zeros([N, N], dtype=torch.bool)
    if edge2_indices.size(0) == 0:
        return dense_adj.numpy()
    dense_adj[edge2_indices[0, :], edge2_indices[1, :]] = True
    shortest_distance_matrix, _ = algos_spd.floyd_warshall(dense_adj.numpy())  # [N, N]
    return shortest_distance_matrix


def get_2d_index(indices1, indices2, N, E):
    # indices1: [E,]   indices2: [E,]
    assert indices1.shape[0] == indices2.shape[0] == E
    x1 = N * indices1.unsqueeze(1).repeat(1, E)
    x2 = indices2.unsqueeze(0).repeat(E, 1)
    indices_2d = x1 + x2  # [E,E]
    return indices_2d


def get_sd_edges(indices1, indices2, N, E, sd_nodes):
    assert sd_nodes.shape[0] == sd_nodes.shape[1] == N
    indices_2d = get_2d_index(indices1, indices2, N, E)  # [E, E]
    sd_edges = torch.index_select(input=sd_nodes.flatten(), dim=0, index=indices_2d.flatten()).reshape((E, E))
    return sd_edges


def get_batch_sd(node_num, edge12_num, padded_index, device):
    # shortest_distances_list: list(tensor(N,N)]
    # padded_index: [B,T,2]
    # we want [B, T, T]
    T = max(edge12_num)
    B = len(edge12_num)
    padded_sd_edges = torch.zeros(B, T, T, dtype=torch.float, device=device)
    for i in range(B):
        N = node_num[i]
        E = edge12_num[i]
        edge12_indices = padded_index[i, :E, :]  # [E,2]
        edge2_indices = edge12_indices[N:, :]  # [E2,2]
        edge2_indices = edge2_indices.transpose(0, 1)  # [2, E2]

        sd_nodes = get_shortest_distance_matrix(N, edge2_indices)  # numpy
        sd_nodes = torch.tensor(sd_nodes, dtype=torch.long, device=device)  # [N, N]
        assert N == sd_nodes.shape[0] == sd_nodes.shape[1]

        sd_edges1 = get_sd_edges(edge12_indices[:, 0], edge12_indices[:, 0], N, E, sd_nodes)  # [E, E]
        sd_edges2 = get_sd_edges(edge12_indices[:, 0], edge12_indices[:, 1], N, E, sd_nodes)  # [E, E]
        sd_edges3 = get_sd_edges(edge12_indices[:, 1], edge12_indices[:, 0], N, E, sd_nodes)  # [E, E]
        sd_edges4 = get_sd_edges(edge12_indices[:, 1], edge12_indices[:, 1], N, E, sd_nodes)  # [E, E]
        sd_edges = torch.cat(
            [sd_edges1.unsqueeze(-1), sd_edges2.unsqueeze(-1), sd_edges3.unsqueeze(-1), sd_edges4.unsqueeze(-1)],
            dim=-1)  # [E, E, 4]
        sd_edges = torch.min(input=sd_edges, dim=-1).values  # [E, E]
        padded_sd_edges[i, :E, :E] = sd_edges

    return padded_sd_edges  # [B, T, T]


def visualize(args, checkpoint_path=None, logger=None, save_path=''):
    assert args.return_attention
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    # load checkpoint
    model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(model_state, strict=True, model_cfg=cfg.model)
    model = model.float()
    del model_state

    model.to(torch.cuda.current_device())
    num_encoder_layers = model.encoder.encoder_layers
    num_heads = model.encoder.num_attention_heads
    # load dataset
    split = args.split
    task.load_dataset(split)

    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # get attention map
    with torch.no_grad():
        model.eval()
        avg_all_attn_dis = torch.zeros(num_encoder_layers, num_heads, dtype=torch.float,
                                       device=torch.cuda.current_device())
        avg_all_mae = 0.
        total_num_samples = 0
        print("len(progress) = ", len(progress))
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            node_num = sample["net_input"]["batched_data"]["node_num"]
            edge_num = sample["net_input"]["batched_data"]["edge_num"]
            target = sample["net_input"]["batched_data"]["y"]

            edge12_num = [node_num[i] + edge_num[i] for i in range(len(node_num))]
            x, attn_dict = model(**sample["net_input"])

            device = x.device
            padded_index = attn_dict['padded_index']  # [B, T, 2]
            attn_map_dict = attn_dict['maps']  # attn_map_dict[0].shape = [H, B, T+2, T+2]  torch.Size([16, 96, 57, 57])

            H, B, _, _ = attn_map_dict[0].size()
            L = len(attn_map_dict.keys())
            T = max(edge12_num)
            assert num_encoder_layers == L
            assert num_heads == H

            # discard the 2 special tokens and renormalize the attention score 
            real_attn_map_dict = {}
            for j in attn_map_dict.keys():
                attn_map_j = attn_map_dict[j][:, :, 2:, 2:]  # [H, B, T, T]

                seq_len = torch.tensor(edge12_num, dtype=torch.long, device=device)[:, None]  # [B, 1]
                token_pos = torch.arange(T, device=device)[None, :].repeat(B, 1)  # [B, T]
                mask = torch.less(token_pos, seq_len)[None, :, :, None]  # [1, B, T, 1]
                attn_map_j = attn_map_j.masked_fill(~mask, float('0'))  # [H, B, T, T]

                attn_map_j_norm = normalize(attn_map_j, p=1, dim=-1)  # [H, B, T, T]
                attn_map_j_norm = attn_map_j_norm.masked_fill(~mask, float('0'))  # [H, B, T, T]

                real_attn_map_dict[j] = attn_map_j_norm

            padded_sd_edges = get_batch_sd(node_num, edge12_num, padded_index, device=device)  # [B, T, T]
            padded_sd_edges = padded_sd_edges[None, None, ...].repeat(L, H, 1, 1, 1)  # [L, H, B, T, T]

            all_attn_map = torch.cat([real_attn_map_dict[j][None, ...] for j in range(L)])  # [L, H, B, T, T]
            avg_attn_dis = torch.einsum('...T,...T->...', padded_sd_edges, all_attn_map)  # [L, H, B, T]

            edge12_num_vec = torch.tensor(edge12_num, dtype=torch.float, device=device)[None, None, :].repeat(L, H, 1)  # [L, H, B]
            avg_attn_dis = torch.sum(avg_attn_dis, dim=-1)  # [L, H, B]
            avg_attn_dis = avg_attn_dis / edge12_num_vec  # [L, H, B]

            avg_all_attn_dis += torch.sum(avg_attn_dis, dim=-1)  # [L, H]

            avg_all_mae += nn.L1Loss(reduction="sum")(x.view(-1), target.view(-1)).item()
            total_num_samples += B

        avg_all_attn_dis /= total_num_samples  # [L, H]
        avg_all_mae /= total_num_samples  # [L, H]
        torch.save(avg_all_attn_dis, save_path)
        print(f'Done, MAE: {avg_all_mae}')


def main():
    parser = options.get_training_parser()
    parser.add_argument("--split", type=str)
    args = options.parse_args_and_arch(parser, modify_parser=None)

    best_checkpoint_path = os.path.join(args.save_dir, "checkpoint_best.pt")
    logger = logging.getLogger(__name__)
    save_path = os.path.join(args.save_dir, f'checkpoint_best_visualize_{args.split}.pt')

    visualize(args, best_checkpoint_path, logger, save_path=save_path)


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    if torch.cuda.current_device() == 0:
        print("total time = ", t1 - t0)
