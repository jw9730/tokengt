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


def validate(args, checkpoint_path=None, logger=None, save_path=''):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    model.encoder.performer_finetune_setup()

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
            target = sample["net_input"]["batched_data"]["y"]

            x = model(**sample["net_input"])
            B = x.size(0)
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

    validate(args, best_checkpoint_path, logger, save_path=save_path)


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    if torch.cuda.current_device() == 0:
        print("total time = ", t1 - t0)
