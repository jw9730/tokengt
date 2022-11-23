import os
from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger

from train import Model
from data import GraphDataModule, get_dataset
import torch


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)

    # ------------
    # training
    # ------------
    metric = 'train_loss'
    logger_dir = args.default_root_dir + f'/tb_log'
    logger_name = 'default'
    os.makedirs(logger_dir, exist_ok=True)

    ckpt_dirpath = args.default_root_dir + f'/checkpoints'
    os.makedirs(ckpt_dirpath, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=ckpt_dirpath,
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=100,
        mode=get_dataset(dm.dataset_name)['metric_mode'],
        save_last=True,
    )
    logger = TensorBoardLogger(save_dir=logger_dir, name='')
    if args.profile:
        trainer = pl.Trainer.from_argparse_args(args, plugins=DDPPlugin(find_unused_parameters=True),
                                                profiler=AdvancedProfiler(filename='perf.txt'), logger=logger,
                                                check_val_every_n_epoch=args.check_val_every_n_epoch)
    else:
        trainer = pl.Trainer.from_argparse_args(args, plugins=DDPPlugin(find_unused_parameters=True), logger=logger,
                                                check_val_every_n_epoch=args.check_val_every_n_epoch)

    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    attn_save_dir = args.default_root_dir + f'/attn'
    os.makedirs(attn_save_dir, exist_ok=True)

    # ------------
    # model
    # ------------
    if args.checkpoint_path != '':
        model = Model.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            baseline=args.baseline,
            dense_setting=args.dense_setting,
            attn_save_dir=attn_save_dir,
            n_layers=args.n_layers,
            dim_hidden=args.dim_hidden,
            dim_qk=args.dim_qk,
            dim_v=args.dim_v,
            dim_ff=args.dim_ff,
            n_heads=args.n_heads,
            last_layer_n_heads=args.last_layer_n_heads,
            input_dropout_rate=args.input_dropout_rate,
            dropout_rate=args.dropout_rate,
            weight_decay=args.weight_decay,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            list_struct_idx_str=args.list_struct_idx_str,
            lap_node_id=args.lap_node_id,
            lap_node_id_dim=args.lap_node_id_dim,
            rand_node_id=args.rand_node_id,
            rand_node_id_dim=args.rand_node_id_dim,
            orf_node_id=args.orf_node_id,
            orf_node_id_dim=args.orf_node_id_dim,
            type_id=args.type_id,
            not_first_order=args.not_first_order,
            maximum_node=dm.maximum_node,
            save_display=args.save_display,
        )
    else:
        model = Model(
            baseline=args.baseline,
            dense_setting=args.dense_setting,
            attn_save_dir=attn_save_dir,
            n_layers=args.n_layers,
            dim_hidden=args.dim_hidden,
            dim_qk=args.dim_qk,
            dim_v=args.dim_v,
            dim_ff=args.dim_ff,
            n_heads=args.n_heads,
            last_layer_n_heads=args.last_layer_n_heads,
            input_dropout_rate=args.input_dropout_rate,
            dropout_rate=args.dropout_rate,
            weight_decay=args.weight_decay,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            list_struct_idx_str=args.list_struct_idx_str,
            lap_node_id=args.lap_node_id,
            lap_node_id_dim=args.lap_node_id_dim,
            rand_node_id=args.rand_node_id,
            rand_node_id_dim=args.rand_node_id_dim,
            orf_node_id=args.orf_node_id,
            orf_node_id_dim=args.orf_node_id_dim,
            type_id=args.type_id,
            not_first_order=args.not_first_order,
            maximum_node=dm.maximum_node,
            save_display=args.save_display,
        )
    if args.test:
        print("test")
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.validate:
        print("validate")
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:  # train
        print("train")
        trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
