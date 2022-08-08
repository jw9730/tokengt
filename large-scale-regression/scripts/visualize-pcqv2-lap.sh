#!/usr/bin/env bash

ulimit -c unlimited

python ../visualize/visualize_attn_dist.py \
--user-dir ../tokengt \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name pcqm4mv2 \
--dataset-source ogb \
--task graph_prediction \
--criterion l1_loss \
--arch tokengt_base \
--lap-node-id \
--lap-node-id-k 16 \
--lap-node-id-sign-flip \
--prenorm \
--num-classes 1 \
--batch-size 64 \
--data-buffer-size 20 \
--save-dir ckpts/pcqv2-tokengt-lap16 \
--split valid \
--seed 12 \
--return-attention
