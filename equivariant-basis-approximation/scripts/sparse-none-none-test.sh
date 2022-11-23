[ -z "${seed}" ] && seed="12"
[ -z "${arch}" ] && arch="--n_layers 0 --dim_hidden 1024 --dim_ff 1024 --dim_qk 128 --dim_v 128 --input_dropout_rate 0.1 --dropout_rate 0.0"
[ -z "${batch_size}" ] && batch_size="128"
echo -e "\n\n"
echo "=====================================ARGS============================"
echo "arg0: $0"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "====================================================================="
default_root_dir="../exps/$seed/sparse/none-none"
ckpt_path=$default_root_dir/"checkpoints/last.ckpt"
mkdir -p $default_root_dir
n_gpu=$(nvidia-smi -L | wc -l)
python ../entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name Synthetic-Barabasi-Albert \
      --gpus $n_gpu --accelerator ddp --precision 32 --gradient_clip_val 5.0 \
      $arch \
      --default_root_dir $default_root_dir \
      --tot_updates 3000 --warmup_updates 1000 \
      --num_graphs 1280 --min_num_nodes 10 --max_num_nodes 20 --min_num_edges_attached 2 --max_num_edges_attached 3\
      --check_val_every_n_epoch 10 \
      --peak_lr 1e-4 --end_lr 1e-9\
      --test --save_display \
      --checkpoint_path $ckpt_path\
