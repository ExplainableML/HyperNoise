#!/bin/bash
#SBATCH ...


python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=6 \
    --master_port=25832 \
    main.py \
    --task="all" \
    --model="sd-turbo" \
    --train_type="noise" \
    --lora_rank=128 \
    --alpha_multiplier=1 \
    --optim="sgd" \
    --lr=1e-3 \
    --batch_size=18 \
    --accumulation_steps=1 \
    --grad_clip=10.0 \
    --warmup_steps=0 \
    --reg_weight=0.5 \
    --latent_type="batch" \
    --epochs=100000 \
    --save_dir="./outputs" \
    --log_every=2500 \
    --save_every=25000 \
    --last_layer_name="conv_out" \
    --viz_noise \
    --latent_type="inf"