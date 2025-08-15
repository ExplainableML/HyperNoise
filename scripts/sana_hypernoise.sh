#!/bin/bash
#SBATCH ...

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12


python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=25811 \
    main.py \
    --task="all" \
    --model=sana \
    --train_type="noise" \
    --lora_rank=128 \
    --alpha_multiplier=2 \
    --optim="sgd" \
    --lr=1e-3 \
    --batch_size=3 \
    --accumulation_steps=3 \
    --grad_clip=1.0 \
    --warmup_steps=100 \
    --reg_weight=0.5 \
    --latent_type="batch" \
    --epochs=5 \
    --save_dir="./outputs" \
    --log_every=10 \
    --save_every=10 \
    --memsave