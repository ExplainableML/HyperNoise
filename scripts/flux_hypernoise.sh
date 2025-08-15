#!/bin/bash
#SBATCH ...

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12


python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=7 \
    --master_port=25851 \
    main.py \
    --task="all" \
    --model=flux \
    --train_type="noise" \
    --lora_rank=128 \
    --alpha_multiplier=5 \
    --optim="adamw" \
    --lr=2e-5 \
    --batch_size=7 \
    --accumulation_steps=4 \
    --grad_clip=1.0 \
    --warmup_steps=100 \
    --reg_weight=0.5 \
    --latent_type="batch" \
    --epochs=30 \
    --save_dir="./outputs" \
    --log_every=5000 \
    --save_every=25000 \
    --memsave
