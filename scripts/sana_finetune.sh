#!/bin/bash
#SBATCH ...

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12


python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=6 \
    --master_port=25821 \
    main.py \
    --task="all" \
    --model=sana \
    --train_type="finetune" \
    --lora_rank=128 \
    --alpha_multiplier=2 \
    --optim="sgd" \
    --lr=1e-3 \
    --batch_size=18 \
    --accumulation_steps=3 \
    --grad_clip=1.0 \
    --warmup_steps=100 \
    --reg_weight=0.5 \
    --latent_type="batch" \
    --epochs=30 \
    --save_dir="./outputs" \
    --log_every=5000 \
    --save_every=25000 \
    --memsave
