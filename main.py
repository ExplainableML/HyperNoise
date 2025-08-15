import blobfile as bf
import json
import logging
import os
from datetime import timedelta
import torch.distributed as dist
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_lightning import seed_everything
import peft
import wandb
import types
from peft.tuners.lora.layer import Linear as LoraLinear

from arguments import parse_args
from models.utils import get_model, find_all_lora_candidates, scaled_base_lora_forward, patch_lora_layer
from rewards import get_reward_losses
from training import get_optimizer, get_lr_scheduler, get_datasets
from training import (  # Import from your new trainer module
    NoiseNetworkTrainer, 
    TrainingConfig, 
    RuntimeConfig, 
    create_trainer
)
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


def initialize_process_group(args) -> Tuple[int, int, int, torch.device]:
    """Initializes the distributed process group."""
    local_rank: int = int(os.environ.get('LOCAL_RANK', 0))
    global_rank: int = int(os.environ.get('RANK', 0))
    world_size: int = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device: torch.device = torch.device('cuda', local_rank)
    return local_rank, global_rank, world_size, device


def configure_logging(args, global_rank: int) -> str:
    """Configures logging for the training process."""
    settings: str = (
        f"reward-{args.train_type}_{args.model}"
        f"_{args.seed}_lr{args.lr}_gc{args.grad_clip}_reg{args.reg_weight if args.enable_reg else '0'}_{args.reg_type}"
        f"_batchsize{args.batch_size}_accum{args.accumulation_steps}_latent{args.latent_type}"
        f"{'_lora' + str(args.lora_rank)}{'_gradnorm' if args.grad_normalization else ''}"
        f"{'_checkpoint' if args.use_checkpoint else ''}"
    )
    bf.makedirs(f"{args.save_dir}/{args.task}/{settings}")
    bf.makedirs(f"{args.save_dir}/logs/{args.task}")  # Make sure logs dir exists
    
    if global_rank == 0:
        file_stream = open(f"{args.save_dir}/logs/{args.task}/{settings}.txt", "w")
        handler = logging.StreamHandler(file_stream)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel("INFO")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logging.info(args)
    return settings


def initialize_wandb(args, global_rank: int, settings: str) -> None:
    """Initializes Weights & Biases for tracking the training process."""
    if global_rank == 0:
        wandb.init(
            project="hypernoise",
            name=f"{args.task}_{settings}",
        )


def setup_model_with_lora(args, device: torch.device, global_rank: int, local_rank: int) -> Tuple[nn.Module, List[int]]:
    """Setup model with LoRA using your existing utils."""
    dtype = torch.bfloat16
    pipe = get_model(
        model_name=args.model,
        dtype=dtype,
        device=device,
        cache_dir=args.cache_dir,
        memsave=args.memsave,
        enable_sequential_cpu_offload=False
    )

    # Calculate latent shape based on model type
    if args.model.startswith("flux"):
        # Flux has a different latent shape format
        height, width = 512, 512
        latent_shape = [
            16 * 64,
            64,
        ]
    else:
        # SANA and other models use standard format
        height, width = 1024, 1024
        
        vae_scale_factor = pipe.vae_scale_factor
        in_channels = pipe.transformer.config.in_channels
        
        latent_shape = [
            in_channels,
            height // vae_scale_factor,
            width // vae_scale_factor
        ]

    if global_rank == 0:
        logging.info(f"Model: {args.model}, Latent shape: {latent_shape}")

    # Setup LoRA using your existing function
    all_layers = find_all_lora_candidates(pipe.transformer)
    
    if global_rank == 0:
        logging.info(f"Found {len(all_layers)} LoRA target layers")
        logging.info(f"LoRA targets: {all_layers[:5]}...")  # Log first 5
    
    main_lora_config = peft.LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * getattr(args, 'alpha_multiplier', 2),
        init_lora_weights="gaussian",
        target_modules=all_layers,
    )
    pipe.transformer.add_adapter(main_lora_config, adapter_name="hypernoise_adapter")
    # patch final layer to be initialized with 0
    if args.train_type == "noise":
        patch_lora_layer(pipe, args.last_layer_name, global_rank)
    
    # Wrap with DDP
    pipe.transformer = DDP(pipe.transformer, device_ids=[local_rank], output_device=local_rank)
    
    # Log parameter count
    params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    if global_rank == 0:
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

    pipe.transformer.module.train()
    pipe.transformer.module.enable_adapters()
    
    return pipe, latent_shape


def create_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments."""
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        n_inference_steps=getattr(args, 'n_inference_steps', 1),
        regularize=args.enable_reg,
        regularization_weight=args.reg_weight,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        log_every=args.log_every,
        save_every=getattr(args, 'save_every', 25000),
        latent_type=args.latent_type,
        reg_type=args.reg_type,
        enable_modulate_noise=args.enable_modulate_noise,
        grad_normalization=args.grad_normalization,
        one_prompt_per_batch=args.one_prompt_per_batch,
        use_checkpoint=args.use_checkpoint,
    )


def create_runtime_config(args, device: torch.device, global_rank: int, world_size: int) -> RuntimeConfig:
    """Create runtime configuration from arguments."""
    return RuntimeConfig(
        device=device,
        dtype=torch.bfloat16,
        seed=args.seed,
        rank=global_rank,
        world_size=world_size,
        log_metrics=(global_rank == 0),
    )


def get_prompts_from_jsonl(file_path: str) -> List[str]:
    """Extracts prompts from a JSONL file."""
    prompts: List[str] = []
    try:
        with open(file_path) as fp:
            for line in fp:
                metadata: Dict[str, str] = json.loads(line)
                if 'prompt' in metadata:
                    prompts.append(metadata['prompt'])
    except FileNotFoundError:
        logging.warning(f"Could not find eval prompts file: {file_path}")
        prompts = ["A cat sitting on a table", "A dog running in a park"]  # Default prompts
    return prompts


def main(args) -> None:
    """Main function to orchestrate the training process."""
    local_rank, global_rank, world_size, device = initialize_process_group(args)
    settings = configure_logging(args, global_rank)

    seed_everything(args.seed + global_rank)
    initialize_wandb(args, global_rank, settings)

    # Setup model using your existing utils
    pipe, latent_shape = setup_model_with_lora(args, device, global_rank, local_rank)
    
    # Get reward losses
    dtype = torch.bfloat16
    reward_losses = get_reward_losses(args, dtype, device, getattr(args, 'cache_dir', None))
    
    # Setup optimizer and scheduler
    params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optimizer = get_optimizer(args.optim, params, args.lr, False)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, world_size)

    # Create configurations
    training_config = create_training_config(args)
    runtime_config = create_runtime_config(args, device, global_rank, world_size)
    
    trainer = create_trainer(
        model_type=args.model,
        reward_losses=reward_losses,
        pipe=pipe,
        train_type=args.train_type,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        runtime_config=runtime_config,
        latent_shape=latent_shape,
    )

    # Get data and evaluation prompts
    prompt_loader = get_datasets(args, global_rank, world_size)
    eval_prompts = get_prompts_from_jsonl("../geneval/prompts/evaluation_metadata.jsonl")

    # Start training
    trainer.train(
        prompt_loader=prompt_loader,
        eval_prompts=eval_prompts,
        save_dir=f"{args.save_dir}/{args.task}/{settings}",
        save_images=True,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)