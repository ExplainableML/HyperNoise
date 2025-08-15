from typing import Dict, List, Optional, Tuple, Callable, Any
import os
import gc
import logging
import math
from dataclasses import dataclass
from collections import defaultdict

import PIL
import PIL.Image
import torch
import torch.amp
import torch.distributed as dist
import torch.utils
from torch.amp.grad_scaler import GradScaler
import torch.utils.checkpoint
import torchvision
from torchvision.utils import make_grid
import wandb
import time
import peft
from peft import get_peft_model_state_dict
from peft.tuners.lora.layer import Linear as LoraLinear

from rewards import clip_img_transform
from rewards.base_reward import BaseRewardLoss


@dataclass
class TrainingConfig:
    """Configuration for noise network training."""
    epochs: int
    batch_size: int
    eval_batch_size: int
    n_inference_steps: int = 1
    regularize: bool = True
    regularization_weight: float = 0.01
    grad_clip: float = 0.1
    accumulation_steps: int = 1
    log_every: int = 50
    save_every: int = 25000
    latent_type: str = "inf"  # "one", "multi", "inf", "batch"
    reg_type: str = "l2"  # "l2", "kl"
    enable_modulate_noise: bool = True
    grad_normalization: bool = False
    one_prompt_per_batch: bool = False
    norm_dims: List[int] = None  # Dimensions for norm computation, will be set by model
    use_checkpoint: bool = True  # Whether to use gradient checkpointing


@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.bfloat16
    seed: int = 42
    rank: int = 0
    world_size: int = 1
    log_metrics: bool = True


@dataclass
class ModelFunctions:
    """Model-specific functions."""
    prepare_latents_fn: Callable  # (trainer, init_latents, batch_size) -> (latents, extra_data)
    encode_prompt_fn: Callable    # (trainer, prompts) -> encoding_data
    transformer_forward_fn: Callable  # (trainer, latents, encoding_data, extra_data) -> pred_latents
    apply_fn: Callable           # (trainer, latents, encoding_data) -> images
    init_guidance_fn: Optional[Callable] = None  # (trainer) -> guidance


class NoiseNetworkTrainer:
    """Universal trainer for noise networks with configurable model functions."""
    
    def __init__(
        self,
        reward_losses: List[BaseRewardLoss],
        pipe,
        train_type: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_config: TrainingConfig,
        runtime_config: RuntimeConfig,
        model_functions: ModelFunctions,
        latent_shape: List[int] = [4, 64, 64],
    ):
        self.reward_losses = reward_losses
        self.pipe = pipe
        self.train_type = train_type
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = training_config
        self.runtime = runtime_config
        self.model_fns = model_functions
        
        # Derived properties
        self.latent_shape = latent_shape
        self.latent_dim = math.prod(latent_shape)
        self.preprocess_fn = clip_img_transform(224)
        self.scaler = GradScaler(enabled=True)
        
        # Initialize model-specific guidance if needed
        if self.model_fns.init_guidance_fn:
            self.guidance = self.model_fns.init_guidance_fn(self)
        else:
            self.guidance = None
        
        # Training state
        self.best_loss = torch.inf
        self.metric_list = []
        
    def _generate_latents(self, batch_size: int, iter_step: int = 0) -> torch.Tensor:
        """Generate initial latents based on latent_type."""
        if self.config.latent_type == "one":
            generator = torch.Generator(self.runtime.device).manual_seed(self.runtime.seed)
            return torch.stack([torch.randn(
                self.latent_shape,
                device=self.runtime.device,
                dtype=self.runtime.dtype,
                generator=generator,
            )] * batch_size)
        
        elif self.config.latent_type == "multi":
            generator = torch.Generator(self.runtime.device).manual_seed(self.runtime.seed)
            return torch.randn(
                [batch_size, *self.latent_shape],
                device=self.runtime.device,
                dtype=self.runtime.dtype,
                generator=generator,
            )
        
        elif self.config.latent_type == "inf":
            return torch.randn(
                [batch_size, *self.latent_shape],
                device=self.runtime.device,
                dtype=self.runtime.dtype,
            )
        
        elif self.config.latent_type == "batch":
            generator = torch.Generator(self.runtime.device).manual_seed(
                self.runtime.seed + (iter_step // (self.config.accumulation_steps))
            )
            latent = torch.randn(
                [1, *self.latent_shape],
                device=self.runtime.device,
                dtype=self.runtime.dtype,
                generator=generator,
            )
            return latent.expand(batch_size, *self.latent_shape)
        
        else:
            raise ValueError(f"Unknown latent type: {self.config.latent_type}")

    def _compute_regularization(self, pred_latents: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss."""
        if not self.config.regularize or self.train_type != "noise":
            return torch.tensor(0.0, device=self.runtime.device)
            
        if self.config.reg_type == "l2":
            return torch.mean(torch.square(pred_latents))
        else:
            norm_dims = self.config.norm_dims or list(range(1, len(latents.shape)))
            latent_norm = torch.linalg.vector_norm(latents, dim=norm_dims)
            regularization = (
                0.5 * latent_norm**2 - (self.latent_dim - 1) * torch.log(latent_norm + 1e-8)
            )
            return regularization.mean()

    def _compute_total_loss(self, images: torch.Tensor, prompts: List[str], 
                           pred_latents: Optional[torch.Tensor] = None,
                           latents: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss from rewards and regularization."""
        preprocessed_image = self.preprocess_fn(images)
        total_loss = torch.tensor(0.0, device=self.runtime.device)
        metrics = {}
        
        # Compute reward losses
        for reward_loss in self.reward_losses:
            if self.config.grad_normalization:
                loss = GradNormFunction.gradnorm(reward_loss(preprocessed_image, prompts))
            else:
                loss = reward_loss(preprocessed_image, prompts, images)
            total_loss = total_loss + loss * reward_loss.weighting
            metrics[reward_loss.name] = loss.detach().cpu().item()
        
        # Add regularization
        if pred_latents is not None:
            regularization = self._compute_regularization(pred_latents, latents)
            metrics["regularization"] = regularization.detach().cpu().item()
            total_loss = total_loss + regularization * self.config.regularization_weight
        
        # Add norm metric
        if latents is not None:
            norm_dims = self.config.norm_dims or list(range(1, len(latents.shape)))
            latent_norm = torch.linalg.vector_norm(latents, dim=norm_dims)
            metrics["norm"] = latent_norm.detach().cpu().mean().item()
        
        metrics["total"] = total_loss.detach().cpu().item()
        return total_loss, metrics

    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics across distributed processes."""
        if self.runtime.world_size <= 1:
            return metrics
            
        metric_keys = list(metrics.keys())
        metric_values = [metrics[k] for k in metric_keys]
        metrics_tensor = torch.tensor(metric_values, device=self.runtime.device, dtype=self.runtime.dtype)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        reduced_values = (metrics_tensor / self.runtime.world_size).cpu().tolist()
        
        return {key: reduced_values[i] for i, key in enumerate(metric_keys)}

    def _cleanup_memory(self):
        """Clean up GPU memory."""
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to wandb and console."""
        if not self.runtime.log_metrics:
            return
            
        # Log to wandb
        wandb_metrics = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        wandb.log(wandb_metrics, step=step)
        
        # Log to console
        formatted_metrics = {k: f"{v:.4f}" for k, v in metrics.items()}
        logging.info(f"{prefix}Step {step}: {formatted_metrics}")

    def _load_eval_prompts(self, filepath: str = "assets/example_prompts.txt") -> List[str]:
        """Load evaluation prompts for current rank."""
        with open(filepath, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        
        # Split prompts for current rank
        start_idx = self.runtime.rank * (len(prompts) // self.runtime.world_size)
        end_idx = (self.runtime.rank + 1) * (len(prompts) // self.runtime.world_size)
        return prompts[start_idx:end_idx]

    def step(self, init_latents: torch.Tensor, prompts: List[str], iter_step: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform one training step."""
        # Prepare latents using model-specific function
        latents, extra_data = self.model_fns.prepare_latents_fn(self, init_latents, self.config.batch_size)
        
        # Encode prompts
        encoding_data = self.model_fns.encode_prompt_fn(self, prompts)
        
        pred_latents = None
        if self.train_type == "noise":
            self.pipe.transformer.module.train()
            self.pipe.transformer.module.enable_adapters()
            
            # Forward pass through transformer (with optional checkpointing)
            if self.config.use_checkpoint:
                def callable_fn():
                    return self.model_fns.transformer_forward_fn(self, latents, encoding_data, extra_data)
                pred_latents = torch.utils.checkpoint.checkpoint(callable_fn, use_reentrant=False)
            else:
                pred_latents = self.model_fns.transformer_forward_fn(self, latents, encoding_data, extra_data)
            
            if self.config.enable_modulate_noise:
                latents = pred_latents + latents
            else:
                latents = pred_latents
            self.pipe.transformer.module.disable_adapters()
        
        # Generate images
        images = self.model_fns.apply_fn(self, latents, encoding_data)
        
        # Compute loss
        total_loss, metrics = self._compute_total_loss(images, prompts, pred_latents, latents)
        total_loss /= self.config.accumulation_steps
        
        # Backward pass
        self.pipe.transformer.module.train()
        self.pipe.transformer.module.enable_adapters()
        total_loss.backward()
        
        # Clean up intermediate tensors
        del images, latents, pred_latents
        if isinstance(extra_data, torch.Tensor):
            del extra_data
        if isinstance(encoding_data, (list, tuple)):
            del encoding_data
        
        # Optimizer step
        if (iter_step + 1) % self.config.accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.pipe.transformer.parameters(), max_norm=float('inf'), norm_type=2
            )
            metrics["grad_norm"] = grad_norm.item()
            
            torch.nn.utils.clip_grad_norm_(self.pipe.transformer.parameters(), self.config.grad_clip)
            
            if has_nan_gradients(self.pipe.transformer):
                logging.warning("Skipping batch due to NaN gradients")
                self.optimizer.zero_grad(set_to_none=True)
                return metrics, {}
            
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            metrics["grad_norm"] = 0.0
        
        return metrics, {k: v for k, v in metrics.items() if k in [loss.name for loss in self.reward_losses]}

    def eval_step(self, init_latents: torch.Tensor, batch_size: int, prompts: List[str], 
                  noise_prompts: Optional[List[str]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform one evaluation step."""
        images = []
        eval_metrics = defaultdict(list)
        
        self.pipe.transformer.module.eval()
        with torch.inference_mode():
            for batch_start in range(0, len(prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]
                batch_init_latents = init_latents[batch_start:batch_end]
                
                # Prepare latents
                batch_latents, extra_data = self.model_fns.prepare_latents_fn(
                    self, batch_init_latents, len(batch_prompts)
                )
                
                # Encode prompts
                encoding_data = self.model_fns.encode_prompt_fn(self, batch_prompts)
                
                # Handle noise prompts if provided
                if noise_prompts is not None:
                    batch_noise_prompts = noise_prompts[batch_start:batch_end]
                    noise_encoding_data = self.model_fns.encode_prompt_fn(self, batch_noise_prompts)
                else:
                    noise_encoding_data = encoding_data
                
                # Generate latents if using noise network
                if self.train_type == "noise":
                    self.pipe.transformer.module.enable_adapters()
                    pred_latents = self.model_fns.transformer_forward_fn(
                        self, batch_latents, noise_encoding_data, extra_data
                    )
                    
                    if self.config.enable_modulate_noise:
                        batch_latents = pred_latents + batch_latents
                    else:
                        batch_latents = pred_latents
                    
                    self.pipe.transformer.module.disable_adapters()
                
                # Generate images
                batch_images = self.model_fns.apply_fn(self, batch_latents, encoding_data)
                
                # Compute metrics
                total_loss, metrics = self._compute_total_loss(batch_images, batch_prompts)
                images.append(batch_images.detach())
                
                for key, value in metrics.items():
                    eval_metrics[key].append(value)
        
        # Aggregate metrics
        aggregated_metrics = {key: sum(values) / len(values) for key, values in eval_metrics.items()}
        eval_images = torch.cat(images, dim=0)
        
        self.pipe.transformer.module.train()
        self.pipe.transformer.module.enable_adapters()
        
        # Cleanup
        del batch_images, images, total_loss
        self._cleanup_memory()
        
        return eval_images, aggregated_metrics

    def eval_rewards(self, prompts: List[str], noise_prompts: List[str], 
                    batch_size: int = 8, num_samples: int = 1) -> Dict[str, float]:
        """Evaluate rewards for given prompts."""
        self._cleanup_memory()
        
        # Split prompts across devices
        start_idx = self.runtime.rank * (len(prompts) // self.runtime.world_size)
        end_idx = (self.runtime.rank + 1) * (len(prompts) // self.runtime.world_size)
        device_prompts = prompts[start_idx:end_idx]
        device_noise_prompts = noise_prompts[start_idx:end_idx]
        
        all_rewards = {reward_loss.name: [] for reward_loss in self.reward_losses}
        all_rewards["total"] = []
        
        with torch.inference_mode():
            self.pipe.transformer.module.eval()
            
            for batch_start in range(0, len(device_prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(device_prompts))
                batch_prompts = device_prompts[batch_start:batch_end]
                batch_noise_prompts = device_noise_prompts[batch_start:batch_end]
                
                for i in range(num_samples):
                    # Generate and evaluate
                    generator = torch.Generator(self.runtime.device).manual_seed(self.runtime.seed + i)
                    init_latents = torch.randn(
                        [len(batch_prompts), *self.latent_shape],
                        device=self.runtime.device,
                        dtype=self.runtime.dtype,
                        generator=generator,
                    )
                    images, latents = self._generate_images(init_latents, batch_prompts, batch_noise_prompts)
                    
                    # Calculate rewards
                    total_loss, metrics = self._compute_total_loss(images, batch_prompts)
                    for name, value in metrics.items():
                        if name in all_rewards:
                            all_rewards[name].append(value)
        
        # Synchronize results across processes
        eval_metrics = {}
        total_samples = self.runtime.world_size * len(all_rewards["total"]) if all_rewards["total"] else 0
        
        for name, rewards in all_rewards.items():
            if rewards:
                rewards_tensor = torch.tensor(rewards, device=self.runtime.device)
                if self.runtime.world_size > 1:
                    dist.all_reduce(rewards_tensor, op=dist.ReduceOp.SUM)
                eval_metrics[name] = rewards_tensor.sum().item() / total_samples if total_samples > 0 else 0.0
        
        self.pipe.transformer.module.train()
        self.pipe.transformer.module.enable_adapters()
        self._cleanup_memory()
        return eval_metrics

    def _generate_images(self, init_latents: torch.Tensor, prompts: List[str], 
                        noise_prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate images from latents and prompts."""
        # Prepare latents
        latents, extra_data = self.model_fns.prepare_latents_fn(self, init_latents, len(prompts))
        
        # Encode prompts
        encoding_data = self.model_fns.encode_prompt_fn(self, prompts)
        
        if prompts != noise_prompts:
            noise_encoding_data = self.model_fns.encode_prompt_fn(self, noise_prompts)
        else:
            noise_encoding_data = encoding_data
        
        if self.train_type == "noise":
            self.pipe.transformer.module.enable_adapters()
            pred_latents = self.model_fns.transformer_forward_fn(
                self, latents, noise_encoding_data, extra_data
            )
            
            if self.config.enable_modulate_noise:
                latents = pred_latents + latents
            else:
                latents = pred_latents
            self.pipe.transformer.module.disable_adapters()
        
        images = self.model_fns.apply_fn(self, latents, encoding_data)
        return images, latents

    def save_model(self, save_dir: str, step: Optional[int] = None, is_best: bool = False):
        """Save model checkpoint."""
        if not self.runtime.log_metrics:
            return
            
        suffix = "best_model" if is_best else f"model_step{step}" if step else "final_model"
        adapter_save_directory = os.path.join(save_dir, suffix)
        os.makedirs(adapter_save_directory, exist_ok=True)
        
        self.pipe.transformer.module.enable_adapters()
        model_with_adapter = self.pipe.transformer.module
        
        # Save adapter weights
        peft_model_state_dict = get_peft_model_state_dict(
            model_with_adapter, adapter_name="hypernoise_adapter"
        )
        torch.save(peft_model_state_dict, os.path.join(adapter_save_directory, "adapter_model.bin"))
        
        # Save adapter config
        adapter_config = model_with_adapter.peft_config["hypernoise_adapter"]
        adapter_config.save_pretrained(adapter_save_directory)
        
        loss_info = f" (loss: {self.best_loss:.4f})" if is_best else ""
        logging.info(f"Successfully saved LoRA adapter to {adapter_save_directory}{loss_info}")

    def _periodic_evaluation(self, toy_eval_prompts: List[str], eval_prompts: List[str], 
                           save_dir: str, epoch: int, iter_idx: int, iter_step: int):
        """Perform periodic evaluation during training."""
        # Generate evaluation images
        if save_dir:
            self._generate_and_save_eval_images(toy_eval_prompts, save_dir, epoch, iter_idx)
        
        # Evaluate on full eval set
        geneval_metrics = self.eval_rewards(eval_prompts, eval_prompts, 
                                          self.config.eval_batch_size, num_samples=4)
        red_prompts = ["red"] * len(eval_prompts)
        geneval_metrics_red = self.eval_rewards(eval_prompts, red_prompts, 
                                              self.config.eval_batch_size, num_samples=4)
        
        # Update best model
        if geneval_metrics.get("total", float('inf')) < self.best_loss:
            self.best_loss = geneval_metrics["total"]
            self.save_model(save_dir, is_best=True)
        
        # Log evaluation metrics
        self._log_metrics(geneval_metrics, iter_step, "geneval_")
        self._log_metrics(geneval_metrics_red, iter_step, "geneval_red_")
        
        # Average recent training metrics
        if self.metric_list:
            avg_metrics = {}
            for key in self.metric_list[0].keys():
                avg_metrics[key] = sum([m[key] for m in self.metric_list]) / len(self.metric_list)
            self._log_metrics(avg_metrics, iter_step, "train_")
            self.metric_list = []  # Reset after logging
        
        self._cleanup_memory()

    def _generate_and_save_eval_images(self, prompts: List[str], save_dir: str, epoch: int, iter_idx: int):
        """Generate and save evaluation images."""
        generator = torch.Generator(self.runtime.device).manual_seed(self.runtime.seed)
        init_latents = torch.randn(
            [len(prompts), *self.latent_shape],
            device=self.runtime.device,
            dtype=self.runtime.dtype,
            generator=generator,
        )
        
        # Regular evaluation
        images, _ = self.eval_step(init_latents, self.config.eval_batch_size, prompts)
        self._save_image_grid(images, save_dir, f"grid_epoch{epoch}_{iter_idx}.png")
        
        # Red conditioning evaluation
        red_prompts = ["red"] * len(prompts)
        images_red, _ = self.eval_step(init_latents, self.config.eval_batch_size, prompts, red_prompts)
        self._save_image_grid(images_red, save_dir, f"grid_epoch{epoch}_{iter_idx}_red.png")

    def _save_image_grid(self, images: torch.Tensor, save_dir: str, filename: str):
        """Save image grid with distributed gathering."""
        if self.runtime.world_size > 1:
            dist.barrier()
            images = images.contiguous()
            tensor_list = [torch.zeros_like(images) for _ in range(self.runtime.world_size)]
            dist.all_gather(tensor_list, images)
            full_images = torch.cat(tensor_list, dim=0).cpu()
            dist.barrier()
        else:
            full_images = images.cpu()
        
        if self.runtime.log_metrics:
            save_path = os.path.join(save_dir, filename)
            save_image_grid(full_images, save_path, nrow=int(math.sqrt(full_images.size(0))))

    def train(self, prompt_loader: torch.utils.data.DataLoader, eval_prompts: List[str], 
              save_dir: Optional[str] = None, save_images: bool = False):
        """Main training loop."""
        if self.runtime.log_metrics:
            logging.info(f"Training noise network for {self.config.epochs} epochs.")
        
        toy_eval_prompts = self._load_eval_prompts()
        self.optimizer.zero_grad(set_to_none=True)
        
        for epoch in range(self.config.epochs):
            if self.runtime.world_size > 1:
                dist.barrier()
                prompt_loader.sampler.set_epoch(epoch)
            
            self._cleanup_memory()
            
            for iter_idx, prompts in enumerate(prompt_loader):
                iter_step = iter_idx + epoch * len(prompt_loader)
                init_time = time.time()
                
                # Prepare batch
                if self.config.one_prompt_per_batch:
                    prompts = [prompts[0]] * self.config.batch_size
                
                init_latents = self._generate_latents(self.config.batch_size, iter_step)
                
                # Training step
                to_log_metrics, curr_rewards = self.step(init_latents, prompts, iter_step)
                step_time = time.time() - init_time
                
                # Sync metrics across processes
                to_log_metrics = self._sync_metrics(to_log_metrics)
                curr_rewards = self._sync_metrics(curr_rewards)
                
                # Log training metrics
                self._log_metrics(to_log_metrics, iter_step)
                self.metric_list.append(dict(to_log_metrics))
                
                if self.runtime.log_metrics:
                    post_time = time.time() - init_time
                    logging.info(f"Step time: {step_time:.4f} seconds, Post-processing time: {post_time:.4f} seconds")
                
                # Periodic evaluation and saving
                if save_images and (iter_step + 1) % self.config.log_every == 0:
                    self._periodic_evaluation(toy_eval_prompts, eval_prompts, save_dir, epoch, iter_idx, iter_step)
                
                # Periodic checkpointing
                if (iter_step + 1) % self.config.save_every == 0:
                    self.save_model(save_dir, iter_step)
            
            # Log epoch metrics
            if self.runtime.log_metrics and self.metric_list:
                epoch_metrics = {}
                for key in self.metric_list[0].keys():
                    epoch_metrics[key] = sum([m[key] for m in self.metric_list]) / len(self.metric_list)
                
                epoch_metrics_logged = {f"epoch_{k}": v for k, v in epoch_metrics.items()}
                wandb.log(epoch_metrics_logged, step=int(epoch * len(prompt_loader)))
                formatted_epoch_metrics = {k: f"{v:.4f}" for k, v in epoch_metrics.items()}
                logging.info(f"Epoch {epoch} Metrics: {formatted_epoch_metrics}")
        
        # Final save
        if save_dir:
            self.save_model(save_dir)
        
        return None


# Model-specific function implementations

def create_sana_functions() -> ModelFunctions:
    """Create model functions for SANA."""
    
    def prepare_latents_fn(trainer, init_latents, batch_size):
        # SANA uses latents directly
        return init_latents, None
    
    def encode_prompt_fn(trainer, prompts):
        return trainer.pipe.encode_prompt(prompt=prompts, device=trainer.runtime.device)
    
    def transformer_forward_fn(trainer, latents, encoding_data, extra_data):
        prompt_embeds, prompt_attention_mask = encoding_data
        # account for batches smaller than batch size
        guidance = trainer.guidance[:latents.shape[0]]
        
        return trainer.pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            guidance=guidance,
            timestep=torch.tensor([1.0], device=trainer.runtime.device, dtype=trainer.runtime.dtype),
        ).sample
    
    def apply_fn(trainer, latents, encoding_data):
        prompt_embeds, prompt_attention_mask = encoding_data
        return trainer.pipe.apply(
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )
    
    def init_guidance_fn(trainer):
        guidance = torch.full([1], 4.5, device=trainer.runtime.device, dtype=trainer.runtime.dtype)
        guidance = guidance.expand(trainer.config.batch_size).to(trainer.runtime.dtype)
        return guidance * trainer.pipe.transformer.module.config.guidance_embeds_scale
    
    return ModelFunctions(
        prepare_latents_fn=prepare_latents_fn,
        encode_prompt_fn=encode_prompt_fn,
        transformer_forward_fn=transformer_forward_fn,
        apply_fn=apply_fn,
        init_guidance_fn=init_guidance_fn,
    )


def create_flux_functions() -> ModelFunctions:
    """Create model functions for Flux."""
    
    def prepare_latents_fn(trainer, init_latents, batch_size):
        generator = torch.Generator(trainer.runtime.device).manual_seed(trainer.runtime.seed)
        return trainer.pipe.prepare_latents(
            batch_size,
            trainer.pipe.transformer.module.config.in_channels // 4,
            512, 512,  # Use 512x512 for Flux
            trainer.runtime.dtype,
            trainer.runtime.device,
            generator,
            init_latents,
        )
    
    def encode_prompt_fn(trainer, prompts):
        return trainer.pipe.encode_prompt(
            prompt=prompts, prompt_2=None, device=trainer.runtime.device
        )
    
    def transformer_forward_fn(trainer, latents, encoding_data, extra_data):
        prompt_embeds, pooled_prompt_embeds, text_ids = encoding_data
        latent_image_ids = extra_data
        
        return trainer.pipe.transformer(
            hidden_states=latents,
            timestep=torch.tensor([1.0]).to(trainer.runtime.device, dtype=trainer.runtime.dtype),
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
    
    def apply_fn(trainer, latents, encoding_data):
        prompt_embeds, pooled_prompt_embeds, text_ids = encoding_data
        return trainer.pipe.apply(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=0.0,
            num_inference_steps=1,
        )
    
    return ModelFunctions(
        prepare_latents_fn=prepare_latents_fn,
        encode_prompt_fn=encode_prompt_fn,
        transformer_forward_fn=transformer_forward_fn,
        apply_fn=apply_fn,
        init_guidance_fn=None,  # Flux doesn't need guidance initialization
    )


def create_trainer(model_type: str, **kwargs) -> NoiseNetworkTrainer:
    """Factory function to create trainer with appropriate model functions."""
    
    # Set model-specific configurations
    if model_type.lower() == "sana":
        model_functions = create_sana_functions()
        # SANA-specific config adjustments
        if 'training_config' in kwargs:
            kwargs['training_config'].norm_dims = [1, 2, 3]  # SANA norm dimensions
    
    elif model_type.lower() == "flux":
        model_functions = create_flux_functions()
        # Flux-specific config adjustments
        if 'training_config' in kwargs:
            kwargs['training_config'].norm_dims = [1, 2]  # Flux norm dimensions
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return NoiseNetworkTrainer(model_functions=model_functions, **kwargs)


# Utility functions
def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 8, scale_factor: float = 0.25):
    """Save image grid with optional scaling."""
    images = images.detach().cpu().float()
    images = torch.clamp(images, 0, 1)
    
    grid = make_grid(images, nrow=nrow, normalize=False, padding=2)
    grid_image = torchvision.transforms.ToPILImage()(grid)
    
    if scale_factor != 1.0:
        new_width = int(grid_image.width * scale_factor)
        new_height = int(grid_image.height * scale_factor)
        grid_image = grid_image.resize((new_width, new_height), resample=PIL.Image.BILINEAR)
    
    grid_image.save(save_path, quality=20)


def has_nan_gradients(model: torch.nn.Module) -> bool:
    """Check if model has NaN gradients."""
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False