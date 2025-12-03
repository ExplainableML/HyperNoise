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
    one_prompt_per_batch: bool = False
    use_checkpoint: bool = True  # Whether to use gradient checkpointing
    viz_noise: bool = False # Whether to visualize the modulated noise
    compute_geneval_rewards: bool = True  # Whether to compute geneval metrics


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
    noise_network_forward_fn: Callable  # (trainer, latents, encoding_data, extra_data) -> pred_latents
    get_noise_network: Callable  # (trainer) -> noise_network
    apply_fn: Callable           # (trainer, latents, encoding_data) -> images
    model_name: str
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
            latent_norm = torch.linalg.vector_norm(latents, dim=list(range(1, len(pred_latents.shape))))
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
            latent_norm = torch.linalg.vector_norm(latents, dim=list(range(1, len(latents.shape))))
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

    def _load_visualization_eval_prompts(self, filepath: str = "assets/example_prompts.txt") -> List[str]:
        """Load evaluation prompts."""
        with open(filepath, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        return prompts

    def step(self, init_latents: torch.Tensor, prompts: List[str], iter_step: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform one training step."""
        # Prepare latents using model-specific function
        latents, extra_data = self.model_fns.prepare_latents_fn(self, init_latents, self.config.batch_size)
        
        # Encode prompts
        encoding_data = self.model_fns.encode_prompt_fn(self, prompts)
        
        pred_latents = None
        if self.train_type == "noise":
            self.model_fns.get_noise_network(self).module.train()
            self.model_fns.get_noise_network(self).module.enable_adapters()

            # Forward pass through transformer (with optional checkpointing)
            if self.config.use_checkpoint:
                def callable_fn():
                    return self.model_fns.noise_network_forward_fn(self, latents, encoding_data, extra_data)
                pred_latents = torch.utils.checkpoint.checkpoint(callable_fn, use_reentrant=False)
            else:
                pred_latents = self.model_fns.noise_network_forward_fn(self, latents, encoding_data, extra_data)
            
            if self.config.enable_modulate_noise:
                latents = pred_latents + latents
            else:
                latents = pred_latents
            self.model_fns.get_noise_network(self).module.disable_adapters()

        # Generate images
        images = self.model_fns.apply_fn(self, latents, encoding_data)
        
        # Compute loss
        total_loss, metrics = self._compute_total_loss(images, prompts, pred_latents, latents)
        total_loss /= self.config.accumulation_steps
        
        # Backward pass
        self.model_fns.get_noise_network(self).module.train()
        self.model_fns.get_noise_network(self).module.enable_adapters()
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
                self.model_fns.get_noise_network(self).parameters(), max_norm=float('inf'), norm_type=2
            )
            metrics["grad_norm"] = grad_norm.item()

            torch.nn.utils.clip_grad_norm_(self.model_fns.get_noise_network(self).parameters(), self.config.grad_clip)

            if has_nan_gradients(self.model_fns.get_noise_network(self)):
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

    def _generate_images(self, init_latents: torch.Tensor, prompts: List[str], 
                        noise_prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            self.model_fns.get_noise_network(self).module.enable_adapters()
            pred_latents = self.model_fns.noise_network_forward_fn(
                self, latents, noise_encoding_data, extra_data
            )
            
            if self.config.enable_modulate_noise:
                latents = pred_latents + latents
                delta_latents = pred_latents
            else:
                delta_latents = pred_latents - latents
                latents = pred_latents
            self.model_fns.get_noise_network(self).module.disable_adapters()
        else:
            delta_latents = latents
        
        images = self.model_fns.apply_fn(self, latents, encoding_data)
        return images, latents, delta_latents

    def gen_images_and_eval_rewards(self, prompts: List[str], noise_prompts: List[str], 
                    batch_size: int = 8, num_samples: int = 1, return_imgs: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Generate images and evaluate rewards for given prompts."""
        self._cleanup_memory()
        
        # Split prompts across devices
        start_idx = self.runtime.rank * (len(prompts) // self.runtime.world_size)
        end_idx = (self.runtime.rank + 1) * (len(prompts) // self.runtime.world_size)
        device_prompts = prompts[start_idx:end_idx]
        device_noise_prompts = noise_prompts[start_idx:end_idx]

        all_rewards = {reward_loss.name: [] for reward_loss in self.reward_losses}
        all_rewards["total"] = []
        images = []
        noises = []
        delta_noises = []
        
        with torch.inference_mode():
            self.model_fns.get_noise_network(self).module.eval()

            for batch_start in range(0, len(device_prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(device_prompts))
                batch_prompts = device_prompts[batch_start:batch_end]
                batch_noise_prompts = device_noise_prompts[batch_start:batch_end]
                
                for i in range(num_samples):
                    # Generate and evaluate
                    generator = torch.Generator(self.runtime.device).manual_seed(self.runtime.seed + i)
                    if self.config.latent_type == "one":
                        init_latents = torch.stack([torch.randn(
                            self.latent_shape,
                            device=self.runtime.device,
                            dtype=self.runtime.dtype,
                            generator=generator,
                        )] * len(batch_prompts))
                    else:
                        init_latents = torch.randn(
                            [len(batch_prompts), *self.latent_shape],
                            device=self.runtime.device,
                            dtype=self.runtime.dtype,
                            generator=generator,
                        )
                    batch_images, batch_noises, batch_delta_noises = self._generate_images(init_latents, batch_prompts, batch_noise_prompts)
                    if return_imgs:
                        images.append(batch_images)
                        if self.model_fns.model_name == "sd-turbo":
                            batch_noises = batch_noises / self.pipe.vae.config.scaling_factor
                            batch_delta_noises = batch_delta_noises / self.pipe.vae.config.scaling_factor
                        elif self.model_fns.model_name == "flux":
                            # reshape noises accordingly
                            batch_noises = self.pipe._unpack_latents(
                                batch_noises, 512, 512, self.pipe.vae.config.scaling_factor
                            )
                            batch_noises = (
                                batch_noises / self.pipe.vae.config.scaling_factor
                            ) + self.pipe.vae.config.shift_factor
                            batch_delta_noises = self.pipe._unpack_latents(
                                batch_delta_noises, 512, 512, self.pipe.vae.config.scaling_factor
                            )
                            batch_delta_noises = (
                                batch_delta_noises / self.pipe.vae.config.scaling_factor
                            ) + self.pipe.vae.config.shift_factor
                        decoded_noise = self.pipe.vae.decode(batch_noises).sample
                        decoded_noise = (decoded_noise / 2 + 0.5).clamp(0, 1)
                        decoded_delta_noise = self.pipe.vae.decode(batch_delta_noises).sample
                        decoded_delta_noise = (decoded_delta_noise / 2 + 0.5).clamp(0, 1)
                        
                        noises.append(decoded_noise)
                        delta_noises.append(decoded_delta_noise)

                    # Calculate rewards
                    _, metrics = self._compute_total_loss(batch_images, batch_prompts)
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
        
        if return_imgs:
            images = torch.cat(images, dim=0)
            noises = torch.cat(noises, dim=0)
            delta_noises = torch.cat(delta_noises, dim=0)
            
            if self.runtime.world_size > 1:
                dist.barrier()
                
                images = images.contiguous()
                noises = noises.contiguous()
                delta_noises = delta_noises.contiguous()
                images_list = [torch.zeros_like(images) for _ in range(self.runtime.world_size)]
                noises_list = [torch.zeros_like(noises) for _ in range(self.runtime.world_size)]
                delta_noises_list = [torch.zeros_like(delta_noises) for _ in range(self.runtime.world_size)]
                dist.all_gather(images_list, images)
                dist.all_gather(noises_list, noises)
                dist.all_gather(delta_noises_list, delta_noises)
                
                if self.runtime.rank == 0:
                    images = torch.cat(images_list, dim=0)
                    noises = torch.cat(noises_list, dim=0)
                    delta_noises = torch.cat(delta_noises_list, dim=0)
                else:
                    images = torch.empty(0)
                    noises = torch.empty(0)
                    delta_noises = torch.empty(0)
                    
                dist.barrier()
        else:
            images = torch.empty(0)
            noises = torch.empty(0)
            delta_noises = torch.empty(0)

        self.model_fns.get_noise_network(self).module.train()
        self.model_fns.get_noise_network(self).module.enable_adapters()
        self._cleanup_memory()
        return images, noises, delta_noises, eval_metrics

    def save_model(self, save_dir: str, step: Optional[int] = None, is_best: bool = False):
        """Save model checkpoint."""
        if not self.runtime.log_metrics:
            return 
            
        suffix = "best_model" if is_best else f"model_step{step}" if step else "final_model"
        adapter_save_directory = os.path.join(save_dir, suffix)
        os.makedirs(adapter_save_directory, exist_ok=True)

        self.model_fns.get_noise_network(self).module.enable_adapters()
        model_with_adapter = self.model_fns.get_noise_network(self).module

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
        del model_with_adapter  # Free up memory

    def _periodic_evaluation(self, toy_eval_prompts: List[str], eval_prompts: List[str], 
                           save_dir: str, epoch: int, iter_idx: int, iter_step: int):
        """Perform periodic evaluation during training."""
        # Generate evaluation images
        if save_dir:
            toy_metrics, toy_metrics_red = self._generate_and_save_eval_images(toy_eval_prompts, save_dir, epoch, iter_idx)
            total_loss = toy_metrics.get("total", float('inf'))
            self._log_metrics(toy_metrics, iter_step, "toy_eval_")
            self._log_metrics(toy_metrics_red, iter_step, "toy_eval_red_")
    
        if self.config.compute_geneval_rewards:
            # Evaluate on full eval set
            _, _, geneval_metrics = self.gen_images_and_eval_rewards(eval_prompts, eval_prompts, 
                                            self.config.eval_batch_size, num_samples=4, return_imgs=False)
            red_prompts = ["red"] * len(eval_prompts)
            _, _, geneval_metrics_red = self.gen_images_and_eval_rewards(eval_prompts, red_prompts, 
                                                self.config.eval_batch_size, num_samples=4, return_imgs=False)
            
            total_loss = geneval_metrics.get("total", float('inf'))
        
            # Log evaluation metrics
            self._log_metrics(geneval_metrics, iter_step, "geneval_")
            self._log_metrics(geneval_metrics_red, iter_step, "geneval_red_")
        
        # Average recent training metrics
        if self.runtime.log_metrics and self.metric_list:
            avg_metrics = {}
            for key in self.metric_list[0].keys():
                avg_metrics[key] = sum([m[key] for m in self.metric_list]) / len(self.metric_list)
            self._log_metrics(avg_metrics, iter_step, "train_")

        # Update best model
        if self.runtime.log_metrics:
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.save_model(save_dir, is_best=True)

        self._cleanup_memory()

    def _generate_and_save_eval_images(self, prompts: List[str], save_dir: str, epoch: int, iter_idx: int):
        """Generate and save evaluation images."""
        # Regular evaluation
        images, noises, delta_noises, rewards = self.gen_images_and_eval_rewards(prompts, prompts, self.config.eval_batch_size, num_samples=1, return_imgs=True)
        self._save_image_grid(images, save_dir, f"grid_epoch{epoch}_{iter_idx}.png")
        if self.config.viz_noise:
            self._save_image_grid(noises, save_dir, f"grid_epoch{epoch}_{iter_idx}_decoded_noise.png")
            self._save_image_grid(delta_noises, save_dir, f"grid_epoch{epoch}_{iter_idx}_decoded_delta_noise.png")

        # Red conditioning evaluation
        red_prompts = ["red"] * len(prompts)
        images_red, noises_red, delta_noises_red, rewards_red = self.gen_images_and_eval_rewards(prompts, red_prompts, self.config.eval_batch_size, num_samples=1, return_imgs=True)
        self._save_image_grid(images_red, save_dir, f"grid_epoch{epoch}_{iter_idx}_red.png")
        if self.config.viz_noise:
            self._save_image_grid(noises_red, save_dir, f"grid_epoch{epoch}_{iter_idx}_red_decoded_noise.png")
            self._save_image_grid(delta_noises_red, save_dir, f"grid_epoch{epoch}_{iter_idx}_red_decoded_delta_noise.png")
        
        return rewards, rewards_red

    def _save_image_grid(self, images: torch.Tensor, save_dir: str, filename: str):
        if self.runtime.log_metrics:
            save_path = os.path.join(save_dir, filename)
            save_image_grid(images, save_path, nrow=int(math.sqrt(images.size(0))))

    def train(self, prompt_loader: torch.utils.data.DataLoader, eval_prompts: List[str], 
              save_dir: Optional[str] = None, save_images: bool = False):
        """Main training loop."""
        if self.runtime.log_metrics:
            logging.info(f"Training noise network for {self.config.epochs} epochs.")
        
        toy_eval_prompts = self._load_visualization_eval_prompts()
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
                if save_images and (iter_step) % self.config.log_every == 0:
                    self._periodic_evaluation(toy_eval_prompts, eval_prompts, save_dir, epoch, iter_idx, iter_step)
                
                # Periodic checkpointing
                if (iter_step + 1) % self.config.save_every == 0:
                    self.save_model(save_dir, iter_step)
            
            # Log epoch metrics
            if self.runtime.log_metrics and self.metric_list:
                epoch_metrics = {}
                for key in self.metric_list[0].keys():
                    epoch_metrics[key] = sum([m[key] for m in self.metric_list]) / len(self.metric_list)
                    
                self._log_metrics(epoch_metrics, step=int(epoch * len(prompt_loader)), prefix="epoch_")
                self.metric_list = []
        
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
    
    def noise_network_forward_fn(trainer, latents, encoding_data, extra_data):
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
        noise_network_forward_fn=noise_network_forward_fn,
        apply_fn=apply_fn,
        init_guidance_fn=init_guidance_fn,
        get_noise_network=lambda trainer: trainer.pipe.transformer,
        model_name="sana",
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
    
    def noise_network_forward_fn(trainer, latents, encoding_data, extra_data):
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
        noise_network_forward_fn=noise_network_forward_fn,
        apply_fn=apply_fn,
        get_noise_network=lambda trainer: trainer.pipe.transformer,
        init_guidance_fn=None,  # Flux doesn't need guidance initialization
        model_name="flux",
    )


def create_sdturbo_functions() -> ModelFunctions:
    """Create model functions for SDTurbo."""
    
    def prepare_latents_fn(trainer, init_latents, batch_size):
        # SDTurbo uses latents directly
        return init_latents, None
    
    def encode_prompt_fn(trainer, prompts):
        prompt_embeds, _ = trainer.pipe.encode_prompt(
            prompt=prompts,
            device=trainer.runtime.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=False,
        )
        return prompt_embeds
    
    def noise_network_forward_fn(trainer, latents, encoding_data, extra_data):
        prompt_embeds = encoding_data
        return trainer.pipe.unet(
            sample=latents,
            timestep=999.0,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
    
    def apply_fn(trainer, latents, encoding_data):
        prompt_embeds = encoding_data
        return trainer.pipe.apply(
            latents=latents,
            prompt_embeds=prompt_embeds,
            num_inference_steps=1,
            guidance_scale=0.0,
        )
    
    return ModelFunctions(
        prepare_latents_fn=prepare_latents_fn,
        encode_prompt_fn=encode_prompt_fn,
        noise_network_forward_fn=noise_network_forward_fn,
        apply_fn=apply_fn,
        get_noise_network=lambda trainer: trainer.pipe.unet,
        init_guidance_fn=None,
        model_name="sd-turbo",
    )


def create_trainer(model_type: str, **kwargs) -> NoiseNetworkTrainer:
    """Factory function to create trainer with appropriate model functions."""
    
    # Set model-specific configurations
    if model_type.lower() == "sana":
        model_functions = create_sana_functions()
    elif model_type.lower() == "flux":
        model_functions = create_flux_functions()
    elif model_type.lower() == "sd-turbo":
        model_functions = create_sdturbo_functions()
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