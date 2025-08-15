import logging
from typing import List, Optional
import torch
import torch.nn as nn
import diffusers
from safetensors.torch import load_file

import logging
import types
from typing import Any, Optional
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
    Transformer2DModel,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from peft.tuners.lora.layer import Linear as LoraLinear

from models.RewardStableDiffusion import RewardStableDiffusion
from models.RewardSana import RewardSanaPipeline
from models.RewardFlux import RewardFluxPipeline


def get_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    cache_dir: str,
    memsave: bool = False,
    enable_sequential_cpu_offload: bool = False,
):
    logging.info(f"Loading model: {model_name}")
    if model_name == "sd-turbo":
        pipe = RewardStableDiffusion.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=dtype,
            variant="fp16",
            cache_dir=cache_dir,
        ).to(device, dtype)
        
        # optionally enable memsave_torch
        if memsave:
            import memsave_torch.nn

            self.vae = memsave_torch.nn.convert_to_memory_saving(self.vae)
            self.unet = memsave_torch.nn.convert_to_memory_saving(self.unet)
            self.text_encoder = memsave_torch.nn.convert_to_memory_saving(
                self.text_encoder
            )
        self.text_encoder.gradient_checkpointing_enable()
    elif model_name == "sana":
        pipe = RewardSanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        ).to(device, dtype)
        if memsave:
            import memsave_torch.nn
            
            pipe.transformer = memsave_torch.nn.convert_to_memory_saving(pipe.transformer)
            pipe.vae = memsave_torch.nn.convert_to_memory_saving(pipe.vae)
            pipe.text_encoder = memsave_torch.nn.convert_to_memory_saving(pipe.text_encoder)
        pipe.text_encoder._set_gradient_checkpointing(True)
    elif model_name == "flux":
        pipe = RewardFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        ).to(device, torch.bfloat16)
        # optionally enable memsave_torch
        if memsave:
            import memsave_torch.nn

            pipe.vae = memsave_torch.nn.convert_to_memory_saving(pipe.vae)
            pipe.transformer = memsave_torch.nn.convert_to_memory_saving(pipe.transformer)
            pipe.text_encoder = memsave_torch.nn.convert_to_memory_saving(pipe.text_encoder)
            pipe.text_encoder_2 = memsave_torch.nn.convert_to_memory_saving(pipe.text_encoder_2)
        pipe.text_encoder.gradient_checkpointing_enable()
        pipe.text_encoder_2.gradient_checkpointing_enable()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    return pipe


def find_all_lora_candidates(model):
    lora_candidates = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    
    # Look for Linear and Conv layers, including those nested inside other modules
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
            
        # Direct Linear/Conv layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            lora_candidates.add(name)
            
        # Look for attention modules
        elif "attn" in name.lower():
            for subname, submodule in module.named_modules():
                if isinstance(submodule, torch.nn.Linear):
                    full_name = f"{name}.{subname}" if subname else name
                    lora_candidates.add(full_name)
                    
        # Look inside transformer blocks
        elif "transformer" in name.lower() and not any(parent in name for parent in lora_candidates):
            for subname, submodule in module.named_modules():
                if isinstance(submodule, torch.nn.Linear) and not any(mm_keyword in f"{name}.{subname}" for mm_keyword in multimodal_keywords):
                    full_name = f"{name}.{subname}" if subname else name
                    lora_candidates.add(full_name)
                    
        # Include caption projection
        elif "caption_projection" in name:
            for subname, submodule in module.named_modules():
                if isinstance(submodule, torch.nn.Linear):
                    full_name = f"{name}.{subname}" if subname else name
                    lora_candidates.add(full_name)
    
    return sorted(list(lora_candidates))


def scaled_base_lora_forward(self, x, *args, **kwargs):
    """
    Custom forward pass for a LoRA layer.
    Scales the base layer output when adapters are active.
    Uses the currently active hypernoise adapter.
    """
    base_result = self.base_layer(x, *args, **kwargs)
    if self.disable_adapters:
        return base_result
    else:
        lora_A = self.lora_A["hypernoise_adapter"]
        lora_B = self.lora_B["hypernoise_adapter"]
        scaling = self.scaling["hypernoise_adapter"]
        dropout = self.lora_dropout["hypernoise_adapter"]

        lora_A_out = dropout(x)
        lora_A_out = lora_A(x)
        lora_B_out = lora_B(lora_A_out)

        return lora_B_out * scaling



def patch_lora_layer(pipe, target_layer_name: str, global_rank: int):
    """Patch a specific LoRA layer with custom forward function using your existing function."""
    if global_rank == 0:
        logging.info(f"Attempting to patch LoRA layer: '{target_layer_name}'")
    
    patched_count = 0
    
    for name, module in pipe.transformer.named_modules():
        if name == target_layer_name and isinstance(module, LoraLinear):
            if global_rank == 0:
                logging.info(f"Found target LoRA layer '{name}' (Type: {type(module)}). Applying patch.")
            
            # Use your existing scaled_base_lora_forward function
            module.forward = types.MethodType(scaled_base_lora_forward, module)
            patched_count += 1
            break

    if global_rank == 0:
        if patched_count > 0:
            logging.info(f"Successfully patched {patched_count} layer(s) named '{target_layer_name}'.")
            
            # Verification
            try:
                target_module = dict(pipe.transformer.named_modules())[target_layer_name]
                if target_module.forward.__func__ == scaled_base_lora_forward:
                    logging.info(f"Verification successful: '{target_layer_name}'.forward is now scaled_base_lora_forward.")
                else:
                    logging.warning(f"Verification failed: '{target_layer_name}'.forward is not the new function.")
            except KeyError:
                logging.error(f"Verification error: Could not access module named '{target_layer_name}'.")
        else:
            logging.error(f"No LoRA layer with name '{target_layer_name}' was found.")