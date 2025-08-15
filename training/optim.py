import torch
from diffusers.optimization import get_scheduler
import bitsandbytes.optim as bnb_optim

def get_optimizer(
    optimizer_name: str, params: torch.Tensor, lr: float, nesterov: bool
):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, nesterov=nesterov, momentum=0.9)
        # optimizer = bnb_optim.SGD(params, lr=lr, nesterov=nesterov, momentum=0.9)
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=10,
            history_size=3,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")
    return optimizer

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, n_warmup_steps: int, world_size: int
):
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=n_warmup_steps * world_size,
    )
    return lr_scheduler