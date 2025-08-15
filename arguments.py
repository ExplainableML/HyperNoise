import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Hypernoise Reward Optimization.")

    # update paths here
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="HF cache directory",
        default=".",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save images",
        default="outputs",
    )

    # model and optim
    parser.add_argument("--model", type=str, help="Model to use", default="sana")
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument(
        "--optim",
        choices=["sgd", "adam", "adamw", "lbfgs"],
        default="sgd",
        help="Optimizer to be used",
    )
    parser.add_argument(
        "--grad_clip", type=float, help="Gradient clipping", default=1.0
    )
    parser.add_argument("--seed", type=int, help="Seed to use", default=0)

    # reward losses
    parser.add_argument("--disable_hps", default=True, action="store_false",dest="enable_hps")
    parser.add_argument(
        "--hps_weighting", type=float, help="Weighting for HPS", default=5.0
    )
    parser.add_argument("--disable_imagereward", default=True, action="store_false",dest='enable_imagereward')
    parser.add_argument(
        "--imagereward_weighting",
        type=float,
        help="Weighting for ImageReward",
        default=1.0,
    )
    parser.add_argument("--disable_clip", default=True, action="store_false",dest='enable_clip')
    parser.add_argument(
        "--clip_weighting", type=float, help="Weighting for CLIP", default=0.01
    )
    parser.add_argument("--disable_pickscore", default=True, action="store_false",dest='enable_pickscore')
    parser.add_argument(
        "--pickscore_weighting",
        type=float,
        help="Weighting for PickScore",
        default=0.05,
    )
    parser.add_argument("--disable_aesthetic", default=False, action="store_false",dest='enable_aesthetic')
    parser.add_argument(
        "--aesthetic_weighting",
        type=float,
        help="Weighting for Aesthetic",
        default=0.0,
    )
    parser.add_argument("--enable_red", default=False, action="store_true")
    parser.add_argument(
        "--red_weighting", type=float, help="Weighting for Redness", default=1.0
    )
    parser.add_argument("--disable_reg", default=True, action="store_false",dest='enable_reg')
    parser.add_argument(
        "--reg_type", type=str, help="Regularization type", default="l2"
    )
    parser.add_argument(
        "--reg_weight", type=float, help="Regularization weight", default=0.5
    )
    parser.add_argument("--memsave", default=False, action="store_true")
    parser.add_argument(
        "--task",
        type=str,
        help="Task to run",
        default="single",
        choices=[
            "example-prompts",
            "geneval",
            "pickapic",
            "all",
        ],
    )
    parser.add_argument("--log_every", type=int, default=2500)
    parser.add_argument("--save_every", type=int, default=25000)
    # noise network
    parser.add_argument("--disable_modulate_noise", default=True, action="store_false", dest='enable_modulate_noise')
    parser.add_argument("--latent_type", type=str, help="Latent type", default="inf")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank", default=128)
    parser.add_argument("--alpha_multiplier", type=int, help="LoRA alpha multiplier", default=2)
    parser.add_argument("--last_layer_name", type=str, help="Pretrained path", default="proj_out") # change to conv_out for sd-turbo
    # optimization
    parser.add_argument("--epochs", type=int, help="Epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--warmup_steps", type=int, help="Warmup steps", default=10)
    parser.add_argument("--accumulation_steps", type=int, help="Accumulation steps", default=1)
    parser.add_argument("--one_prompt_per_batch", default=False, action="store_true")
    parser.add_argument("--grad_normalization", default=False, action="store_true")
    parser.add_argument("--train_type", type=str, help="Train type", default="noise")
    parser.add_argument(
        "--use_checkpoint", 
        action="store_true",
        help="Use gradient checkpointing to save memory during training"
    )
    args = parser.parse_args()
    return args
