import json
import os
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int):
        return self.prompts[idx]

def get_prompt_loader(prompts: List[str], batch_size: int, world_size: int, global_rank: int):
    dataset = PromptDataset(prompts)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank, 
        shuffle=True,
        drop_last=False,
    )
    prompt_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return prompt_loader



def get_datasets(args, global_rank: int, world_size: int) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Retrieves the appropriate datasets based on the task."""
    if args.task == "all":
        with open('assets/pickapic_prompts.json', 'r') as file:
            data = json.load(file)
        pickapic_prompts = list(data.values())
        t2i_compbench_prompts = load_compbench_prompts()
        with open("assets/ABC-6K.txt", "r") as f:
            abc_prompts = [line.strip() for line in f.readlines()]
        prompts = t2i_compbench_prompts + abc_prompts + pickapic_prompts
        if args.one_prompt_per_batch:
            prompt_loader: DataLoader = get_prompt_loader(prompts, 1, world_size, global_rank)
        else:
            prompt_loader: DataLoader = get_prompt_loader(prompts, args.batch_size // world_size, world_size, global_rank)
        return prompt_loader
    elif args.task == "pickapic":
        with open('assets/pickapic_prompts.json', 'r') as file:
            data = json.load(file)
        pickapic_prompts = list(data.values())
        if args.one_prompt_per_batch:
            prompt_loader: DataLoader = get_prompt_loader(pickapic_prompts, 1, world_size, global_rank)
        else:
            prompt_loader: DataLoader = get_prompt_loader(pickapic_prompts, args.batch_size // world_size, world_size, global_rank)
        return prompt_loader
    elif args.task == "geneval":
        geneval_prompts = get_prompts_from_jsonl("../geneval/prompts/evaluation_metadata.jsonl")
        if args.one_prompt_per_batch:
            prompt_loader: DataLoader = get_prompt_loader(geneval_prompts, 1, world_size, global_rank)
        else:
            prompt_loader: DataLoader = get_prompt_loader(geneval_prompts, args.batch_size // world_size, world_size, global_rank)
        return prompt_loader
    elif args.task == "example-prompts":
        with open("assets/example_prompts.txt", "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        if args.one_prompt_per_batch:
            prompt_loader: DataLoader = get_prompt_loader(prompts, 1, world_size, global_rank)
        else:
            prompt_loader: DataLoader = get_prompt_loader(prompts, args.batch_size // world_size, world_size, global_rank)
        return prompt_loader
    else:
        raise ValueError(f"Unknown task: {args.task}")



def load_compbench_prompts():
    train_files = [
        "color_train.txt", "complex_train.txt", "texture_train.txt",
        "shape_train.txt", "spatial_train.txt",
        "non_spatial_train.txt", "numeracy_train.txt",
    ]
    all_prompts = []
    
    for filename in train_files:
        file_path = os.path.join("../T2I-CompBench/examples/dataset/", filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            all_prompts.extend([line.strip() for line in f if line.strip()])
    return all_prompts


def get_prompts_from_jsonl(file_path: str) -> List[str]:
    """Extracts prompts from a JSONL file."""
    prompts: List[str] = []
    with open(file_path) as fp:
        for line in fp:
            metadata: Dict[str, str] = json.loads(line)
            if 'prompt' in metadata:
                prompts.append(metadata['prompt'])
    return prompts