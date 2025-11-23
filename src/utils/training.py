import torch

def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_gpu_memory(stage: str = ""):
    if torch.cuda.is_available():
        print(f"\n[GPU MEMORY] {stage}")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"Max reserved:  {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
