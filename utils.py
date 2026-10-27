import torch
from torch import Tensor





# Formats byte counts as human-readable storage sizes.
def format_bytes(num_bytes) -> str:
    if num_bytes is None: return "unknown size"
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024.0: return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"






# Sets reproducible random seeds and generates uniformly distributed random values.
def set_seed(seed: int = 42) -> torch.Generator:
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def rand_uniform(shape, low: float, high: float, generator: torch.Generator) -> Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator)
