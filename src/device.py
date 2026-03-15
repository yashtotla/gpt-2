import torch


def get_device():
    # priority: cuda > mps > cpu (highest to lowest compute capability)
    # - cuda: nvidia GPU
    # - mps: apple silicon GPU backend (fairly capable depending on macbook)
    # - cpu: always available, slowest - usable for short runs without torch.compile
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
