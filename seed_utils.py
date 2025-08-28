# seed_utils.py
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Fixes RNG seeds across random, numpy and torch.

    Args
    ----
    seed : int
        The seed value to use everywhere.
    deterministic : bool
        If True, sets CuDNN / CUDA to deterministic mode.
        This slows training slightly but removes all nondeterminism.
    """
    # -------- Python ---------------------------------------------------------
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)        # python ≥3.3 hash seed

    # -------- NumPy ----------------------------------------------------------
    np.random.seed(seed)

    # -------- PyTorch --------------------------------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)                    # current GPU
    torch.cuda.manual_seed_all(seed)                # all GPUs (multi-GPU)

    if deterministic:
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False      # turn off autotune
    else:
        # Faster but non-deterministic (use if perf >> reproducibility)
        torch.backends.cudnn.benchmark = True

    # --------  Extra safety: torch >= 1.8  -----------------------------------
    # Ensures some ops that still have nondeterministic implementations
    # raise an error if they’re hit.
    torch.use_deterministic_algorithms(deterministic)

    # -------- Informative print (optional) -----------------------------------
    print(f"Global seed set to {seed} (deterministic={deterministic})")

# Example usage ---------------------------------------------------------------
if __name__ == "__main__":
    set_global_seed(1234)
    x = torch.randn(3, 3)  # will be identical every run
    print(x)
