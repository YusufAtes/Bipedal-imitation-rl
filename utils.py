# seed_utils.py
import os
import random
import numpy as np
import torch
from scipy.interpolate import CubicSpline
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

def create_noisy_plane(gamma,omega, row_size = 32, col_size = 1024,simulation_res = 0.05):    #5 cm resolutions defined for the simulation
    # Create the plane
    full_plane = np.zeros(col_size)
    end_point = col_size * simulation_res 
    mid_point = end_point / 2

    plane_coarse = np.arange(mid_point + gamma, end_point, gamma)
    plane_fine = np.arange(mid_point + gamma , end_point, simulation_res)
    plane = np.zeros(len(plane_coarse))
    prev_height = 0.0
    for i in range(len(plane_coarse)):
        #truncated normal noise
        noise = np.random.normal()  # Adjust the standard
        noise = np.clip(noise, -omega, omega)  # Clip to a range
        height = prev_height + noise
        plane[i] = height
        prev_height = height
    cs = CubicSpline(plane_coarse, plane, bc_type='natural')
    full_plane[-len(plane_fine):] = cs(plane_fine)
    full_plane_data = np.repeat(full_plane, row_size)  # Repeat the plane data for each row

    return full_plane_data
