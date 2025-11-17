import os
import random
from typing import Optional


def enable_tf32() -> None:
    """Enable TF32 for faster matmuls on Ampere+ GPUs (A100, A6000, RTX 3090+).

    TF32 provides ~2x speedup for FP32 operations with minimal precision loss.
    Only available on Ampere (SM 8.0+) and newer architectures.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Enable TF32 for matmul operations
            torch.backends.cuda.matmul.allow_tf32 = True
            # Enable TF32 for cudnn operations
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def set_seed(seed: int = 42, deterministic: bool = True, env_var: Optional[str] = None) -> int:
    """Set RNG seeds for Python, NumPy, and PyTorch (if available).

    Args:
        seed: Random seed to use
        deterministic: If True, enable deterministic algorithms (slower but reproducible).
                      If False, enable cudnn.benchmark for faster training.
        env_var: Environment variable name to override seed

    Returns:
        The final seed used
    """
    if env_var and (v := os.getenv(env_var)):
        try:
            seed = int(v)
        except ValueError:
            pass

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        np = None  # type: ignore

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        torch = None  # type: ignore

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Deterministic mode: reproducible but slower
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
        else:
            # Performance mode: faster but not reproducible
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            except Exception:
                pass

    return seed

