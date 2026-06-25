import os
import random
import numpy as np
import torch


def fix_seed(seed: int = 42, deterministic: bool = True, warn_only: bool = True):


    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # torch.use_deterministic_algorithms(True) 쓸 때
        # CUDA matmul 계열에서 필요한 경우가 있음.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    else:
        torch.use_deterministic_algorithms(False)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return generator, seed_worker