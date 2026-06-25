import os
import random
import numpy as np
import torch


def fix_seed(seed: int = 42, deterministic: bool = True, warn_only: bool = True):
    """
    PyTorch 실험용 seed 고정 함수.
    rank별 seed가 필요하면 호출할 때 직접 seed + rank처럼 넘기면 됨.

    예:
        fix_seed(1234)          # 모든 rank 같은 seed
        fix_seed(1234 + rank)   # rank마다 다른 seed
    """

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