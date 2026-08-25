import os
import random
import numpy as np
import torch


def seed_everything(seed: int, rank: int):
    seed = seed + rank
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.seed_all(seed)

    return seed


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
