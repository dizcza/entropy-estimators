import random
import time
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.utils.data


class Timer:
    timings = defaultdict(list)

    @staticmethod
    def checkpoint(fpath=None):
        def print_all(file=None):
            for func_name, elapsed_ms_list in Timer.timings.items():
                print(f"{func_name}: {np.mean(elapsed_ms_list):.3f} ms", file=file)
        print_all()
        if fpath is not None:
            with open(fpath, 'w') as f:
                print_all(file=f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def timer_profile(func):
    """
    For debug purposes only.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start
        elapsed *= 1e3
        Timer.timings[func.__name__].append(elapsed)
        return res

    return wrapped
