import time
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.utils.data


class Timer:
    timings = defaultdict(list)

    @staticmethod
    def checkpoint():
        for func_name, elapsed_ms_list in Timer.timings.items():
            print(f"{func_name}: {np.mean(elapsed_ms_list):.3f} ms")


def set_seed(seed: int):
    import random
    import numpy as np
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
