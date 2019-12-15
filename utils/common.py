import time
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.utils.data


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
    func_duration = defaultdict(list)

    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start
        elapsed *= 1e3
        func_duration[func.__name__].append(elapsed)
        print(f"{func.__name__} {elapsed: .3f} (mean: {np.mean(func_duration[func.__name__]): .3f}) ms")
        return res

    return wrapped
