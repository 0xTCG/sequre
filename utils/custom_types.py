import random

from functools import partial, reduce
from copy import deepcopy

import numpy as np

from utils.param import BASE_P, BASE_LEN


def random_ndarray(base: int, shape: tuple) -> np.ndarray:
    return np.random.randint(base, size=shape)


# Numpy overrides
zeros = partial(np.zeros, dtype=np.int64)
ones = partial(np.ones, dtype=np.int64)


# Temp modular arithmetic wrappers (add_mod, mul_mod and matmul_mod)
def add_mod(x: np.ndarray, y: np.ndarray, field: int) -> np.ndarray:
    return np.mod(x - (-y + field), field)


def mul_mod(x: np.ndarray, y: np.ndarray, field: int) -> np.ndarray: 
    res: np.ndarray = zeros(shape=x.shape)
    broadcast_y: np.ndarray = zeros(shape=x.shape)
    broadcast_y[:] = np.broadcast_to(y, x.shape)
    
    x = np.mod(x, field)
    while np.any(broadcast_y > 0): 
        indices = np.where((broadcast_y & 1) == 1)
        res[indices] = add_mod(res[indices], x[indices], field)
  
        x = add_mod(x, x, field)
        broadcast_y >>= 1
  
    return np.mod(res, field)


def matmul_mod(x: np.ndarray, y: np.ndarray, field: int) -> np.ndarray:
    assert x.shape[1] == y.shape[0]

    new_mat = zeros((x.shape[0], y.shape[1]))

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            new_mat[i][j] = reduce(
                partial(add_mod, field=field), mul_mod(x[i], y.T[j], field), np.array(0, dtype=np.int64))
    
    return new_mat
