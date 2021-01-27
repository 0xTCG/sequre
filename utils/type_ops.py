import numpy as np

import utils.param as param

from utils.utils import random_ndarray


class TypeOps:
    @staticmethod
    def set_bit(x: int, p: int) -> int:
        if p < 0:
            raise ValueError(f'Invalid bit index: {p}')

        return x | (1 << p)
    
    @staticmethod
    def trunc_elem(elem: int, k: int) -> int:
        return elem & ((1 << k) - 1)
    
    @staticmethod
    def left_shift(elem: int, k: int) -> int:
        if k < 0:
            return elem >> k
        
        return elem << k
    
    @staticmethod
    def right_shift(elem: int, k: int) -> int:
        if k < 0:
            return elem << k
        
        return elem >> k
    
    @staticmethod
    def bit(elem: int, k: int) -> bool:
        return (elem & (1 << k)) != 0
    
    @staticmethod
    def get_mat_len(m: int, n: int) -> int:
        return m * n * param.BASE_LEN + m - 1 + (n - 1) * m
    
    @staticmethod
    def get_vec_len(n: int) -> int:
        return n * param.BASE_LEN + n - 1
    
    @staticmethod
    def switch_pair(t: tuple) -> tuple:
        return t[1], t[0]
    
    @staticmethod
    def mod_inv(value: int, field: int) -> int:
        return pow(int(value), field - 2, field)
    
    @staticmethod
    def get_bytes_len(arr: np.ndarray) -> int:
        if arr.ndim == 2:
            return TypeOps.get_mat_len(*arr.shape)
        
        if arr.ndim == 1:
            return TypeOps.get_vec_len(*arr.shape)
        
        if arr.ndim == 0:
            return param.BASE_LEN
        
        raise ValueError(f'Invalid operand. {arr} has inapt dimension.')

    @staticmethod
    def to_bytes(arr: np.ndarray) -> str:
        if arr.ndim > 2:
            raise ValueError(f'Ivalid dimension of arr to stringify: {arr.dim}')
        
        if arr.ndim == 0:
            base = b'0' * param.BASE_LEN
            str_val: str = str(arr)  # '{:f}'.format(arr)
            return (base + str_val.encode('utf-8'))[len(str_val):]
        
        separator: str = b';' if arr.ndim == 2 else b'.'

        return separator.join([TypeOps.to_bytes(v) for v in arr])
    
    @staticmethod
    def double_to_fp(x: float, k: int, f: int, field: int = param.BASE_P) -> int:
        sn: int = 1
        if x < 0:
            x = -x
            sn = -sn

        az: int = int(x)

        az_shift: int = TypeOps.left_shift(az, f)
        az_trunc: int = TypeOps.trunc_elem(az_shift, k - 1)

        xf: float = x - az  # remainder
        for fbit in range(f - 1, -1, -1):
            xf *= 2
            if (xf >= 1):
                xf -= int(xf)
                az_trunc = TypeOps.set_bit(az_trunc, fbit)
        
        return (az_trunc * sn) % field
    
    @staticmethod
    def fp_to_double(a: np.ndarray, k: int, f: int, field: int = param.BASE_P) -> np.ndarray:
        twokm1: int = TypeOps.left_shift(1, k - 1)

        sn: np.ndarray = np.where(a > twokm1, -1, 1)
        x: np.ndarray = np.where(a > twokm1, field - a, a)
        x_trunc: np.ndarray = TypeOps.trunc_elem(x, k - 1)
        x_int: np.ndarray = TypeOps.right_shift(x_trunc, f)

        # TODO: consider better ways of doing this?
        x_frac = np.zeros(a.shape)
        for bi in range(f):
            x_frac = np.where(TypeOps.bit(x_trunc, bi) > 0, x_frac + 1, x_frac)
            x_frac /= 2

        return sn * (x_int + x_frac)
    
    @staticmethod
    def rand_bits(shape: tuple, num_bits: int, field: int) -> np.ndarray:
        upper_limit: int = field - 1
        if num_bits > 63:
            print(f'Warning: Number of bits too big for numpy: {num_bits}')
        else:
            upper_limit = (1 << num_bits) - 1
    
        return random_ndarray(upper_limit, shape=shape) % field
