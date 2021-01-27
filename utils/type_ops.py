import numpy as np

import utils.param as param


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
