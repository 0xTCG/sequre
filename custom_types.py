import random

from functools import partial
from copy import deepcopy

import numpy as np

from param import BASE_P, BASE_LEN


zeros = partial(np.zeros, dtype=np.int64)
ones = partial(np.ones, dtype=np.int64)

def random_ndarray(base: int, shape: tuple) -> np.ndarray:
    return np.random.randint(base, size=shape)


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
    def bit(elem: int, k: int) -> int:
        return int((elem & (1 << k)) != 0)
    
    @staticmethod
    def get_mat_len(m: int, n: int) -> int:
        return m * n * BASE_LEN + m - 1 + (n - 1) * m
    
    @staticmethod
    def get_vec_len(n: int) -> int:
        return n * BASE_LEN + n - 1
    
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
            return BASE_LEN
        
        raise ValueError(f'Invalid operand. {arr} has inapt dimension.')

    @staticmethod
    def to_bytes(arr: np.ndarray) -> str:
        if arr.ndim > 2:
            raise ValueError(f'Ivalid dimension of arr to stringify: {arr.dim}')
        
        if arr.ndim == 0:
            base = b'0' * BASE_LEN
            str_val: str = str(arr)
            return (base + str_val.encode('utf-8'))[len(str_val):]
        
        separator: str = b';' if arr.ndim == 2 else b'.'

        return separator.join([TypeOps.to_bytes(v) for v in arr])



class Zp:
    def __init__(self: 'Zp', value: int, base: int = BASE_P):
        # if not isinstance(value, int):
        #     raise ValueError('Invalid value for Zp: ', value)
        self.base = base
        self.value = value % self.base
    
    def __int__(self: 'Zp') -> int:
        return self.value

    def __neg__(self: 'Zp') -> 'Zp':
        return Zp(-self.value, base=self.base)
    
    def __iadd__(self: 'Zp', other: 'Zp') -> 'Zp':
        other_val = int(other) % self.base
        self.value = (self.value + other_val) % self.base
        return self
    
    def __isub__(self: 'Zp', other: 'Zp') -> 'Zp':
        self += -other
        return self
    
    def __imul__(self: 'Zp', other: 'Zp') -> 'Zp':
        other_val = int(other) % self.base
        self.value = (self.value * other_val) % self.base
        return self
    
    def __add__(self: 'Zp', other: 'Zp') ->  'Zp':
        z = Zp(self.value, base=self.base)
        z += other
        return z
    
    def __sub__(self: 'Zp', other: 'Zp') ->  'Zp':
        return self + -other
    
    def __mul__(self: 'Zp', other: 'Zp') -> 'Zp':
        z = Zp(self.value, base=self.base)
        z *= other
        return z
    
    def __imod__(self: 'Zp', field: int) -> 'Zp':
        self.value %= field
        self.base = field
        return self

    def __mod__(self: 'Zp', field: int) -> 'Zp':
        z = Zp(self.value, base=self.base)
        z %= field
        return z
    
    def __eq__(self: 'Zp', other: 'Zp') -> bool:
        return self.value == other.value
    
    def __neq__(self: 'Zp', other: 'Zp') -> bool:
        return self.value != other.value
    
    def __lt__(self: 'Zp', other: 'Zp') -> bool:
        return self.value < other.value
    
    def __gt__(self: 'Zp', other: 'Zp') -> bool:
        return self.value > other.value
    
    def __le__(self: 'Zp', other: 'Zp') -> bool:
        return self.value <= other.value
    
    def __ge__(self: 'Zp', other: 'Zp') -> bool:
        return self.value >= other.value
    
    def __str__(self: 'Zp') -> str:
        return str(self.value)
    
    def __hash__(self: 'Zp') -> int:
        return hash(self.value)
    
    def __pow__(self: 'Zp', e: int) -> 'Zp':
        return Zp(pow(self.value, e, self.base), base=self.base)
    
    def __getitem__(self: 'Zp', idx: int) -> int:
        return self.value
    
    def to_bytes(self: 'Zp') -> str:
        base = b'0' * BASE_LEN
        return (base + str(self).encode('utf-8'))[len(str(self)):]
    
    def to_int(self: 'Zp') -> int:
        return int(self)
    
    def inv(self: 'Zp', p: int = None) -> 'Zp':
        p: int = self.base if p is None else p
        return Zp(pow(self.value, p - 2, p), base=p)
    
    def set_field(self: 'Zp', field: int) -> 'Zp':
        if field != self.base:
            self.base = field
            self.value %= self.base
    
        return self

    def change_field(self: 'Zp', field: int) -> 'Zp':
        self.base = field
        return self
    
    def to_field(self: 'Zp', field: int) -> 'Zp':
        return Zp(self.value, base=field)
    
    def get_bytes_len(self: 'Zp') -> int:
        return BASE_LEN
    
    @staticmethod
    def randzp(base: int = BASE_P) -> 'Zp':
        return Zp(random.randint(0, BASE_P - 1), base=base)


class Vector:
    def __init__(self: 'Vector', value: list = None):
        self.value = []
        if value is not None:
            self.value = value
        self.type_ = type(value[0]) if value else None
    
    def __neg__(self: 'Vector') -> 'Vector':
        return Vector([-e for e in self.value])
    
    def __iadd__(self: 'Vector', other: 'Vector') -> 'Vector':
        # TODO: Enable debug mode for logging message bellow.
        # if isinstance(other, float):
        #     print('BE CAUTIOUS: Floats used in vector addition!')
        if isinstance(other, Zp) or isinstance(other, int) or isinstance(other, float):
            other = Vector([other] * len(self))
        self.value = [self_e + other_e for self_e, other_e in zip(self, other)]
        return self
    
    def __isub__(self: 'Vector', other: 'Vector') -> 'Vector':
        self += -other
        return self
    
    def __imul__(self: 'Vector', other: 'Vector') -> 'Vector':
        # TODO: Enable debug mode for logging message bellow.
        # if isinstance(other, float):
        #     print('BE CAUTIOUS: Floats used in vector multiplication!')
        if isinstance(other, int):
            self.value = [self_e * other for self_e in self]
            return self

        self.value = [self_e * other_e for self_e, other_e in zip(self, other)]
        return self
    
    def __add__(self: 'Vector', other: 'Vector') ->  'Vector':
        z = Vector(self.value)
        z += other
        return z
    
    def __sub__(self: 'Vector', other: 'Vector') ->  'Vector':
        return self + -other
    
    def __mul__(self: 'Vector', other: 'Vector') -> 'Vector':
        z = Vector(self.value)
        z *= other
        return z
    
    def __imod__(self: 'Vector', field: int) -> 'Vector':
        for i, v in enumerate(self):
            self[i] = v % field
        return self

    def __mod__(self: 'Vector', field: int) -> 'Vector':
        z = Vector(self.value)
        z %= field
        return z
    
    def __eq__(self: 'Vector', other: 'Vector') -> bool:
        for e, o in zip(self.value, other.value):
            if e != o:
                return False
        return True
    
    def __neq__(self: 'Vector', other: 'Vector') -> bool:
        return not self == other
    
    def __lt__(self: 'Vector', other: 'Vector') -> bool:
        for self_e, other_e in zip(self.value, other.value):
            if not self_e < other_e:
                return False
        return True
    
    def __gt__(self: 'Vector', other: 'Vector') -> bool:
        for self_e, other_e in zip(self.value, other.value):
            if not self_e > other_e:
                return False
        return True
    
    def __str__(self: 'Vector') -> str:
        return str([str(e) for e in self.value])
    
    def __hash__(self: 'Vector') -> int:
        return hash(tuple(self.value))
    
    def __len__(self: 'Vector') -> int:
        return len(self.value)
    
    def __getitem__(self: 'Vector', idx: int) -> object:
        return self.value[idx]
    
    def __setitem__(self: 'Vector', idx: int, value: object):
        self.value[idx] = value
    
    def num_rows(self: 'Vector') -> int:
        return len(self.value)
    
    def num_cols(self: 'Vector') -> int:
        return len(self.value[0])

    def to_bytes(self: 'Vector') -> bytes:
        return b'.'.join([(Zp(e, BASE_P) if isinstance(e, int) else e).to_bytes() for e in self.value])
    
    def append(self: 'Vector', elem: object):
        if self.type_ is None:
            self.type_ = type(elem)
        self.value.append(elem)
    
    def mult(self: 'Vector', other: 'Vector') -> 'Matrix':
        assert self.num_cols() == other.num_rows()

        type_ = type(self[0][0])  # Temp hack
        new_mat = Matrix(self.num_rows(), other.num_cols(), t=type_)

        for i in range(self.num_rows()):
            for j in range(other.num_cols()):
                new_mat[i][j] = sum(
                    [self[i][k] * other[k][j] for k in range(self.num_cols())],
                    Zp(0, base=self[0][0].base) if isinstance(self[0][0], Zp) else type_(0))
        
        return new_mat
    
    def set_field(self: 'Vector', field: int = BASE_P) -> 'Vector':
        for i, v in enumerate(self.value):
            if isinstance(v, int):
                self.value[i] = Zp(v, base=field)
                self.type_ = Zp
            else:
                v.set_field(field)
        return self
    
    def change_field(self: 'Vector', field: int) -> 'Vector':
        if isinstance(self.value[0], int):
            return self
        
        for e in self.value:
            e.change_field(field)
        return self
    
    def to_field(self: 'Vector', field: int) -> 'Vector':
        is_int: bool = isinstance(self[0], int)
        return Vector([(Zp(e, field) if is_int else e).to_field(field) for e in self.value])
    
    def to_int(self: 'Vector') -> 'Vector':
        for i, v in enumerate(self.value):
            if isinstance(v, Zp):
                self.value[i] = int(v)
            else:
                v.to_int()
        
        return self
    
    def get_bytes_len(self: 'Vector') -> int:
        return TypeOps.get_vec_len(len(self))


# This inheritance will be easy to refactor to .seq via extend
class Matrix(Vector):
    def __init__(self: 'Matrix', m: int = 0, n: int = 0, randomise: bool = False, t: object = Zp, base: int = BASE_P):
        type_constructor = partial(Zp, base=base) if t == Zp else t
        self.value = [
            Vector([(Zp.randzp(base) if randomise else type_constructor(0)) for _ in range(n)])
            for _ in range(m)]
        self.type_ = Vector
    
    def __str__(self: 'Matrix') -> str:
        s: str = '\n'
        for v in self.value:
            s += f'{v}\n'
        
        return s
    
    def from_value(self: 'Matrix', value: Vector) -> 'Matrix':
        self.value = Vector(value.value)
        return self
    
    def get_dims(self: 'Matrix') -> tuple:
        return len(self), len(self[0])

    def set_dims(self: 'Matrix', m: int, n: int, base: int = BASE_P):
        new_mat: list = [Vector([Zp(0, base=base)] * n) for _ in range(m)]

        for i in range(min(len(self.value), m)):
            for j in range(min(len(self.value[i]), n)):
                new_mat[i][j] = self.value[i][j]
        
        self.value = new_mat
    
    def num_rows(self: 'Matrix') -> int:
        return len(self.value)
    
    def num_cols(self: 'Matrix') -> int:
        return len(self.value[0])
    
    def to_bytes(self: 'Matrix') -> bytes:
        return b';'.join([v.to_bytes() for v in self.value])
    
    def flatten(self: 'Matrix') -> Vector:
        return Vector([e for v in self.value for e in v])

    def reshape(self: 'Matrix', nrows: int, ncols: int):
        assert self.num_rows() * self.num_cols() == nrows * ncols

        b = Matrix(nrows, ncols)

        ai: int = 0
        aj: int = 0
        for i in range(nrows):
            for j in range(ncols):
                b[i][j] = self[ai][aj]
                aj += 1
                if aj == self.num_cols():
                    ai += 1
                    aj = 0
        
        self.value = b.value

    def get_bytes_len(self: 'Matrix') -> int:
        return TypeOps.get_mat_len(len(self), len(self[0]))
    
    def transpose(self: 'Matrix', inplace: bool) -> 'Matrix':
        m, n = self.get_dims()
        t_mat = Matrix(n, m)

        for i, row in enumerate(self.value):
            for j, cell in enumerate(row):
                t_mat[j][i] = deepcopy(cell)
        
        if inplace:
            self.value = t_mat.value
        
        return t_mat
