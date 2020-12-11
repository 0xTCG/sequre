import random
from param import BASE_P


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


class Zp:
    def __init__(self: 'Zp', value: int = 0, base: int = BASE_P):
        self.base = base
        self.value = value % self.base
    
    def __int__(self: 'Zp') -> int:
        return self.value

    def __neg__(self: 'Zp') -> 'Zp':
        return Zp(-self.value % BASE_P)
    
    def __iadd__(self: 'Zp', other: 'Zp') -> 'Zp':
        self.value = (self.value + other.value) % BASE_P
        return self
    
    def __isub__(self: 'Zp', other: 'Zp') -> 'Zp':
        self += -other
        return self
    
    def __imul__(self: 'Zp', other: 'Zp') -> 'Zp':
        self.value = (self.value * other.value) % BASE_P
        return self
    
    def __add__(self: 'Zp', other: 'Zp') ->  'Zp':
        z = Zp(self.value)
        z += other
        return z
    
    def __sub__(self: 'Zp', other: 'Zp') ->  'Zp':
        return self + -other
    
    def __mul__(self: 'Zp', other: 'Zp') -> 'Zp':
        z = Zp(self.value)
        z *= other
        return z
    
    def __eq__(self: 'Zp', other: 'Zp') -> bool:
        return self.value == other.value
    
    def __neq__(self: 'Zp', other: 'Zp') -> bool:
        return self.value != other.value
    
    def __str__(self: 'Zp') -> str:
        return str(self.value)
    
    def __hash__(self: 'Zp') -> int:
        return hash(self.value)
    
    def __pow__(self: 'Zp', e: int) -> 'Zp':
        return Zp(pow(self.value, e, self.base))
    
    def to_bytes(self: 'Zp') -> bytes:
        return self.value.to_bytes((self.value.bit_length() + 7) // 8, 'big')
    
    def inv(self: 'Zp') -> 'Zp':
        p: int = self.base
        return Zp(pow(self.value, p - 2, p))
    
    @staticmethod
    def randzp(base: int = BASE_P) -> 'Zp':
        return Zp(random.randint(0, base))


class Vector:
    def __init__(self: 'Vector', value: list = None):
        self.value = [] if value is None else value
        self.type_ = type(value[0]) if value else None
    
    def __neg__(self: 'Vector') -> 'Vector':
        return Vector([-e for e in self.value])
    
    def __iadd__(self: 'Vector', other: 'Vector') -> 'Vector':
        if isinstance(other, Zp):
            other = Vector([other] * len(self))
        self.value = [self_e + other_e for self_e, other_e in zip(self.value, other.value)]
        return self
    
    def __isub__(self: 'Vector', other: 'Vector') -> 'Vector':
        self += -other
        return self
    
    def __imul__(self: 'Vector', other: 'Vector') -> 'Vector':
        if isinstance(other, Zp):
            other = Vector([other] * len(self))
        self.value = [self_e * other_e for self_e, other_e in zip(self.value, other.value)]
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
    
    def __eq__(self: 'Vector', other: 'Vector') -> bool:
        for e, o in zip(self.value, other.value):
            if e != o:
                return False
        return True
    
    def __neq__(self: 'Vector', other: 'Vector') -> bool:
        return not self == other
    
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
        return b'.'.join([bytes(str(e), encoding='utf8') for e in self.value])
    
    def append(self: 'Vector', elem: object):
        if self.type_ is None:
            self.type_ = type(elem)
        self.value.append(elem)


# This inheritance will be easy to refactor to .seq via extend
class Matrix(Vector):
    def __init__(self: 'Matrix', m: int = 0, n: int = 0, randomise: bool = False, t: object = Zp):
        self.value = [
            Vector([(Zp.randzp() if randomise else t(0)) for _ in range(n)])
            for _ in range(m)]
        self.type_ = Vector
    
    def __str__(self: 'Matrix') -> str:
        s: str = '\n'
        for v in self.value:
            s += f'{v}\n'
        
        return s
    
    def from_value(self: 'Matrix', value: Vector) -> 'Matrix':
        self.value = value
        return self
    
    def set_dims(self: 'Matrix', m: int, n: int):
        new_mat: list = [Vector([Zp(0)] * n) for _ in range(m)]

        for i, row in enumerate(self.value):
            for j, cell in enumerate(row):
                new_mat[i][j] = cell
        
        self.value = new_mat
    
    def num_rows(self: 'Matrix') -> int:
        return len(self.value)
    
    def num_cols(self: 'Matrix') -> int:
        return len(self.value[0])
    
    def to_bytes(self: 'Matrix') -> bytes:
        return b';'.join([v.to_bytes() for v in self.value])
