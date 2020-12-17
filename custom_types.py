import random

from functools import partial

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
    def __init__(self: 'Zp', value: int, base: int):
        self.base = base
        self.value = value % self.base
    
    def __int__(self: 'Zp') -> int:
        return self.value

    def __neg__(self: 'Zp') -> 'Zp':
        return Zp(-self.value, base=self.base)
    
    def __iadd__(self: 'Zp', other: 'Zp') -> 'Zp':
        self.value = (self.value + other.value) % self.base
        return self
    
    def __isub__(self: 'Zp', other: 'Zp') -> 'Zp':
        self += -other
        return self
    
    def __imul__(self: 'Zp', other: 'Zp') -> 'Zp':
        self.value = (self.value * other.value) % self.base
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
    
    def __eq__(self: 'Zp', other: 'Zp') -> bool:
        return self.value == other.value
    
    def __neq__(self: 'Zp', other: 'Zp') -> bool:
        return self.value != other.value
    
    def __str__(self: 'Zp') -> str:
        return str(self.value)
    
    def __hash__(self: 'Zp') -> int:
        return hash(self.value)
    
    def __pow__(self: 'Zp', e: int) -> 'Zp':
        return Zp(pow(self.value, e, self.base), base=self.base)
    
    def to_bytes(self: 'Zp') -> bytes:
        return self.value.to_bytes((self.value.bit_length() + 7) // 8, 'big')
    
    def inv(self: 'Zp', p: int = None) -> 'Zp':
        p: int = self.base if p is None else p
        return Zp(pow(self.value, p - 2, p), base=p)
    
    def set_field(self: 'Zp', field: int):
        if field != self.base:
            self.base = field
            self.value %= self.base
    
    def to_field(self: 'Zp', field: int) -> 'Zp':
        return Zp(self.value, base=field)
    
    @staticmethod
    def randzp(base: int = BASE_P) -> 'Zp':
        return Zp(random.randint(0, 30), base=base)


class Vector:
    def __init__(self: 'Vector', value: list = None):
        self.value = []
        if value is not None:
            for v in value:
                type_v = type(v)
                v_ = v if isinstance(v, int) or isinstance(v, float) else Zp(v.value, v.base) if isinstance(v, Zp) else type_v(v.value)  # This hack will be avoided in .seq
                self.value.append(v_)
        self.type_ = type(value[0]) if value else None
    
    def __neg__(self: 'Vector') -> 'Vector':
        return Vector([-e for e in self.value])
    
    def __iadd__(self: 'Vector', other: 'Vector') -> 'Vector':
        if isinstance(other, Zp) or isinstance(other, int):
            other = Vector([other] * len(self))
        self.value = [self_e + other_e for self_e, other_e in zip(self.value, other.value)]
        return self
    
    def __isub__(self: 'Vector', other: 'Vector') -> 'Vector':
        self += -other
        return self
    
    def __imul__(self: 'Vector', other: 'Vector') -> 'Vector':
        if isinstance(other, Zp) or isinstance(other, int):
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
    
    def to_field(self: 'Vector', field: int) -> 'Vector':
        new_v = Vector(self)
        new_v.set_field(field)
        return new_v
    
    def to_int(self: 'Vector') -> 'Vector':
        for i, v in enumerate(self.value):
            if isinstance(v, Zp):
                self.value[i] = int(v)
            else:
                v.to_int()
        
        return self


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
        self.value = value
        return self
    
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
        if int(self[0][0]) == 119881670874624059276163306221842029527465:
            print('FOUND ONE!!!!!!!!!!!!')
        return b';'.join([v.to_bytes() for v in self.value])
    
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
