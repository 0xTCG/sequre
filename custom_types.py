from param import BASE_P


class Zp:
    def __init__(self: 'Zp', value: int):
        self.value = value
    
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
