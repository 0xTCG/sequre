""" This is temporary module. Will be replaced with a rather different seq lang paradigms."""

class Vector:
    def __init__(self: 'Vector', l: list):
        self._v = l
    
    def __len__(self: 'Vector') -> int:
        return len(self._v)
    
    def __str__(self: 'Vector') -> str:
        return str(self._v)
    
    def __sub__(self: 'Vector', other: 'Vector') -> 'Vector':
        if not isinstance(other, Vector):
            other = Vector([other] * len(self._v))
        return Vector([e_1 - e_2 for e_1, e_2 in zip(self._v, other._v)])

    def __add__(self: 'Vector', other: 'Vector') -> 'Vector':
        if not isinstance(other, Vector):
            other = Vector([other] * len(self._v))
        return Vector([e_1 + e_2 for e_1, e_2 in zip(self._v, other._v)])
