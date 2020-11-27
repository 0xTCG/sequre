from sequre import Sequre

from custom_types.vector import Vector

with Sequre() as sqr:
    assert sqr.add(1, 1) == 2
    assert sqr.add_public(2, 1) == 3
    assert sqr.multiply_public(5, 7) == 35
    assert sqr.multiply(6, 7) == 42
    assert sqr.evaluate_polynomial(Vector([1, 2]), [3, 2, 1], [[0, 1], [2, 3], [4, 5]]) == 54

    print('All tests passed!')
