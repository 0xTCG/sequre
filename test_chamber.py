from sequre import Sequre

with Sequre() as sqr:
    assert sqr.add(1, 1) == 2
    assert sqr.add_public(2, 1) == 3
    assert sqr.multiply_public(5, 7) == 35
    assert sqr.multiply(6, 7) == 42
