from typing import Any

from mpc import arithmetics
from networking.mainframe import Mainframe


class Sequre:
    def __init__(self: 'Sequre'):
        self.mf = Mainframe()

    def __enter__(self: 'Sequre'):
        self.mf.__enter__()
        return self

    def __exit__(self: 'Sequre', exc_type, exc_val, exc_tb):
        self.mf.__exit__(exc_type, exc_val, exc_tb)

    def add(self: 'Sequre', x: Any, y: Any) -> Any:
        return self.mf(arithmetics.add)(
            x, y, secret_args_mask='11', preprocess=None)

    def add_public(self: 'Sequre', x: Any, a: Any) -> Any:
        return self.mf(arithmetics.add_scalar)(
            x, a, secret_args_mask='10', preprocess=None)
    
    def multiply_public(self: 'Sequre', x: Any, a: Any) -> Any:
        return self.mf(arithmetics.multiply_scalar)(
            x, a, secret_args_mask='10', preprocess=None)
    
    def multiply(self: 'Sequre', x: Any, y: Any) -> Any:
        return self.mf(arithmetics.multiply)(
            x, y, secret_args_mask='11', preprocess=arithmetics.get_multiplication_triple)
