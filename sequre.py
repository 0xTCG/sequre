from typing import Any

from mpc import preprocess, arithmetics
from networking.mainframe import Mainframe

from custom_types.vector import Vector


class Sequre:
    def __init__(self: 'Sequre'):
        self.mf = Mainframe()

    def __enter__(self: 'Sequre'):
        self.mf.__enter__()
        return self

    def __exit__(self: 'Sequre', exc_type, exc_val, exc_tb):
        self.mf.__exit__(exc_type, exc_val, exc_tb)

    def add(self: 'Sequre', x: Any, y: Any, share_inputs: bool) -> Any:
        return self.mf(arithmetics.add)(
            x, y, secret_args_mask=3 * share_inputs)

    def add_public(self: 'Sequre', x: Any, a: Any, share_inputs: bool) -> Any:
        return self.mf(arithmetics.add_scalar)(
            x, a, secret_args_mask=2 * share_inputs)
    
    def multiply_public(self: 'Sequre', x: Any, a: Any, share_inputs: bool) -> Any:
        return self.mf(arithmetics.multiply_scalar)(
            x, a, secret_args_mask=2 * share_inputs)
    
    def multiply(self: 'Sequre', x: Any, y: Any, share_inputs: bool) -> Any:
        return self.mf(arithmetics.multiply)(
            x, y, secret_args_mask=3 * share_inputs,
            preprocess=preprocess.get_multiplication_triple)

    def evaluate_polynomial(self: 'Sequre', x: Vector, coef: list, degrees_list: list, share_inputs: bool) -> Any:
        return self.mf(arithmetics.evaluate_polynomial)(
            x, coef, degrees_list, secret_args_mask=4 * share_inputs,
            preprocess=preprocess.beaver_partition)
