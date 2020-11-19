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

    def add(self: 'Sequre', *args) -> Any:
        return self.mf(arithmetics.add)(*args, args_mask='11')
