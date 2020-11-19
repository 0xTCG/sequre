from typing import Any


def subtract(args: list) -> Any:
    if len(args) != 2:
        raise ValueError('Invalid params for subtraction!')
    
    return args[0] - args[1]
