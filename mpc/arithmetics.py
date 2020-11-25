import math

from typing import Any, List

from networking.client import Client
from utils.constants import CP2
from utils.lambdas import subtract

from custom_types.vector import Vector


def add(client: Client, context_id: int, x: Any, y: Any) -> tuple:
    return 0, x + y


def add_scalar(client: Client, context_id: int, x: Any, a: Any) -> tuple:
    a *= int(client.pid == CP2)
    return 0, x + a


def multiply_scalar(client: Client, context_id: int, x: Any, a: Any) -> tuple:
    return 0, x * a


def multiply(client: Client, context_id: int, x: Any, y: Any, a: Any, b: Any, c: Any) -> tuple:
    x_a = client.get_counter_client().get_shared(
        context_id=context_id,
        secrets=[0, 2],
        transform=subtract)
    y_b = client.get_counter_client().get_shared(
        context_id=context_id,
        secrets=[1, 3],
        transform=subtract)
    
    x_a += x - a
    y_b += y - b
    
    return x_a * y_b, x_a * b + y_b * a + c


def evaluate_polynomial(client: Client, context_id: int, x: Vector, coef: Vector, exp: Vector, r: Vector, x_r: Vector) -> tuple:
    raise NotImplementedError()
