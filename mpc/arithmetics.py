import math

from typing import Any, List

from networking.client import Client
from mpc.secret import generate_random_number, share_secret
from utils.constants import CP2
from utils.lambdas import subtract


def add(x: Any, y: Any, client: Client, context_id: int) -> tuple:
    return 0, x + y


def add_scalar(x: Any, a: Any, client: Client, context_id: int) -> tuple:
    a *= int(client.pid == CP2)
    return 0, x + a


def multiply_scalar(x: Any, a: Any, client: Client, context_id: int) -> tuple:
    return 0, x * a


def multiply(x: Any, y: Any, a: Any, b: Any, c: Any, client: Client, context_id: int) -> tuple:
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
