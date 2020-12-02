import random
from typing import Any, Tuple, Callable

from networking.client import Client

from custom_types.vector import Vector


def generate_random_number() -> int:
    # TODO: Implement PRG
    # TODO: Do not forget to change the output type to Zp (and similar types)

    return random.randint(2 ** 0, 2 ** 32)


def decompose(arg: Any) -> Tuple[Any, Any]:
    r: int = generate_random_number()

    if isinstance(arg, Vector):
        return Vector([r] * len(arg)), arg - r
    
    return r, arg - r


def append_to_context(clients: list, context_id: int, shared_tuple: tuple):
    for client, shared in zip(clients[:len(shared_tuple)], shared_tuple):
        client.append_to_context(context_id, shared)


def prune_context(clients: list, context_id: int):
    for client in clients:
        client.prune_context(context_id)


def share_secret(clients: list, value: Any, context_id: int, private: bool = True):
    shared_pair: tuple = decompose(value) if private else (value, ) * 2
    append_to_context(clients, context_id, shared_pair)


def load_shared_from_path(clients: list, context_id: Any, data_path: str):
        for client in clients:
            client.load_shared_from_path(context_id, data_path)


def decompose_mask(mask: int, limit: int) -> bool:
    for _ in range(limit):
        yield bool(mask % 2)
        mask //= 2
