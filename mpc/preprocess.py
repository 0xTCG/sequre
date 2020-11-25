import math
from typing import List

from mpc.secret import generate_random_number, decompose, share_secret
from networking.client import Client
from custom_types.vector import Vector
from utils.lambdas import subtract


def get_multiplication_triple(client: Client, context_id: int) -> List[tuple]:
    triple: List[int] = [generate_random_number()] * 2
    triple.append(math.prod(triple))

    for e in triple:
        share_secret(clients=client.get_other_clients(), value=e, context_id=context_id)


def beaver_partition(client: Client, context_id: int) -> List[tuple]:
    x: Vector = client.get_param(0)
    peaky_blinders: Vector = Vector([generate_random_number()] * len(x))
    computing_clients: list = client.get_other_clients()
    share_secret(clients=computing_clients, context_id=context_id, value=peaky_blinders)

    x_r = sum([
        client.get_counter_client().get_shared(
            context_id=context_id,
            secrets=[0, -1],
            transform=subtract)
        for client in computing_clients])
    
    share_secret(clients=computing_clients, context_id=context_id, value=x_r, mask=0)

