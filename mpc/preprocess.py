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
    computing_clients: list = client.get_other_clients()
    vector_len: int = len(computing_clients[0].get_param(context_id, 0))
    peaky_blinders: Vector = Vector([generate_random_number()] * vector_len)
    client.append_to_context(context_id=context_id, shared=peaky_blinders)
    share_secret(clients=computing_clients, context_id=context_id, value=peaky_blinders)

    x_r = sum([
        client.get_shared(
            context_id=context_id,
            secrets=[0, -1],
            transform=subtract)
        for client in computing_clients], Vector([0] * vector_len))
    
    share_secret(clients=computing_clients, context_id=context_id, value=x_r, private=False)
