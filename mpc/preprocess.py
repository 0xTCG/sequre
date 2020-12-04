import math
from typing import List

from mpc.secret import generate_random_number, decompose, share_secret
from networking.client import Client
from custom_types.vector import Vector
from utils.lambdas import subtract
from utils.numerics import term_masks


def get_multiplication_triple(client: Client, context_id: int) -> List[tuple]:
    triple: List[int] = [generate_random_number()] * 2
    triple.append(math.prod(triple))

    for e in triple:
        share_secret(clients=client.get_other_clients(), value=e, context_id=context_id)


def beaver_partition(client: Client, context_id: int) -> Vector:
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
    
    return peaky_blinders


def polinomial_partition(client: Client, context_id: int, degrees_list: list = None):
    peaky_blinders: Vector = beaver_partition(client=client, context_id=context_id)
    
    computing_clients: list = client.get_other_clients()
    all_term_masks = set()
    degrees_list = computing_clients[0].get_param(context_id, 2) if degrees_list is None else degrees_list

    for degrees in degrees_list:
        all_term_masks |= set(term_masks(degrees))
    all_term_masks = sorted(list(all_term_masks))
    
    term_values = Vector([
        math.prod([peaky_blinder ** exponent for peaky_blinder, exponent in zip(peaky_blinders, term_mask)])
        for term_mask in all_term_masks])
    share_secret(clients=computing_clients, context_id=context_id, value=term_values)


def lookup_partition(client: Client, context_id: int):
    computing_clients: list = client.get_other_clients()
    table_id: int = computing_clients[0].get_param(context_id, 1)
    lagrange_coefs: Vector = Vector(computing_clients[0].get_lagrange_coefficients(table_id))
    degrees: list = [[i] for i in range(len(lagrange_coefs))]
    polinomial_partition(client, context_id, degrees)
