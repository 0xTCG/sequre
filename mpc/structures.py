from networking.client import Client
from custom_types.vector import Vector
from mpc.arithmetics import evaluate_polynomial

def table_lookup(client: Client, context_id: int, k: Vector, table_id: int, r: Vector, x_r: Vector, R: Vector) -> tuple:
    lagrange_coefs: list = client.get_lagrange_coefficients(table_id)
    degrees_list: list = [[i] for i in len(lagrange_coefs)]
    return evaluate_polynomial(client, context_id, k, lagrange_coefs, degrees_list, r, x_r, R)
