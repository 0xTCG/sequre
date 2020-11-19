import math
from typing import List

from mpc.secret import generate_random_number, share_secret


def get_multiplication_triple() -> List[tuple]:
    triple: List[tuple] = [generate_random_number()] * 2
    triple.append(math.prod(triple))

    return [share_secret(e) for e in triple]
