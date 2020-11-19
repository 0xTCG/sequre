import random
from typing import Any, Tuple

def generate_random_number() -> int:
    # TODO: Implement PRG

    return random.randint(2 ** 0, 2 ** 32)


def share_secret(arg: Any) -> Tuple[Any, Any]:
    r: int = generate_random_number()
    # TODO: Implement async sharing with clients

    return r, arg - r


def reconstruct_secret(args: list) -> Any:
    # TODO: Implement async sharing with clients

    return sum(args)
