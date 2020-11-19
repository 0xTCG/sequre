import random
from typing import Any, Tuple, Callable

from networking.client import Client


def generate_random_number() -> int:
    # TODO: Implement PRG
    # TODO: Do not forget to change the output type to Zp (and similar types)

    return random.randint(2 ** 0, 2 ** 32)


def share_secret(arg: Any) -> Tuple[Any, Any]:
    r: int = generate_random_number()

    return r, arg - r
