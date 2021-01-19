import numpy as np

import param
import random


def get_address(port: int) -> str:
    return f"{param.AF_PREFIX}.{port}"


def get_cache_path(pid: int, name: str) -> str:
    return f'{param.CACHE_FILE_PREFIX}_P{pid}_{name}.bin'


def get_output_path(pid: int, name: str) -> str:
    return f'{param.OUTPUT_FILE_PREFIX}_P{pid}_{name}.txt'


def get_temp_path(pid: int, name: str) -> str:
    return f'temp/temp_P{pid}_{name}.txt'


def rand_int(lower_limit: int, upper_limit: int) -> int:
    return random.randint(lower_limit, upper_limit)


def bytes_to_arr(bytes_str: str) -> np.ndarray:
    for elem in bytes_str.split(b'.'):
        yield int(elem)
