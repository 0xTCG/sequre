import param


def get_address(port: int) -> str:
    return f"{param.AF_PREFIX}.{port}"

def get_cache_path(pid: int, name: str) -> str:
    return f'{param.CACHE_FILE_PREFIX}_P{pid}_{name}.bin'

def get_output_path(pid: int, name: str) -> str:
    return f'{param.OUTPUT_FILE_PREFIX}_P{pid}_{name}.txt'

def get_temp_path(pid: int, name: str) -> str:
    return f'temp/temp_P{pid}_{name}.txt'
