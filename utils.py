import param


def get_address(port: int) -> str:
    return f"{param.AF_PREFIX}.{port}"
