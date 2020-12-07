import time
from custom_types import Zp
from mpc import MPCEnv


def assert_values(result, expected):
    assert result == expected, f'Result: {result}. Expected: {expected}'


def test_all(mpc: MPCEnv = None, pid: int = None):
    # Zp
    assert_values(Zp(1) + Zp(1), Zp(2))
    test_val = Zp(1)
    test_val += Zp(1)
    assert_values(test_val, Zp(2))
    assert_values(Zp(2) - Zp(1), Zp(1))
    test_val = Zp(2)
    test_val -= Zp(1)
    assert_values(test_val, Zp(1))
    assert_values(Zp(2) * Zp(3), Zp(6))
    test_val = Zp(2)
    test_val *= Zp(3)
    assert_values(test_val, Zp(6))

    if mpc is not None and pid is not None:
        if pid != 0:
            revealed_value: Zp = mpc.reveal_sym(Zp(10) if pid == 1 else Zp(7))
            assert_values(revealed_value, Zp(17))

    print(f'All tests passed at {pid}!')


if __name__ == "__main__":
    test_all()
