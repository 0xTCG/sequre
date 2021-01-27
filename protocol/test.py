from mpc.mpc import MPCEnv
from tests.tests import test_all, benchmark


def test_protocol(mpc: MPCEnv, pid: int) -> bool:
    test_all(mpc, pid)
    # benchmark(mpc, pid, m=5, n=5)
    return True
