""" GWAS client module """

import os

import param
from mpc import MPCEnv
from utils import get_address
from custom_types import Zp

from tests import test_all


def gwas_protocol(mpc: MPCEnv, pid: int) -> bool:
    test_all(mpc, pid)
    return True


def client(pid: int):
    pairs: list = [(i, j) for i in range(2) for j in range(i + 1, 3)]

    # Initialize MPC environment
    mpc = MPCEnv()
    if (not mpc.initialize(pid, pairs)):
        raise ValueError("MPC environment initialization failed")
    
    print(f"Initialized MPC for {pid}")
    success: bool = gwas_protocol(mpc, pid)

    # This is here just to keep P0 online until the end for data transfer
    # In practice, P0 would send data in advance before each phase and go offline
    # if (pid == 0): mpc.receive_bool(2)
    # elif (pid == 2): mpc.send_bool(True, 0)

    mpc.clean_up()

    if (success): print("Protocol successfully completed")
    else: raise ValueError("Protocol abnormally terminated")


def main():
    for port in param.ALL_PORTS:
        address: str = get_address(port)
        if os.path.exists(address):
            os.unlink(address)

    pid: int = os.fork()

    if (pid == 0): client(0)
    else:
        pid: int = os.fork()
        if (pid == 0): client(1)
        else: client(2)

if __name__ == "__main__":
    main()
