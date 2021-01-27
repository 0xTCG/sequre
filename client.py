""" GWAS client module """

import os

from functools import partial

import utils.param as param
from mpc.mpc_env import MPCEnv
from utils.utils import get_address

from protocol.test import test_protocol


protocols: dict = {
    'gwas': None,
    'logireg': None,
    'test': test_protocol
}


def client(pid: int, protocol_func: callable):
    # Initialize MPC environment
    mpc = MPCEnv(pid)
    print(f"Initialized MPC for {pid}")

    success: bool = protocol_func(mpc, pid)

    # This is here just to keep P0 online until the end for data transfer
    # In practice, P0 would send data in advance before each phase and go offline
    if (pid == 0): mpc.comms.receive_bool(2)
    elif (pid == 2): mpc.comms.send_bool(True, 0)

    mpc.comms.clean_up()

    if (success): print(f"Protocol successfully completed for {pid}")
    else: raise ValueError(f"Protocol abnormally terminated for {pid}")


def invoke(protocol_func: callable):
    for port in param.ALL_PORTS:
        address: str = get_address(port)
        if os.path.exists(address):
            os.unlink(address)

    client_func: callable = partial(client, protocol_func=protocol_func)
    pid: int = os.fork()

    if (pid == 0): client_func(0)
    else:
        pid: int = os.fork()
        if (pid == 0): client_func(1)
        else: client_func(2)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    
    invoke(protocols[args[0]])
