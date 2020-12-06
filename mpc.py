import sys

import param
from c_socket import CSocket
from connect import connect, open_channel


class MPCEnv:
    def __init__(self: 'MPCEnv'):
        self.sockets: dict = dict()
        self.pid: int = None
    
    def initialize(self: 'MPCEnv', pid: int, pairs: list) -> bool:
        self.pid = pid

        if (not self.setup_channels(pairs)):
            raise ValueError("MPCEnv::Initialize: failed to initialize communication channels")
    
    def setup_channels(self: 'MPCEnv', pairs: list) -> bool:
        for pair in pairs:
            p_1, p_2 = pair

            if (p_1 != self.pid and p_2 != self.pid):
                continue

            port: int = 8000
            if (p_1 == 0 and p_2 == 1):
                port = param.PORT_P0_P1
            elif (p_1 == 0 and p_2 == 2):
                port = param.PORT_P0_P2
            elif (p_1 == 1 and p_2 == 2):
                port = param.PORT_P1_P2
            elif (p_1 == 1 and p_2 == 3):
                port = param.PORT_P1_P3
            elif (p_1 == 2 and p_2 == 3):
                port = param.PORT_P2_P3

            pother: int = p_1 + p_2 - self.pid
            self.sockets[pother] = CSocket(self.pid)

            if (p_1 == self.pid):
                open_channel(self.sockets[pother], port)
            elif (not connect(self.sockets[pother], port)):
                    raise ValueError("Failed to connect with P{pother}")

        return True

    def receive_bool(self: 'MPCEnv', from_pid: int) -> bool:
        return self.sockets[from_pid].receive(sys.getsizeof(bool))

    def send_bool(self: 'MPCEnv', flag: bool, to_pid: int):
        self.sockets[to_pid].send(flag, sys.getsizeof(bool))

    def clean_up(self: 'MPCEnv'):
        for socket in self.sockets:
            socket.close()
