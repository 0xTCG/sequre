import sys
import time
import random

import param
from c_socket import CSocket
from connect import connect, open_channel
from custom_types import Zp


class MPCEnv:
    def __init__(self: 'MPCEnv'):
        self.sockets: dict = dict()
        self.prg_states: dict = dict()
        self.pid: int = None
    
    def initialize(self: 'MPCEnv', pid: int, pairs: list) -> bool:
        self.pid = pid

        if (not self.setup_channels(pairs)):
            raise ValueError("MPCEnv::Initialize: failed to initialize communication channels")
            
        if (not self.setup_prgs()):
            raise ValueError("MPCEnv::Initialize: failed to initialize PRGs")

        return True
    
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
                raise ValueError(f"Failed to connect with P{pother}")

        return True

    def setup_prgs(self: 'MPCEnv') -> bool:
        for other_pid in set(range(3)) - {self.pid}:
            random.seed(hash((min(self.pid, other_pid), max(self.pid, other_pid))))
            self.prg_states[other_pid] = random.getstate()
        
        return True

    def receive_bool(self: 'MPCEnv', from_pid: int) -> bool:
        return self.sockets[from_pid].receive()

    def send_bool(self: 'MPCEnv', flag: bool, to_pid: int):
        self.sockets[to_pid].send(bytes(flag), sys.getsizeof(flag))

    def send_elem(self: 'MPCEnv', elem: Zp, to_pid: int):
        self.sockets[to_pid].send(elem.to_bytes(), sys.getsizeof(elem.value))
    
    def receive_elem(self: 'MPCEnv', from_pid: int) -> Zp:
        return Zp(int.from_bytes(self.sockets[from_pid].receive(), 'big'))

    def clean_up(self: 'MPCEnv'):
        for socket in self.sockets.values():
            socket.close()
  
    def reveal_sym(self: 'MPCEnv', elem: Zp) -> Zp:
        if (self.pid == 0):
            return None

        received_elem: Zp = None
        if (self.pid == 1):
            self.send_elem(elem, 3 - self.pid)
            received_elem = self.receive_elem(3 - self.pid)
        else:
            received_elem = self.receive_elem(3 - self.pid)
            self.send_elem(elem, 3 - self.pid)

        return elem + received_elem
    