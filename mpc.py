import sys
import time
import random

from functools import partial

import param
from c_socket import CSocket
from connect import connect, open_channel
from custom_types import Zp, Vector


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
        random.seed()
        self.prg_states[self.pid] = random.getstate() 
        
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
    
    def receive_vector(self: 'MPCEnv', from_pid: int) -> Vector:
        values = self.sockets[from_pid].receive().decode("utf-8").split('.')
        return Vector([Zp(int(v)) for v in values])

    def clean_up(self: 'MPCEnv'):
        for socket in self.sockets.values():
            socket.close()
  
    def reveal_sym(self: 'MPCEnv', elem: Zp) -> Zp:
        if (self.pid == 0):
            return None

        receive_func = None
        if isinstance(elem, Zp):
            receive_func = self.receive_elem
        if isinstance(elem, Vector):
            receive_func = self.receive_vector

        received_elem: Zp = None
        if (self.pid == 1):
            self.send_elem(elem, 3 - self.pid)
            received_elem = receive_func(3 - self.pid)
        else:
            received_elem = receive_func(3 - self.pid)
            self.send_elem(elem, 3 - self.pid)

        return elem + received_elem
    
    def switch_seed(self: 'MPCEnv', pid: int):
        self.prg_states[self.pid] = random.getstate()
        random.setstate(self.prg_states[pid])
    
    def restore_seed(self: 'MPCEnv', pid: int):
        self.prg_states[pid] = random.getstate()
        random.setstate(self.prg_states[self.pid])

    def rand_elem(self: 'MPCEnv') -> Zp:
        return Zp(random.randint(0, param.BASE_P))
    
    def rand_vector(self: 'MPCEnv', size: int) -> Vector:
        return Vector([self.rand_elem() for _ in range(size)])

    def beaver_partition(self: 'MPCEnv', x: object) -> tuple:
        type_ = type(x)
        rand_func = None
        if isinstance(x, Zp):
            rand_func = self.rand_elem
        elif isinstance(x, Vector):
            rand_func = partial(self.rand_vector, size=len(x))
        x_r = type_()
        r = type_()
        if (self.pid == 0):
            self.switch_seed(1)
            r_1: Zp = rand_func()
            self.restore_seed(1)

            self.switch_seed(2)
            r_2: Zp = rand_func()
            self.restore_seed(2)

            r: Zp = r_1 + r_2
        else:
            self.switch_seed(0)
            r: Zp = rand_func()
            self.restore_seed(0)

            x_r = self.reveal_sym(x - r)
        
        return x_r, r
    
    def powers(self: 'MPCEnv', x: Vector, pow: int) -> Matrix:
        assert pow >= 1

        n: int = len(x)
        b = Matrix()
        
        if (pow == 1):
            b.set_dims(2, n)
        if (self.pid > 0):
            if (self.pid == 1):
                b[0] += Zp(1)
            b[1] = x
        else: # pow > 1
            x_r, r = self.beaver_partition(x)

            if (self.pid == 0):
                r_pow = Matrix(pow - 1, n)
                r_pow[0] = self.mul_elem(r, r)
                
                for p in range(1, r_pow.num_rows()):
                    r_pow[p] = self.mul_elem(r_pow[p - 1], r)

                self.switch_seed(1)
                r_ = self.rand_mat(pow - 1, n)
                self.restore_seed(1)

                r_pow -= r_

                self.send_mat(r_pow, 2)

                b.set_dims(pow + 1, n)
            else:
                r_pow = Matrix()
                if (self.pid == 1):
                    self.switch_seed(0)
                    r_pow = self.rand_mat(pow - 1, n)
                    self.restore_seed(0)
                else:  # pid == 2
                    r_pow = self.receive_mat(0, pow - 1, n)

                x_r_pow = Matrix(pow - 1, n)
                x_r_pow[0] = self.mul_elem(x_r, x_r)
                for p in range(1, x_r_pow.num_rows()):
                    x_r_pow[p] = self.mul_elem(x_r_pow[p - 1], x_r)

                pascal_matrix = self.get_pascal_matrix(pow)

                b.set_dims(pow + 1, n)

                if (self.pid == 1):
                    b[0] += Zp(1)
                b[1] = x

                for p in range(2, pow + 1):
                    if (self.pid == 1): b[p] = x_r_pow[p - 2]

                    if (p == 2):
                        b[p] += pascal_matrix[p][1] * self.mul_elem(x_r, r)
                    else:
                        b[p] += pascal_matrix[p][1] * self.mul_elem(x_r_pow[p - 3], r)

                        for j in range(2, p - 1):
                            b[p] += pascal_matrix[p][j] * self.mul_elem(x_r_pow[p - 2 - j], r_pow[j - 2])

                        b[p] += pascal_matrix[p][p - 1] * self.mul_elem(x_r, r_pow[p - 3])

                    b[p] += r_pow[p - 2]
        
        return b
    