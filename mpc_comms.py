import numpy as np

import utils.param as param

from network.c_socket import CSocket
from network.connect import open_channel, connect
from utils.type_ops import TypeOps
from utils.custom_types import zeros, add_mod
from utils.utils import bytes_to_arr

class MPCComms:
    def __init__(self: 'MPCComms', pid: int):
        self.pid = pid
        self.sockets: dict = dict()

        for p_1 in range(2):
            for p_2 in range(p_1 + 1, 3):
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

    def receive_bool(self: 'MPCComms', from_pid: int) -> bool:
        return bool(int(self.sockets[from_pid].receive(msg_len=1)))

    def send_bool(self: 'MPCComms', flag: bool, to_pid: int):
        self.sockets[to_pid].send(str(int(flag)).encode('utf-8'))

    def send_elem(self: 'MPCComms', elem: np.ndarray, to_pid: int) -> int:
        return self.sockets[to_pid].send(TypeOps.to_bytes(elem))
    
    def receive_elem(self: 'MPCComms', from_pid: int, msg_len: int) -> np.ndarray:
        return np.array(int(self.sockets[from_pid].receive(msg_len=msg_len)))

    def receive_vector(self: 'MPCComms', from_pid: int, msg_len: int, shape: tuple) -> np.ndarray:
        received_vec: np.ndarray = zeros(shape)

        for i, elem in enumerate(bytes_to_arr(self.sockets[from_pid].receive(msg_len=msg_len))):
            received_vec[i] = elem

        return received_vec
    
    def receive_matrix(self: 'MPCComms', from_pid: int, msg_len: int, shape: tuple) -> np.ndarray:
        matrix: np.ndarray = zeros(shape)
        row_values = self.sockets[from_pid].receive(msg_len=msg_len).split(b';')

        for i, row_value in enumerate(row_values):
            for j, elem in enumerate(bytes_to_arr(row_value)):
                matrix[i][j] = elem
        
        return matrix
    
    def receive_ndarray(self: 'MPCComms', from_pid: int, msg_len: int, ndim: int, shape: tuple) -> np.ndarray:
        if ndim == 2:
            return self.receive_matrix(from_pid, msg_len, shape)
        
        if ndim == 1:
            return self.receive_vector(from_pid, msg_len, shape)
        
        if ndim == 0:
            return self.receive_elem(from_pid, msg_len)
        
        raise ValueError(f'Invalid dimension expected: {ndim}. Should be either 0, 1 or 2.')
  
    def reveal_sym(self: 'MPCComms', elem: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        if self.pid == 0:
            return elem
        
        msg_len = TypeOps.get_bytes_len(elem)

        received_elem: np.ndarray = None
        if self.pid == 1:
            sent_data = self.send_elem(elem, 3 - self.pid)
            assert sent_data == msg_len, f'Sent {sent_data} bytes but expected {msg_len}'
            received_elem = self.receive_ndarray(3 - self.pid, msg_len=msg_len, ndim=elem.ndim, shape=elem.shape)
        else:
            received_elem = self.receive_ndarray(3 - self.pid, msg_len=msg_len, ndim=elem.ndim, shape=elem.shape)
            sent_data = self.send_elem(elem, 3 - self.pid)
            assert sent_data == msg_len, f'Sent {sent_data} bytes but expected {msg_len}'
            
        return add_mod(elem, received_elem, field)
    
    def clean_up(self: 'MPCComms'):
        for socket in self.sockets.values():
            socket.close()
