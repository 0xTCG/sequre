import socket
from typing import Any

from utils import get_address


class CSocket:
    def __init__(self: 'CSocket', pid: int = 0):
        self._pid = pid
        self._port = 0
        self.m_sock = None
        self.bytes_sent: int = 0
        self.bytes_received: int = 0
    
    def socket(self: 'CSocket'):
        self.close()
        self.m_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    
    def connect(self: 'CSocket', address: str) -> bool:
        try:
            self.m_sock.connect(address)
        except socket.error:
            return False
        
        return True
    
    def close(self: 'CSocket'):
        if self.m_sock is not None and not self.m_sock._closed:
            self.m_sock.shutdown(0)
            self.m_sock.close()
    
    def bind(self: 'CSocket', port: int):
        self._port = port
        self.m_sock.bind(get_address(port))
    
    def listen(self: 'CSocket'):
        self.m_sock.listen()
    
    def accept(self: 'CSocket'):
        self.m_sock, _ = self.m_sock.accept()
    
    def send(self: 'CSocket', data: bytes, nLen: int, nFlags: int = 0) -> int:
        self.bytes_sent += nLen
        return self.m_sock.send(data, nFlags)

    def receive(self: 'CSocket', nLen: int, nFlags: int = 0) -> bytes:
        self.bytes_received += nLen

        p: bytes = bytes(0)
        n: int = nLen
        ret: int = 0
        
        while (n > 0):
            ret = self.m_sock.recv(n, nFlags)
            p += ret
            n -= len(ret)

        return nLen
    
    def reset_stats(self: 'CSocket'):
        self.bytes_received = 0
        self.bytes_sent = 0
