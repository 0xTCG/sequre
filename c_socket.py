
import sys
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
        except socket.error as msg:
            print(f"Could not connect to {address}. {msg}")
            return False
        
        return True
    
    def close(self: 'CSocket'):
        if self.m_sock is not None and not self.m_sock._closed:
            self.m_sock.shutdown(socket.SHUT_RDWR)
            self.m_sock.close()
    
    def bind(self: 'CSocket', port: int):
        self._port = port
        self.m_sock.bind(get_address(port))
    
    def listen(self: 'CSocket'):
        self.m_sock.listen()
    
    def accept(self: 'CSocket'):
        self.m_sock, _ = self.m_sock.accept()
    
    def send(self: 'CSocket', data: str):
        totalsent = 0
        while totalsent < len(data):
            sent = self.m_sock.send(data[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent
        return None

    def receive(self: 'CSocket', msg_len: int) -> bytes:
        chunks = []
        bytes_recd = 0
        while bytes_recd < msg_len:
            chunk = self.m_sock.recv(min(msg_len - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)
    
    def reset_stats(self: 'CSocket'):
        self.bytes_received = 0
        self.bytes_sent = 0
