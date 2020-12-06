import time

import param
from c_socket import CSocket
from utils import get_address


RETRY_CONNECT: int = 100


def connect(socket: CSocket, port: int) -> bool:
    for _ in range(RETRY_CONNECT):
        socket.socket()

        if (socket.connect(get_address(port))):
            return True
        
        print("Connection failed, retrying..")
        time.sleep(1)

    return False


def listen(socket: CSocket, port: int):
    socket.socket()
    socket.bind(port)
    socket.listen()
    socket.accept()


def open_channel(socket: CSocket, port: int):
    print(f'Listening init {port}')
    listen(socket, port)
    print(f'Listening at {port}')
