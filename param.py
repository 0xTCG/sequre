""" Module containing all params """
# Ports
# The party with smaller ID listens on the port
# and the other connects to it. Make sure the firewall
# setting of the listener allows these ports.
PORT_P0_P1: int = 8000
PORT_P0_P2: int = 8001
PORT_P1_P2: int = 8002
PORT_P1_P3: int = 8003
PORT_P2_P3: int = 8004
ALL_PORTS: list = [PORT_P0_P1, PORT_P0_P2, PORT_P1_P2, PORT_P1_P3, PORT_P2_P3]

AF_PREFIX: str = "./_socket"

BASE_P: int = 1461501637330902918203684832716283019655932542929
