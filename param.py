""" Module containing all params """
import math

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

OUTPUT_FILE_PREFIX: str = "out/test"
CACHE_FILE_PREFIX: str = "cache/test"

BASE_P: int = 1461501637330902918203684832716283019655932542929
NBIT_K: int = 60
NBIT_F: int = 45
NBIT_V: int = 64

DIV_MAX_N = 100000

BASE_LEN: int = math.ceil(math.log10(BASE_P))
ITER_PER_EVAL: int = 5

NUM_INDS: int = 1000
NUM_SNPS = 1000
NUM_COVS: int = 10
NUM_DIM_TO_REMOVE = 5