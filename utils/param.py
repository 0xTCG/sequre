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

# BASE_P: int = 1461501637330902918203684832716283019655932542929  # 160 bit
BASE_P: int = 9223372036854775783  # 63 bit
# BASE_P: int = 4294967291  # 32 bit
# NBIT_K: int = 60
NBIT_K: int = 40  # 64 bit
# NBIT_F: int = 45
NBIT_F: int = 20  # 64 bit
# NBIT_V: int = 64
NBIT_V: int = 3  # 64 bit

DIV_MAX_N = 100000

BASE_LEN: int = math.ceil(math.log10(BASE_P))
ITER_PER_EVAL: int = 5

NUM_INDS: int = 1000
NUM_SNPS = 1000
NUM_COVS: int = 10
NUM_DIM_TO_REMOVE = 5

NUM_OVERSAMPLE: int = 10
NUM_POWER_ITER: int = 5

SNP_POS_FILE: int = "test_data/pos.txt"
SIGMOID_APPROX_PATH: str = 'data/sigmoid_approx.txt'

IMISS_UB: float = 0.05
GMISS_UB: float = 0.1
HET_LB: float = 0.25
HET_UB: float = 0.30
MAF_LB: float = 0.4
MAF_UB: float = 0.6
HWE_UB: float = 28.3740
LD_DIST_THRES: int = 1000000
DIV_MAX_N: int = 100000

NUM_INDS: int = 1000
NUM_SNPS: int = 1000
NUM_COVS: int = 10

PITER_BATCH_SIZE: int = 100
PAR_THRES: int = 50
NUM_THREADS: int = 20

SKIP_QC: bool = False