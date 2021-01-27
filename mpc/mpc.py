import sys
import time
import random
import math

from functools import partial, reduce
from copy import deepcopy

import numpy as np

import utils.param as param

from mpc.prg import PRG
from mpc.comms import Comms
from mpc.arithmetic import Arithmetic
from mpc.polynomial import Polynomial
from mpc.boolean import Boolean
from mpc.fp import FP
from mpc.lin_alg import LinAlg
from network.c_socket import CSocket
from network.connect import connect, open_channel
from utils.custom_types import zeros, ones, add_mod, mul_mod, matmul_mod
from utils.type_ops import TypeOps
from utils.utils import bytes_to_arr, rand_int, random_ndarray


class MPCEnv:
    def __init__(self: 'MPCEnv', pid: int):
        self.pid: int = None
        self.primes: dict = {0: param.BASE_P, 1: 31, 2: 17}  # Temp hardcoded. Needs to be calcualted on init.

        self.pid = pid
        self.comms = Comms(self.pid)
        self.prg = PRG(self.pid)
        self.arithmetic = Arithmetic(
            pid=self.pid,
            prg=self.prg,
            comms=self.comms)
        self.polynomial = Polynomial(
            pid=self.pid,
            primes=self.primes,
            prg=self.prg,
            comms=self.comms,
            arithmetic=self.arithmetic)
        self.boolean = Boolean(
            pid=self.pid,
            prg=self.prg,
            comms=self.comms,
            arithmetic=self.arithmetic,
            polynomial=self.polynomial)
        self.fp = FP(
            pid=self.pid,
            primes=self.primes,
            prg=self.prg,
            comms=self.comms,
            arithmetic=self.arithmetic,
            polynomial=self.polynomial,
            boolean=self.boolean)
        self.lin_alg = LinAlg(
            pid=self.pid,
            primes=self.primes,
            arithmetic=self.arithmetic,
            boolean=self.boolean,
            fp=self.fp)
