import math

from functools import partial

import numpy as np

import utils.param as param

from mpc.prg import PRG
from mpc.comms import Comms
from mpc.arithmetic import Arithmetic
from mpc.polynomial import Polynomial
from utils.custom_types import zeros, add_mod, mul_mod
from utils.utils import random_ndarray
from utils.type_ops import TypeOps


class MPCBoolean:
    def __init__(self: 'MPCBoolean', pid: int, prg: PRG, comms: Comms, arithmetic: Arithmetic, polynomial: Polynomial):
        self.pid = pid
        self.prg = prg
        self.comms = comms
        self.arithmetic = arithmetic
        self.polynomial = polynomial

        self.or_lagrange_cache: dict = dict()
    
    def fan_in_or(self: 'MPCBoolean', a: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        n: int = a.shape[0]
        d: int = a.shape[1]
        a_sum = [0] * n

        # TODO: Vectorize a_sum calculation below
        if self.pid > 0:
            for i in range(n):
                a_sum[i] = int(self.pid == 1)
                for j in range(d):
                    a_sum[i] += int(a[i][j])
        
        a_sum = np.mod(a_sum, field)

        coeff = zeros((1, d + 1))

        key: tuple = (d + 1, field)
        if key not in self.or_lagrange_cache:
            y = np.array([int(i != 0) for i in range(d + 1)], dtype=np.int64)
            coeff_param = self.polynomial.lagrange_interp_simple(y, field) # OR function
            self.or_lagrange_cache[key] = coeff_param

        coeff[0][:] = self.or_lagrange_cache[key]
        bmat = self.polynomial.evaluate_poly(a_sum, coeff, field)
        
        return bmat[0]
    
    def prefix_or(self: 'MPCBoolean', a: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        n: int = a.shape[0]

        # Find next largest squared integer
        L: int = int(math.ceil(math.sqrt(a.shape[1])))
        L2: int = L * L

        # Zero-pad to L2 bits
        a_padded: np.ndarray = zeros((n, L2))
        
        if self.pid > 0:
            for i in range(n):
                for j in range(L2):
                    if j >= L2 - a.shape[1]:
                        a_padded[i][j] = a[i][j - L2 + a.shape[1]]

        a_padded = a_padded.reshape((n * L, L))

        x: np.ndarray = self.fan_in_or(a_padded, field)

        xpre: np.ndarray = zeros((n * L, L))
        
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    xpi: int = L * i + j
                    for k in range(L):
                        xpre[xpi][k] = x[L * i + k] * int(k <= j)
        
        y: np.ndarray = self.fan_in_or(xpre, field)

        # TODO: Make it parallel by using ndarray
        f: list = [zeros((1, L)) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    if j == 0:
                        f[i][0][j] = x[L * i]
                    else:
                        f[i][0][j] = y[L * i + j] - y[L * i + j - 1]
                f[i] %= field

        # TODO: Make it parallel by using ndarray
        tmp: list = [zeros((L, L)) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    tmp[i][j][:] = a_padded[L * i + j]
                tmp[i] %= field

        c = self.arithmetic.mult_mat_parallel(f, tmp, field)  # c is a concatenation of n 1-by-L matrices

        cpre: np.ndarray = zeros((n * L, L))
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    cpi: int = L * i + j
                    for k in range(L):
                        cpre[cpi][k] = c[i][0][k] * int(k <= j)

        bdot_vec: np.ndarray = self.fan_in_or(cpre, field)
        
        bdot = [zeros((1, L)) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    bdot[i][0][j] = bdot_vec[L * i + j]

        for i in range(n):
            f[i] = f[i].reshape((L, 1))

        s = self.arithmetic.mult_mat_parallel(f, bdot, field)

        b = zeros(a.shape)
        if self.pid > 0:
            for i in range(n):
                for j in range(a.shape[1]):
                    j_pad: int = L2 - a.shape[1] + j

                    il: int = j_pad // L
                    jl: int = j_pad - il * L

                    b[i][j] = np.mod(
                        add_mod(s[i][il][jl], y[L * i + il], field) - f[i][il][0], field)
        
        return b
    
    def less_than_bits_public(self: 'MPCBoolean', a: np.ndarray, b_pub: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        return self.less_than_bits_aux(a, b_pub, 2, field)

    def less_than_bits(self: 'MPCEnv', a: np.ndarray, b: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        return self.less_than_bits_aux(a, b, 0, field)
    
    def less_than_bits_aux(self: 'MPCEnv', a: np.ndarray, b: np.ndarray, public_flag: int, field: int = param.BASE_P) -> np.ndarray:
        assert a.shape == b.shape

        n: int = a.shape[0]
        L: int = a.shape[1]
        mul_func = partial(mul_mod, field=field)
        add_func = partial(add_mod, field=field)

        # Calculate XOR
        x = zeros((n, L))

        if public_flag == 0:
            x: np.ndarray = self.arithmetic.multiply(a, b, True, field)
            if self.pid > 0:
                x = np.mod(add_func(a, b) - add_func(x, x), field)
        elif self.pid > 0:
            x = mul_func(a, b)
            x = np.mod(add_func(a, b) - add_func(x, x), field)
            if self.pid == 2:
                x = np.mod(x - (a if public_flag == 1 else b), field)
        
        f: np.ndarray = self.prefix_or(x, field)

        if self.pid > 0:
            for i in range(n):
                for j in range(L - 1, 0, -1):
                    f[i][j] = (f[i][j] - f[i][j - 1]) % field
        
        if public_flag == 2:
            c = zeros(n)
            if self.pid > 0:
                fb: np.ndarray = mul_func(f, b)
                # TODO: Implemenput np.sum over axis=1 of c here.
                for i in range(n):
                    for e in fb[i]:
                        c[i] = add_func(c[i], e)
            
            return c
        
        # TODO: optimize
        f_arr = [zeros((1, L)) for _ in range(n)]
        b_arr = [zeros((L, 1)) for _ in range(n)]

        if self.pid > 0:
            for i in range(n):
                f_arr[i][0][:] = f[i]
                for j in range(L):
                    b_arr[i][j][0] = b[i][j]

        c_arr: list = self.arithmetic.mult_mat_parallel(f_arr, b_arr, field)
        
        return np.array([c_arr[i][0][0] if self.pid > 0 else 0 for i in range(n)], dtype=np.int64)
    
    def is_positive(self: 'MPCBoolean', a: np.ndarray, nbits: int, field: int) -> np.ndarray:
        n: int = len(a)

        r: np.ndarray = None
        r_bits: np.ndarray = None
        if self.pid == 0:
            r: np.ndarray = random_ndarray(base=param.BASE_P, shape=n)
            r_bits: np.ndarray = TypeOps.num_to_bits(r, nbits)

            self.prg.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(base=param.BASE_P, shape=n)
            r_bits_mask: np.ndarray = random_ndarray(base=field, shape=(n, nbits))
            self.prg.restore_seed(1)

            r -= r_mask
            r_bits -= r_bits_mask
            r %= param.BASE_P
            r_bits %= field

            self.comms.send_elem(r, 2)
            self.comms.send_elem(r_bits, 2)
        elif self.pid == 2:
            r: np.ndarray = self.comms.receive_vector(0, msg_len=TypeOps.get_vec_len(n), shape=n)
            r_bits: np.ndarray = self.comms.receive_matrix(0, msg_len=TypeOps.get_mat_len(n, nbits), shape=(n, nbits))
        else:
            self.prg.switch_seed(0)
            r: np.ndarray = random_ndarray(base=param.BASE_P, shape=n)
            r_bits: np.ndarray = random_ndarray(base=field, shape=(n, nbits))
            self.prg.restore_seed(0)

        c: np.ndarray = zeros(1)
        if self.pid != 0:
            c = add_mod(add_mod(a, a, param.BASE_P), r, param.BASE_P)

        c = self.comms.reveal_sym(c, field=param.BASE_P)

        c_bits = zeros((n, nbits))
        if self.pid != 0:
            c_bits = TypeOps.num_to_bits(c, nbits)

        # Incorrect result if r = 0, which happens with probaility 1 / BASE_P
        no_overflow: np.ndarray = self.less_than_bits_public(r_bits, c_bits, field=field)

        c_xor_r = zeros(n)
        if self.pid > 0:
            # Warning: Overflow might occur below in numpy.
            for i in range(n):
                c_xor_r[i] = r_bits[i][nbits - 1] - 2 * c_bits[i][nbits - 1] * r_bits[i][nbits - 1]
                if self.pid == 1:
                    c_xor_r[i] += c_bits[i][nbits - 1]
            c_xor_r %= field
        
        lsb: np.ndarray = self.arithmetic.multiply(c_xor_r, no_overflow, True, field)
        if self.pid > 0:
            lsb = add_mod(lsb, lsb, field)
            lsb -= add_mod(no_overflow, c_xor_r, field)
            if self.pid == 1:
                lsb = add_mod(lsb, 1, field)
        lsb %= field

        # 0, 1 -> 1, 2
        if self.pid == 1:
            lsb = add_mod(lsb, 1, field)
        
        b_mat: np.ndarray = self.polynomial.table_lookup(lsb, 0)

        return b_mat[0]
