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
from network.c_socket import CSocket
from network.connect import connect, open_channel
from utils.custom_types import zeros, ones, add_mod, mul_mod, matmul_mod
from utils.type_ops import TypeOps
from utils.utils import bytes_to_arr, rand_int, random_ndarray


class MPCEnv:
    def __init__(self: 'MPCEnv', pid: int):
        self.pid: int = None
        self.primes: dict = {0: param.BASE_P, 1: 31, 2: 17}  # Temp hardcoded. Needs to be calcualted on init.
        self.primes_bits: dict = {k: math.ceil(math.log2(v)) for k, v in self.primes.items()}
        self.primes_bytes: dict = {k: (v + 7) // 8 for k, v in self.primes_bits.items()}
        self.invpow_cache: dict = dict()
        self.or_lagrange_cache: dict = dict()

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

    def print_fp(self: 'MPCEnv', mat: np.ndarray, fid: int) -> np.ndarray:
        if self.pid == 0:
            return None
        revealed_mat: np.ndarray = self.comms.reveal_sym(mat, field=self.primes[fid])
        mat_float: np.ndarray = TypeOps.fp_to_double(revealed_mat, param.NBIT_K, param.NBIT_F, field=self.primes[fid])

        if self.pid == 2:
            print(f'{self.pid}: {mat_float}')
        
        return mat_float
    
    def trunc(self: 'MPCEnv', a: np.ndarray, k: int = param.NBIT_K + param.NBIT_F, m: int = param.NBIT_F, fid: int = 0):
        msg_len: int = TypeOps.get_bytes_len(a)
        
        if self.pid == 0:
            r: np.ndarray = TypeOps.rand_bits(a.shape, k + param.NBIT_V, field=self.primes[fid])
            r_low: np.ndarray = zeros(a.shape)
            
            r_low = np.mod(r & ((1 << m) - 1), self.primes[fid])

            self.prg.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(base=self.primes[fid], shape=a.shape)
            r_low_mask: np.ndarray = random_ndarray(base=self.primes[fid], shape=a.shape)
            self.prg.restore_seed(1)

            r = np.mod(r - r_mask, self.primes[fid])
            r_low = np.mod(r_low - r_low_mask, self.primes[fid])

            self.comms.send_elem(r, 2)
            self.comms.send_elem(r_low, 2)
        elif self.pid == 2:
            r: np.ndarray = self.comms.receive_ndarray(
                from_pid=0, msg_len=msg_len, ndim=a.ndim, shape=a.shape)
            r_low: np.ndarray = self.comms.receive_ndarray(
                from_pid=0, msg_len=msg_len, ndim=a.ndim, shape=a.shape)
        else:
            self.prg.switch_seed(0)
            r: np.ndarray = random_ndarray(base=self.primes[fid], shape=a.shape)
            r_low: np.ndarray = random_ndarray(base=self.primes[fid], shape=a.shape)
            self.prg.restore_seed(0)

        c = add_mod(a, r, self.primes[fid]) if self.pid > 0 else zeros(a.shape)
        c = self.comms.reveal_sym(c, field=self.primes[fid])

        c_low: np.ndarray = zeros(a.shape)
        if self.pid > 0:
            c_low = np.mod(c & ((1 << m) - 1), self.primes[fid])

        if self.pid > 0:
            a = add_mod(a, r_low, self.primes[fid])
            if self.pid == 1:
                a = np.mod(a - c_low, self.primes[fid])

            if m not in self.invpow_cache:
                twoinv: int = TypeOps.mod_inv(2, self.primes[fid])
                twoinvm = pow(twoinv, m, self.primes[fid])
                self.invpow_cache[m] = twoinvm
                
            a = mul_mod(a, self.invpow_cache[m], self.primes[fid])
        
        return a
    
    def fan_in_or(self: 'MPCEnv', a: np.ndarray, fid: int) -> np.ndarray:
        n: int = a.shape[0]
        d: int = a.shape[1]
        a_sum = [0] * n

        # TODO: Vectorize a_sum calculation below
        if self.pid > 0:
            for i in range(n):
                a_sum[i] = int(self.pid == 1)
                for j in range(d):
                    a_sum[i] += int(a[i][j])
        
        a_sum = np.mod(a_sum ,self.primes[fid])

        coeff = zeros((1, d + 1))

        key: tuple = (d + 1, fid)
        if key not in self.or_lagrange_cache:
            y = np.array([int(i != 0) for i in range(d + 1)], dtype=np.int64)
            coeff_param = self.polynomial.lagrange_interp_simple(y, self.primes[fid]) # OR function
            self.or_lagrange_cache[key] = coeff_param

        coeff[0][:] = self.or_lagrange_cache[key]
        bmat = self.polynomial.evaluate_poly(a_sum, coeff, self.primes[fid])
        
        return bmat[0]
    
    def prefix_or(self: 'MPCEnv', a: np.ndarray, fid: int) -> np.ndarray:
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

        x: np.ndarray = self.fan_in_or(a_padded, fid)

        xpre: np.ndarray = zeros((n * L, L))
        
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    xpi: int = L * i + j
                    for k in range(L):
                        xpre[xpi][k] = x[L * i + k] * int(k <= j)
        
        y: np.ndarray = self.fan_in_or(xpre, fid)

        # TODO: Make it parallel by using ndarray
        f: list = [zeros((1, L)) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    if j == 0:
                        f[i][0][j] = x[L * i]
                    else:
                        f[i][0][j] = y[L * i + j] - y[L * i + j - 1]
                f[i] %= self.primes[fid]

        # TODO: Make it parallel by using ndarray
        tmp: list = [zeros((L, L)) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    tmp[i][j][:] = a_padded[L * i + j]
                tmp[i] %= self.primes[fid]

        c = self.arithmetic.mult_mat_parallel(f, tmp, self.primes[fid])  # c is a concatenation of n 1-by-L matrices

        cpre: np.ndarray = zeros((n * L, L))
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    cpi: int = L * i + j
                    for k in range(L):
                        cpre[cpi][k] = c[i][0][k] * int(k <= j)

        bdot_vec: np.ndarray = self.fan_in_or(cpre, fid)
        
        bdot = [zeros((1, L)) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    bdot[i][0][j] = bdot_vec[L * i + j]

        for i in range(n):
            f[i] = f[i].reshape((L, 1))

        s = self.arithmetic.mult_mat_parallel(f, bdot, self.primes[fid])

        b = zeros(a.shape)
        if self.pid > 0:
            for i in range(n):
                for j in range(a.shape[1]):
                    j_pad: int = L2 - a.shape[1] + j

                    il: int = j_pad // L
                    jl: int = j_pad - il * L

                    b[i][j] = np.mod(
                        add_mod(s[i][il][jl], y[L * i + il], self.primes[fid]) - f[i][il][0], self.primes[fid])
        
        return b
    
    def int_to_fp(self: 'MPCEnv', a: int, k: int, f: int, fid: int) -> int:
        sn = 1 if a >= 0 else -1

        az_shift: int = TypeOps.left_shift(a, f)
        az_trunc: int = TypeOps.trunc_elem(az_shift, k - 1)

        return (az_trunc * sn) % self.primes[fid]

    def fp_div(self: 'MPCEnv', a: np.ndarray, b: np.ndarray, fid: int) -> np.ndarray:
        assert len(a) == len(b)

        n: int = len(a)
        field: int = self.primes[fid]
        add_func = partial(add_mod, field=field)
        mul_func = partial(mul_mod, field=field)
        
        if n > param.DIV_MAX_N:
            nbatch: int = math.ceil(n / param.DIV_MAX_N)
            c = zeros(n)
            
            for i in range(nbatch):
                start: int = param.DIV_MAX_N * i
                end: int = start + param.DIV_MAX_N
                
                if end > n:
                    end = n
                batch_size: int = end - start

                a_copy = zeros(batch_size)
                b_copy = zeros(batch_size)
                for j in range(batch_size):
                    a_copy[j] = a[start + j]
                    b_copy[j] = b[start + j]

                c_copy: np.ndarray = self.fp_div(a_copy, b_copy, fid=fid)
                for j in range(batch_size):
                    c[start + j] = c_copy[j]
            return c

        niter: int = 2 * math.ceil(math.log2(param.NBIT_K / 3.5)) + 1

        # Initial approximation: 1 / x_scaled ~= 5.9430 - 10 * x_scaled + 5 * x_scaled^2
        s, _ = self.normalizer_even_exp(b)

        b_scaled: np.ndarray = self.arithmetic.multiply(b, s, True, field=self.primes[fid])
        b_scaled = self.trunc(b_scaled, param.NBIT_K, param.NBIT_K - param.NBIT_F, fid=fid)

        b_scaled_sq: np.ndarray = self.arithmetic.multiply(b_scaled, b_scaled, True, field=self.primes[fid])
        b_scaled_sq = self.trunc(b_scaled_sq, fid=fid)

        scaled_est = zeros(n)
        if self.pid != 0:
            scaled_est = np.mod(
                mul_func(b_scaled_sq, 5) - mul_func(b_scaled, 10), field)
            if self.pid == 1:
                coeff: int = TypeOps.double_to_fp(5.9430, param.NBIT_K, param.NBIT_F, field=self.primes[fid])
                scaled_est = add_func(scaled_est, coeff)

        w: np.ndarray = self.arithmetic.multiply(scaled_est, s, True, field=self.primes[fid])
        # scaled_est has bit length <= NBIT_F + 2, and s has bit length <= NBIT_K
        # so the bit length of w is at most NBIT_K + NBIT_F + 2
        w = self.trunc(w, param.NBIT_K + param.NBIT_F + 2, param.NBIT_K - param.NBIT_F, fid=fid)

        x: np.ndarray = self.arithmetic.multiply(w, b, True, field=self.primes[fid])
        x = self.trunc(x, fid=fid)

        one: int = self.int_to_fp(1, param.NBIT_K, param.NBIT_F, fid=fid)

        x = np.mod(-x, field)
        if self.pid == 1:
            x = add_func(x, one)
        
        y: np.ndarray = self.arithmetic.multiply(a, w, True, field=self.primes[fid])
        y = self.trunc(y, fid=fid)

        for _ in range(niter):
            xr, xm = self.arithmetic.beaver_partition(x, field=self.primes[fid])
            yr, ym = self.arithmetic.beaver_partition(y, field=self.primes[fid])

            xpr = xr.copy()
            if self.pid > 0:
                xpr = add_func(xpr, one)

            y = self.arithmetic.beaver_mult(yr, ym, xpr, xm, True)
            x = self.arithmetic.beaver_mult(xr, xm, xr, xm, True)

            x = self.arithmetic.beaver_reconstruct(x, field=self.primes[fid])
            y = self.arithmetic.beaver_reconstruct(y, field=self.primes[fid])

            x = self.trunc(x, fid=fid)
            y = self.trunc(y, fid=fid)

        if self.pid == 1:
            x = add_func(x, one)
            
        c: np.ndarray = self.arithmetic.multiply(y, x, True, field=self.primes[fid])
        return self.trunc(c, fid=fid)

    def less_than_bits_public(self: 'MPCEnv', a: np.ndarray, b_pub: np.ndarray, fid: int) -> np.ndarray:
        return self.less_than_bits_aux(a, b_pub, 2, fid)

    def num_to_bits(self: 'MPCEnv', a: np.ndarray, bitlen: int) -> np.ndarray:
        b = zeros((len(a), bitlen))
    
        for i in range(len(a)):
            for j in range(bitlen):
                b[i][j] = int(TypeOps.bit(int(a[i]), bitlen - 1 - j))
    
        return b
    
    def __share_random_bits(self: 'MPCEnv', k: int, n: int, fid: int) -> tuple:
        if self.pid == 0:
            r: np.ndarray = TypeOps.rand_bits(n, k + param.NBIT_V, field=self.primes[fid])
            rbits: np.ndarray = self.num_to_bits(r, k)

            self.prg.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(self.primes[fid], n)
            rbits_mask: np.ndarray = random_ndarray(self.primes[fid], (n, k))
            self.prg.restore_seed(1)

            r -= r_mask
            r %= self.primes[fid]

            rbits -= rbits_mask
            rbits %= self.primes[fid]

            self.comms.send_elem(r, 2)
            self.comms.send_elem(rbits, 2)
        elif self.pid == 2:
            r: np.ndarray = self.comms.receive_vector(0, msg_len=TypeOps.get_vec_len(n), shape=(n, ))
            rbits: np.ndarray = self.comms.receive_matrix(0, msg_len=TypeOps.get_mat_len(n, k), shape=(n, k))
        else:
            self.prg.switch_seed(0)
            r: np.ndarray = random_ndarray(self.primes[fid], n)
            rbits: np.ndarray = random_ndarray(self.primes[fid], (n, k))
            self.prg.restore_seed(0)
        
        return r, rbits

    def normalizer_even_exp(self: 'MPCEnv', a: np.ndarray) -> tuple:
        n: int = len(a)
        fid: int = 1
        field: int = self.primes[fid]
        add_func = partial(add_mod, field=field)
        mul_func = partial(mul_mod, field=field)

        r, rbits = self.__share_random_bits(param.NBIT_K, n, fid)

        # Warning: a + r might overflow in numpy.
        e = zeros(n) if self.pid == 0 else a + r
        e = self.comms.reveal_sym(e, field=self.primes[0])

        ebits: np.ndarray = zeros(
            (n, param.NBIT_K)) if self.pid == 0 else self.num_to_bits(e, param.NBIT_K)
        
        c: np.ndarray = self.less_than_bits_public(rbits, ebits, fid)

        if self.pid > 0:
            c = np.mod(-c, field)
            if self.pid == 1:
                c = add_func(c, 1)
        
        ep: np.ndarray = zeros((n, param.NBIT_K + 1))
        if self.pid > 0:
            for i in range(n):
                ep[i][0] = c[i]
                for j in range(1, param.NBIT_K + 1):
                    ep[i][j] = ((1 - 2 * ebits[i][j - 1]) * rbits[i][j - 1]) % field
                    if self.pid == 1:
                        ep[i][j] = add_func(ep[i][j], ebits[i][j - 1])
        
        E: np.ndarray = self.prefix_or(ep, fid)

        tpneg: np.ndarray = zeros((n, param.NBIT_K))
        if self.pid > 0:
            for i in range(n):
                for j in range(param.NBIT_K):
                    tpneg[i][j] = (int(E[i][j]) - (1 - ebits[i][j]) * rbits[i][j]) % field
        
        Tneg: np.ndarray = self.prefix_or(tpneg, fid)
        half_len: int = param.NBIT_K // 2

        efir: np.ndarray = zeros((n, param.NBIT_K))
        rfir: np.ndarray = zeros((n, param.NBIT_K))
        if self.pid > 0:
            efir = mul_func(ebits, Tneg)
        rfir = self.arithmetic.multiply(rbits, Tneg, True, self.primes[fid])

        double_flag: np.ndarray = self.less_than_bits(efir, rfir, fid)

        odd_bits = zeros((n, half_len))
        even_bits = zeros((n, half_len))

        if self.pid > 0:
            for i in range(n):
                for j in range(half_len):
                    odd_bits[i][j] = np.mod(1 - Tneg[i][2 * j + 1], field) if self.pid == 1 else np.mod(-Tneg[i][2 * j + 1], field)
                    if ((2 * j + 2) < param.NBIT_K):
                        even_bits[i][j] = np.mod(1 - Tneg[i][2 * j + 2], field) if self.pid == 1 else np.mod(-Tneg[i][2 * j + 2], field)
                    else:
                        even_bits[i][j] = 0

        odd_bit_sum = zeros(n)
        even_bit_sum = zeros(n)
        
        for i in range(n):
            odd_bit_sum[i] = reduce(add_func, odd_bits[i], 0)
            even_bit_sum[i] = reduce(add_func, even_bits[i], 0)

        if self.pid == 1:
            odd_bit_sum = add_func(odd_bit_sum, 1)
            even_bit_sum = add_func(even_bit_sum, 1)
        
        # If double_flag = true, then use odd_bits, otherwise use even_bits

        diff = zeros(n)
        if self.pid != 0:
            diff: np.ndarray = np.mod(odd_bit_sum - even_bit_sum, field)

        diff = self.arithmetic.multiply(double_flag, diff, True, self.primes[fid])
        
        chosen_bit_sum = zeros(n)
        if self.pid != 0:
            chosen_bit_sum = add_func(even_bit_sum, diff)
        
        b_mat: np.ndarray = self.polynomial.table_lookup(chosen_bit_sum, 1)

        if self.pid > 0:
            b_sqrt: np.ndarray = b_mat[0]
            b: np.ndarray = b_mat[1]
            return b, b_sqrt
        
        return zeros(n), zeros(n)

    def less_than_bits(self: 'MPCEnv', a: np.ndarray, b: np.ndarray, fid: int) -> np.ndarray:
        return self.less_than_bits_aux(a, b, 0, fid)
    
    def less_than_bits_aux(self: 'MPCEnv', a: np.ndarray, b: np.ndarray, public_flag: int, fid: int) -> np.ndarray:
        assert a.shape == b.shape

        n: int = a.shape[0]
        L: int = a.shape[1]
        field: int = self.primes[fid]
        mul_func = partial(mul_mod, field=field)
        add_func = partial(add_mod, field=field)

        # Calculate XOR
        x = zeros((n, L))

        if public_flag == 0:
            x: np.ndarray = self.arithmetic.multiply(a, b, True, self.primes[fid])
            if self.pid > 0:
                x = np.mod(add_func(a, b) - add_func(x, x), field)
        elif self.pid > 0:
            x = mul_func(a, b)
            x = np.mod(add_func(a, b) - add_func(x, x), field)
            if self.pid == 2:
                x = np.mod(x - (a if public_flag == 1 else b), field)
        
        f: np.ndarray = self.prefix_or(x, fid)

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
                    c[i] = reduce(add_func, fb[i], 0)
            
            return c
        
        # TODO: optimize
        f_arr = [zeros((1, L)) for _ in range(n)]
        b_arr = [zeros((L, 1)) for _ in range(n)]

        if self.pid > 0:
            for i in range(n):
                f_arr[i][0][:] = f[i]
                for j in range(L):
                    b_arr[i][j][0] = b[i][j]

        c_arr: list = self.arithmetic.mult_mat_parallel(f_arr, b_arr, self.primes[fid])
        
        return np.array([c_arr[i][0][0] if self.pid > 0 else 0 for i in range(n)], dtype=np.int64)

    def fp_sqrt(self: 'MPCEnv', a: np.ndarray) -> tuple:
        n: int = len(a)
        fid: int = 0
        field: int = self.primes[fid]

        if n > param.DIV_MAX_N:
            nbatch: int = math.ceil(n / param.DIV_MAX_N)
            b = zeros(n)
            b_inv = zeros(n)
            
            for i in range(nbatch):
                start: int = param.DIV_MAX_N * i
                end: int = start + param.DIV_MAX_N
                if end > n: end = n
                batch_size: int = end - start
                a_copy = zeros(batch_size)
                
                for j in range(batch_size):
                    a_copy[j] = a[start + j]
                
                b_copy, b_inv_copy = self.fp_sqrt(a_copy)
                
                for j in range(batch_size):
                    b[start + j] = b_copy[j]
                    b_inv[start + j] = b_inv_copy[j]
            
            return b

        # TODO: Currently using the same iter as division -- possibly need to update
        niter: int = 2 * math.ceil(math.log2((param.NBIT_K) / 3.5))

        # Initial approximation: 1 / sqrt(a_scaled) ~= 2.9581 - 4 * a_scaled + 2 * a_scaled^2
        # Bottleneck
        s, s_sqrt = self.normalizer_even_exp(a)

        a_scaled: np.ndarray = self.arithmetic.multiply(a, s, elem_wise=True, field=self.primes[fid])
        a_scaled = self.trunc(a_scaled, param.NBIT_K, param.NBIT_K - param.NBIT_F, fid=fid)

        a_scaled_sq: np.ndarray = self.arithmetic.multiply(a_scaled, a_scaled, elem_wise=True, field=self.primes[fid])
        a_scaled_sq = self.trunc(a_scaled_sq, fid=fid)

        scaled_est = zeros(n)
        
        if self.pid != 0:
            scaled_est = add_mod(
                mul_mod(-a_scaled, 4, field), add_mod(a_scaled_sq, a_scaled_sq, field), field)
            if self.pid == 1:
                coeff: int = TypeOps.double_to_fp(2.9581, param.NBIT_K, param.NBIT_F)
                scaled_est = add_mod(scaled_est, coeff, field)

        # TODO: Make h_and_g a ndarray
        h_and_g: list = [zeros((1, n)) for _ in range(2)]

        h_and_g[0][0][:] = self.arithmetic.multiply(scaled_est, s_sqrt, elem_wise=True)
        # Our scaled initial approximation (scaled_est) has bit length <= NBIT_F + 2
        # and s_sqrt is at most NBIT_K/2 bits, so their product is at most NBIT_K/2 +
        # NBIT_F + 2
        h_and_g[0] = self.trunc(
            h_and_g[0], param.NBIT_K // 2 + param.NBIT_F + 2, (param.NBIT_K - param.NBIT_F) // 2 + 1, fid=0)

        h_and_g[1][0][:] = add_mod(h_and_g[0][0], h_and_g[0][0], field)
        h_and_g[1][0][:] = self.arithmetic.multiply(h_and_g[1][0], a, elem_wise=True)
        h_and_g[1] = self.trunc(h_and_g[1], k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, fid=0)

        onepointfive: int = TypeOps.double_to_fp(1.5, param.NBIT_K, param.NBIT_F)

        for _ in range(niter):
            r: np.ndarray = self.arithmetic.multiply(h_and_g[0], h_and_g[1], elem_wise=True)
            r = self.trunc(r, k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, fid=0)
            r = np.mod(-r, field)
            if self.pid == 1:
                r[0][:] = add_mod(r[0], onepointfive, field)

            r_dup: list = [r, r]

            h_and_g: list = self.arithmetic.mult_aux_parallel(h_and_g, r_dup, True)
            # TODO: write a version of Trunc with parallel processing (easy with h_and_g as ndarray)
            h_and_g[0] = self.trunc(h_and_g[0], k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, fid=0)
            h_and_g[1] = self.trunc(h_and_g[1], k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, fid=0)

        b_inv = add_mod(h_and_g[0][0], h_and_g[0][0], field)
        b = h_and_g[1][0]
        
        return b, b_inv
    
    def householder(self: 'MPCEnv', x: np.ndarray) -> np.ndarray:
        n: int = len(x)
        fid: int = 0
        field: int = self.primes[fid]
        add_func = partial(add_mod, field=field)
        mul_func = partial(mul_mod, field=field)
        
        xr, xm = self.arithmetic.beaver_partition(x)

        xdot: np.ndarray = self.arithmetic.beaver_inner_prod(xr, xm)
        xdot = np.array([xdot], dtype=np.int64)
        xdot = self.arithmetic.beaver_reconstruct(xdot)
        xdot = self.trunc(xdot)

        # Bottleneck
        xnorm, _ = self.fp_sqrt(xdot)

        x1 = np.array([x[0]], dtype=np.int64)
        x1sign: np.ndarray = self.is_positive(x1)

        x1sign = add_func(x1sign, x1sign)
        if self.pid == 1:
            x1sign[0] = (x1sign[0] - 1) % field

        shift: np.ndarray = self.arithmetic.multiply(xnorm, x1sign, True)

        sr, sm = self.arithmetic.beaver_partition(shift[0])

        xr_0: np.ndarray = np.expand_dims(xr[0], axis=0)
        xm_0: np.ndarray = np.expand_dims(xm[0], axis=0)
        dot_shift: np.ndarray = self.arithmetic.beaver_mult(xr_0, xm_0, sr, sm, True)
        dot_shift = self.arithmetic.beaver_reconstruct(dot_shift)
        dot_shift = self.trunc(dot_shift, fid=0)

        vdot = zeros(1)
        if self.pid > 0:
            vdot = mul_func(add_func(xdot, dot_shift), 2)

        # Bottleneck
        _, vnorm_inv = self.fp_sqrt(vdot)

        invr, invm = self.arithmetic.beaver_partition(vnorm_inv[0])

        vr = zeros(n)
        if self.pid > 0:
            vr = xr.copy()
            vr[0] = add_func(vr[0], sr)
        vm = xm.copy()
        vm[0] = add_func(vm[0], sm)

        v: np.ndarray = self.arithmetic.beaver_mult(vr, vm, invr, invm, True)
        v = self.arithmetic.beaver_reconstruct(v)
        v = self.trunc(v, fid=0)

        return v
    
    def is_positive(self: 'MPCEnv', a: np.ndarray) -> np.ndarray:
        n: int = len(a)
        nbits: int = self.primes_bits[0]
        fid: int = 2
        field: int = self.primes[fid]

        r: np.ndarray = None
        r_bits: np.ndarray = None
        if self.pid == 0:
            r: np.ndarray = random_ndarray(base=self.primes[0], shape=n)
            r_bits: np.ndarray = self.num_to_bits(r, nbits)

            self.prg.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(base=self.primes[0], shape=n)
            r_bits_mask: np.ndarray = random_ndarray(base=field, shape=(n, nbits))
            self.prg.restore_seed(1)

            r -= r_mask
            r_bits -= r_bits_mask
            r %= self.primes[0]
            r_bits %= field

            self.comms.send_elem(r, 2)
            self.comms.send_elem(r_bits, 2)
        elif self.pid == 2:
            r: np.ndarray = self.comms.receive_vector(0, msg_len=TypeOps.get_vec_len(n), shape=n)
            r_bits: np.ndarray = self.comms.receive_matrix(0, msg_len=TypeOps.get_mat_len(n, nbits), shape=(n, nbits))
        else:
            self.prg.switch_seed(0)
            r: np.ndarray = random_ndarray(base=self.primes[0], shape=n)
            r_bits: np.ndarray = random_ndarray(base=field, shape=(n, nbits))
            self.prg.restore_seed(0)

        c: np.ndarray = zeros(1)
        if self.pid != 0:
            c = add_mod(add_mod(a, a, self.primes[0]), r, self.primes[0])

        c = self.comms.reveal_sym(c, field=self.primes[0])

        c_bits = zeros((n, nbits))
        if self.pid != 0:
            c_bits = self.num_to_bits(c, nbits)

        # Incorrect result if r = 0, which happens with probaility 1 / BASE_P
        no_overflow: np.ndarray = self.less_than_bits_public(r_bits, c_bits, fid=fid)

        c_xor_r = zeros(n)
        if self.pid > 0:
            # Warning: Overflow might occur below in numpy.
            for i in range(n):
                c_xor_r[i] = r_bits[i][nbits - 1] - 2 * c_bits[i][nbits - 1] * r_bits[i][nbits - 1]
                if self.pid == 1:
                    c_xor_r[i] += c_bits[i][nbits - 1]
            c_xor_r %= field
        
        lsb: np.ndarray = self.arithmetic.multiply(c_xor_r, no_overflow, True, self.primes[fid])
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
    
    def qr_fact_square(self: 'MPCEnv', A: np.ndarray) -> np.ndarray:
        assert A.shape[0] == A.shape[1]

        n: int = A.shape[0]
        fid: int = 0
        field: int = self.primes[fid]
        add_func = partial(add_mod, field=field)

        R = zeros((n, n))
        Q = zeros((n, n))

        Ap = zeros((n, n))
        if self.pid != 0:
            Ap = A

        one: int = TypeOps.double_to_fp(1, param.NBIT_K, param.NBIT_F)

        for i in range(n - 1):
            v = np.expand_dims(self.householder(Ap[0]), axis=0)
            vt = v.T
            
            P: np.ndarray = self.arithmetic.multiply(vt, v, False)
            P = self.trunc(P, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            if self.pid > 0:
                P = np.mod(-add_func(P, P), self.primes[0])
                if self.pid == 1:
                    np.fill_diagonal(P, add_func(P.diagonal(), one))
            
            B = zeros((n - i, n - i))
            if i == 0:
                Q = P
                B = self.arithmetic.multiply(Ap, P, False)
                B = self.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            else:
                Qsub = zeros((n - i, n))
                if self.pid > 0:
                    Qsub[:n - i] = Q[i: n]

                left: list = [P, Ap]
                right: list = [Qsub, P]

                prod: list = self.arithmetic.mult_mat_parallel(left, right)
                # TODO: parallelize Trunc
                prod[0] = self.trunc(prod[0], param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
                prod[1] = self.trunc(prod[1], param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

                if self.pid > 0:
                    Q[i:n] = prod[0][:n - i]
                    B = prod[1]
            
            Ap = zeros((n - i - 1, n - i - 1))
            if self.pid > 0:
                R[i:n, i] = B[:n-i, 0]
                if i == n - 2: R[n - 1][n - 1] = B[1][1]
                Ap[:n - i - 1, :n - i - 1] = B[1:n - i, 1:n - i]
            
        return Q, R

    def tridiag(self: 'MPCEnv', A: np.ndarray) -> tuple:
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] > 2

        n: int = A.shape[0]
        one: int = TypeOps.double_to_fp(1, param.NBIT_K, param.NBIT_F)
        add_func = partial(add_mod, field=self.primes[0])

        Q = zeros((n, n))
        T = zeros((n, n))
        if self.pid > 0:
            if self.pid == 1:
                for i in range(n):
                    np.fill_diagonal(Q, one)

        Ap = zeros((n, n))
        if self.pid != 0:
            Ap = A

        for i in range(n - 2):
            x = zeros((Ap.shape[1] - 1))
            if self.pid > 0:
                x[:Ap.shape[1] - 1] = Ap[0][1:Ap.shape[1]]

            v = np.expand_dims(self.householder(x), axis=0)
            vt = v.T

            vv: np.ndarray = self.arithmetic.multiply(vt, v, False)
            vv = self.trunc(vv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            P = zeros(Ap.shape)
            if self.pid > 0:
                cols_no = Ap.shape[1]
                P[1:cols_no, 1:cols_no] = np.mod(
                    -add_func(vv[0:cols_no-1, 0:cols_no-1], vv[0:cols_no-1, 0:cols_no-1]), self.primes[0])
                if self.pid == 1:
                    np.fill_diagonal(P, add_func(P.diagonal(), one))

            # TODO: parallelize? (minor improvement)
            PAp: np.ndarray = self.arithmetic.multiply(P, Ap, False)
            PAp = self.trunc(PAp, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            B = self.arithmetic.multiply(PAp, P, False)
            B = self.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            Qsub = zeros((n, n - i))
            if self.pid > 0:
                Qsub[:, :n - i] = Q[:, i:n]

            Qsub: np.ndarray = self.arithmetic.multiply(Qsub, P, False)
            Qsub = self.trunc(Qsub, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            if self.pid > 0:
                Q[:, i:n] = Qsub[:, :n - i]

            if self.pid > 0:
                T[i][i] = B[0][0]
                T[i + 1][i] = B[1][0]
                T[i][i + 1] = B[0][1]
                if i == n - 3:
                    T[i + 1][i + 1] = B[1][1]
                    T[i + 1][i + 2] = B[1][2]
                    T[i + 2][i + 1] = B[2][1]
                    T[i + 2][i + 2] = B[2][2]

            Ap = zeros((B.shape[0] - 1, B.shape[1] - 1))
            if self.pid > 0:
                Ap[:B.shape[0] - 1, :B.shape[1] - 1] = B[1:B.shape[0], 1:B.shape[1]]

        return T, Q

    def eigen_decomp(self: 'MPCEnv', A: np.ndarray) -> tuple:
        assert A.shape[0] == A.shape[1]
        n: int = A.shape[0]
        add_func = partial(add_mod, field=self.primes[0])

        L = zeros(n)

        Ap, Q = self.tridiag(A)
        V: np.ndarray = Q.T

        for i in range(n - 1, 0, -1):
            for _ in range(param.ITER_PER_EVAL):
                shift = Ap[i][i]
                if self.pid > 0:
                    np.fill_diagonal(Ap, np.mod(Ap.diagonal() - shift, self.primes[0]))

                Q, R = self.qr_fact_square(Ap)

                Ap = self.arithmetic.multiply(Q, R, False)
                Ap = self.trunc(Ap, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

                if self.pid > 0:
                    np.fill_diagonal(Ap, add_func(Ap.diagonal(), shift))

                Vsub = zeros((i + 1, n))
                if self.pid > 0:
                    Vsub[:i + 1] = V[:i + 1]

                Vsub = self.arithmetic.multiply(Q, Vsub, False)
                Vsub = self.trunc(Vsub, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

                if self.pid > 0:
                    V[:i + 1] = Vsub[:i + 1]

            L[i] = Ap[i][i]
            if i == 1:
                L[0] = Ap[0][0]

            Ap_copy: np.ndarray = Ap.copy()
            Ap = zeros((i, i))
            if self.pid > 0:
                Ap = Ap_copy[:i, :i]
        
        return V, L

    def orthonormal_basis(self: 'MPCEnv', A: np.ndarray) -> np.ndarray:
        assert A.shape[1] >= A.shape[0]
        add_func: callable = partial(add_mod, field=self.primes[0])

        c: int = A.shape[0]
        n: int = A.shape[1]

        # TODO: Make v_list an ndarray
        v_list: list = []

        Ap = zeros((c, n))
        if self.pid != 0:
            # TODO: Remove copy
            Ap = A.copy()

        one: int = TypeOps.double_to_fp(1, param.NBIT_K, param.NBIT_F)

        for i in range(c):
            v = zeros((1, Ap.shape[1]))
            v[0] = self.householder(Ap[0])

            if self.pid == 0:
                v_list.append(zeros(Ap.shape[1]))
            else:
                v_list.append(v[0])

            vt = zeros((Ap.shape[1], 1))
            if self.pid != 0:
                vt = v.T

            Apv = self.arithmetic.multiply(Ap, vt, False)
            Apv = self.trunc(Apv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            B = self.arithmetic.multiply(Apv, v, False)
            B = self.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            if self.pid > 0:
                B = np.mod(-B, self.primes[0])
                B = add_func(B, B)
                B = add_func(B, Ap)

            Ap = zeros((B.shape[0] - 1, B.shape[1] - 1))
            if self.pid > 0:
                # TODO: Vectorize
                for j in range(B.shape[0] - 1):
                    for k in range(B.shape[1] - 1):
                        Ap[j][k] = B[j + 1][k + 1]

        Q = zeros((c, n))
        if self.pid > 0:
            if self.pid == 1:
                # TODO: Vectorize
                for i in range(c):
                    Q[i][i] = one

        # TODO: Vectorize
        for i in range(c - 1, -1, -1):
            v = zeros((1, len(v_list[i])))
            if self.pid > 0:
                v[0] = v_list[i]

            vt = zeros((v.shape[1], 1))
            if self.pid != 0:
                vt = v.T

            Qsub = zeros((c, n - i))
            if self.pid > 0:
                # TODO: Vectorize
                for j in range(c):
                    for k in range(n - i):
                        Qsub[j][k] = Q[j][k + i]

            Qv = self.arithmetic.multiply(Qsub, vt, False)
            Qv = self.trunc(Qv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            Qvv = self.arithmetic.multiply(Qv, v, False)
            Qvv = self.trunc(Qvv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            if self.pid > 0:
                Qvv = np.mod(-Qvv, self.primes[0])
                Qvv = add_func(Qvv, Qvv)

            if self.pid > 0:
                # TODO: Vectorize
                for j in range(c):
                    for k in range(n - i):
                        Q[j][k + i] = add_func(Q[j][k + i], Qvv[j][k])

        return Q
