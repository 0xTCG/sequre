import math

from functools import partial

import numpy as np

import utils.param as param

from utils.type_ops import TypeOps
from utils.utils import random_ndarray
from utils.custom_types import zeros, add_mod, mul_mod
from mpc.comms import Comms
from mpc.prg import PRG
from mpc.arithmetic import Arithmetic
from mpc.table import Table
from mpc.boolean import Boolean


class FloatingPoint:
    def __init__(self: 'FloatingPoint', pid: int, primes: dict, prg: PRG, comms: Comms, arithmetic: Arithmetic, boolean: Boolean, table: Table):
        self.pid = pid
        self.primes = primes
        self.prg = prg
        self.comms = comms
        self.arithmetic = arithmetic
        self.table = table
        self.boolean = boolean
        self.invpow_cache: dict = dict()
    
    def print_fp(self: 'FloatingPoint', mat: np.ndarray, field: int) -> np.ndarray:
        if self.pid == 0:
            return None
        revealed_mat: np.ndarray = self.comms.reveal_sym(mat, field=field)
        mat_float: np.ndarray = TypeOps.fp_to_double(revealed_mat, param.NBIT_K, param.NBIT_F, field=field)

        if self.pid == 2:
            print(f'{self.pid}: {mat_float}')
        
        return mat_float
    
    def trunc(self: 'FloatingPoint', a: np.ndarray, k: int = param.NBIT_K + param.NBIT_F,
              m: int = param.NBIT_F, field: int = param.BASE_P) -> np.ndarray:
        msg_len: int = TypeOps.get_bytes_len(a)
        
        if self.pid == 0:
            r: np.ndarray = TypeOps.rand_bits(a.shape, k + param.NBIT_V, field=field)
            r_low: np.ndarray = zeros(a.shape)
            
            r_low = np.mod(r & ((1 << m) - 1), field)

            self.prg.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(base=field, shape=a.shape)
            r_low_mask: np.ndarray = random_ndarray(base=field, shape=a.shape)
            self.prg.restore_seed(1)

            r = np.mod(r - r_mask, field)
            r_low = np.mod(r_low - r_low_mask, field)

            self.comms.send_elem(r, 2)
            self.comms.send_elem(r_low, 2)
        elif self.pid == 2:
            r: np.ndarray = self.comms.receive_ndarray(
                from_pid=0, msg_len=msg_len, ndim=a.ndim, shape=a.shape)
            r_low: np.ndarray = self.comms.receive_ndarray(
                from_pid=0, msg_len=msg_len, ndim=a.ndim, shape=a.shape)
        else:
            self.prg.switch_seed(0)
            r: np.ndarray = random_ndarray(base=field, shape=a.shape)
            r_low: np.ndarray = random_ndarray(base=field, shape=a.shape)
            self.prg.restore_seed(0)

        c = add_mod(a, r, field) if self.pid > 0 else zeros(a.shape)
        c = self.comms.reveal_sym(c, field=field)

        c_low: np.ndarray = zeros(a.shape)
        if self.pid > 0:
            c_low = np.mod(c & ((1 << m) - 1), field)

        if self.pid > 0:
            a = add_mod(a, r_low, field)
            if self.pid == 1:
                a = np.mod(a - c_low, field)

            if m not in self.invpow_cache:
                twoinv: int = TypeOps.mod_inv(2, field)
                twoinvm = pow(twoinv, m, field)
                self.invpow_cache[m] = twoinvm
                
            a = mul_mod(a, self.invpow_cache[m], field)
        
        return a

    def int_to_fp(self: 'FloatingPoint', a: int, k: int, f: int, field: int) -> int:
        sn = 1 if a >= 0 else -1

        az_shift: int = TypeOps.left_shift(a, f)
        az_trunc: int = TypeOps.trunc_elem(az_shift, k - 1)

        return (az_trunc * sn) % field

    def normalizer_even_exp(self: 'MPCEnv', a: np.ndarray) -> tuple:
        n: int = len(a)
        fid: int = 1
        field: int = self.primes[fid]
        add_func = partial(add_mod, field=field)
        mul_func = partial(mul_mod, field=field)

        r, rbits = self.__share_random_bits(param.NBIT_K, n, field)

        # Warning: a + r might overflow in numpy.
        e = zeros(n) if self.pid == 0 else a + r
        e = self.comms.reveal_sym(e, field=self.primes[0])

        ebits: np.ndarray = zeros(
            (n, param.NBIT_K)) if self.pid == 0 else TypeOps.num_to_bits(e, param.NBIT_K)
        
        c: np.ndarray = self.boolean.less_than_bits_public(rbits, ebits, field)

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
        
        E: np.ndarray = self.boolean.prefix_or(ep, field)

        tpneg: np.ndarray = zeros((n, param.NBIT_K))
        if self.pid > 0:
            for i in range(n):
                for j in range(param.NBIT_K):
                    tpneg[i][j] = (int(E[i][j]) - (1 - ebits[i][j]) * rbits[i][j]) % field
        
        Tneg: np.ndarray = self.boolean.prefix_or(tpneg, field)
        half_len: int = param.NBIT_K // 2

        efir: np.ndarray = zeros((n, param.NBIT_K))
        rfir: np.ndarray = zeros((n, param.NBIT_K))
        if self.pid > 0:
            efir = mul_func(ebits, Tneg)
        rfir = self.arithmetic.multiply(rbits, Tneg, True, field)

        double_flag: np.ndarray = self.boolean.less_than_bits(efir, rfir, field)

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
        
        # TODO: Implement vectorized sum over axis
        for i in range(n):
            for j in range(half_len):
                odd_bit_sum[i] = add_func(odd_bit_sum[i], odd_bits[i][j])
                even_bit_sum[i] = add_func(even_bit_sum[i], even_bits[i][j])

        if self.pid == 1:
            odd_bit_sum = add_func(odd_bit_sum, 1)
            even_bit_sum = add_func(even_bit_sum, 1)
        
        # If double_flag = true, then use odd_bits, otherwise use even_bits

        diff = zeros(n)
        if self.pid != 0:
            diff: np.ndarray = np.mod(odd_bit_sum - even_bit_sum, field)

        diff = self.arithmetic.multiply(double_flag, diff, True, field)
        
        chosen_bit_sum = zeros(n)
        if self.pid != 0:
            chosen_bit_sum = add_func(even_bit_sum, diff)
        
        b_mat: np.ndarray = self.table.table_lookup(chosen_bit_sum, 1, field=self.primes[0])

        if self.pid > 0:
            b_sqrt: np.ndarray = b_mat[0]
            b: np.ndarray = b_mat[1]
            return b, b_sqrt
        
        return zeros(n), zeros(n)

    
    def fp_div(self: 'FloatingPoint', a: np.ndarray, b: np.ndarray, field: int) -> np.ndarray:
        assert len(a) == len(b)

        n: int = len(a)
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

                c_copy: np.ndarray = self.fp_div(a_copy, b_copy, field=field)
                for j in range(batch_size):
                    c[start + j] = c_copy[j]
            return c

        niter: int = 2 * math.ceil(math.log2(param.NBIT_K / 3.5)) + 1

        # Initial approximation: 1 / x_scaled ~= 5.9430 - 10 * x_scaled + 5 * x_scaled^2
        s, _ = self.normalizer_even_exp(b)

        b_scaled: np.ndarray = self.arithmetic.multiply(b, s, True, field=field)
        b_scaled = self.trunc(b_scaled, param.NBIT_K, param.NBIT_K - param.NBIT_F, field=field)

        b_scaled_sq: np.ndarray = self.arithmetic.multiply(b_scaled, b_scaled, True, field=field)
        b_scaled_sq = self.trunc(b_scaled_sq, field=field)

        scaled_est = zeros(n)
        if self.pid != 0:
            scaled_est = np.mod(
                mul_func(b_scaled_sq, 5) - mul_func(b_scaled, 10), field)
            if self.pid == 1:
                coeff: int = TypeOps.double_to_fp(5.9430, param.NBIT_K, param.NBIT_F, field=field)
                scaled_est = add_func(scaled_est, coeff)

        w: np.ndarray = self.arithmetic.multiply(scaled_est, s, True, field=field)
        # scaled_est has bit length <= NBIT_F + 2, and s has bit length <= NBIT_K
        # so the bit length of w is at most NBIT_K + NBIT_F + 2
        w = self.trunc(w, param.NBIT_K + param.NBIT_F + 2, param.NBIT_K - param.NBIT_F, field=field)

        x: np.ndarray = self.arithmetic.multiply(w, b, True, field=field)
        x = self.trunc(x, field=field)

        one: int = self.int_to_fp(1, param.NBIT_K, param.NBIT_F, field=field)

        x = np.mod(-x, field)
        if self.pid == 1:
            x = add_func(x, one)
        
        y: np.ndarray = self.arithmetic.multiply(a, w, True, field=field)
        y = self.trunc(y, field=field)

        for _ in range(niter):
            xr, xm = self.arithmetic.beaver_partition(x, field=field)
            yr, ym = self.arithmetic.beaver_partition(y, field=field)

            xpr = xr.copy()
            if self.pid > 0:
                xpr = add_func(xpr, one)

            y = self.arithmetic.beaver_mult(yr, ym, xpr, xm, True, field=param.BASE_P)
            x = self.arithmetic.beaver_mult(xr, xm, xr, xm, True, field=param.BASE_P)

            x = self.arithmetic.beaver_reconstruct(x, field=field)
            y = self.arithmetic.beaver_reconstruct(y, field=field)

            x = self.trunc(x, field=field)
            y = self.trunc(y, field=field)

        if self.pid == 1:
            x = add_func(x, one)
            
        c: np.ndarray = self.arithmetic.multiply(y, x, True, field=field)
        
        return self.trunc(c, field=field)
    
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

        a_scaled: np.ndarray = self.arithmetic.multiply(a, s, elem_wise=True, field=field)
        a_scaled = self.trunc(a_scaled, param.NBIT_K, param.NBIT_K - param.NBIT_F, field=field)

        a_scaled_sq: np.ndarray = self.arithmetic.multiply(a_scaled, a_scaled, elem_wise=True, field=field)
        a_scaled_sq = self.trunc(a_scaled_sq, field=field)

        scaled_est = zeros(n)
        
        if self.pid != 0:
            scaled_est = add_mod(
                mul_mod(-a_scaled, 4, field), add_mod(a_scaled_sq, a_scaled_sq, field), field)
            if self.pid == 1:
                coeff: int = TypeOps.double_to_fp(2.9581, param.NBIT_K, param.NBIT_F, field=self.primes[0])
                scaled_est = add_mod(scaled_est, coeff, field)

        # TODO: Make h_and_g a ndarray
        h_and_g: list = [zeros((1, n)) for _ in range(2)]

        h_and_g[0][0][:] = self.arithmetic.multiply(scaled_est, s_sqrt, elem_wise=True, field=self.primes[0])
        # Our scaled initial approximation (scaled_est) has bit length <= NBIT_F + 2
        # and s_sqrt is at most NBIT_K/2 bits, so their product is at most NBIT_K/2 +
        # NBIT_F + 2
        h_and_g[0] = self.trunc(
            h_and_g[0], param.NBIT_K // 2 + param.NBIT_F + 2, (param.NBIT_K - param.NBIT_F) // 2 + 1, field=self.primes[0])

        h_and_g[1][0][:] = add_mod(h_and_g[0][0], h_and_g[0][0], field)
        h_and_g[1][0][:] = self.arithmetic.multiply(h_and_g[1][0], a, elem_wise=True, field=self.primes[0])
        h_and_g[1] = self.trunc(h_and_g[1], k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, field=self.primes[0])

        onepointfive: int = TypeOps.double_to_fp(1.5, param.NBIT_K, param.NBIT_F, field=self.primes[0])

        for _ in range(niter):
            r: np.ndarray = self.arithmetic.multiply(h_and_g[0], h_and_g[1], elem_wise=True, field=self.primes[0])
            r = self.trunc(r, k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, field=self.primes[0])
            r = np.mod(-r, field)
            if self.pid == 1:
                r[0][:] = add_mod(r[0], onepointfive, field)

            r_dup: list = [r, r]

            h_and_g: list = self.arithmetic.mult_aux_parallel(h_and_g, r_dup, True, field=field)
            # TODO: write a version of Trunc with parallel processing (easy with h_and_g as ndarray)
            h_and_g[0] = self.trunc(h_and_g[0], k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, field=self.primes[0])
            h_and_g[1] = self.trunc(h_and_g[1], k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, field=self.primes[0])

        b_inv = add_mod(h_and_g[0][0], h_and_g[0][0], field)
        b = h_and_g[1][0]
        
        return b, b_inv
    
    def __share_random_bits(self: 'MPCEnv', k: int, n: int, field: int) -> tuple:
        if self.pid == 0:
            r: np.ndarray = TypeOps.rand_bits(n, k + param.NBIT_V, field=field)
            rbits: np.ndarray = TypeOps.num_to_bits(r, k)

            self.prg.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(field, n)
            rbits_mask: np.ndarray = random_ndarray(field, (n, k))
            self.prg.restore_seed(1)

            r -= r_mask
            r %= field

            rbits -= rbits_mask
            rbits %= field

            self.comms.send_elem(r, 2)
            self.comms.send_elem(rbits, 2)
        elif self.pid == 2:
            r: np.ndarray = self.comms.receive_vector(0, msg_len=TypeOps.get_vec_len(n), shape=(n, ))
            rbits: np.ndarray = self.comms.receive_matrix(0, msg_len=TypeOps.get_mat_len(n, k), shape=(n, k))
        else:
            self.prg.switch_seed(0)
            r: np.ndarray = random_ndarray(field, n)
            rbits: np.ndarray = random_ndarray(field, (n, k))
            self.prg.restore_seed(0)
        
        return r, rbits
    