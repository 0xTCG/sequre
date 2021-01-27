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
    
    def __share_random_bits(self: 'MPCEnv', k: int, n: int, fid: int) -> tuple:
        if self.pid == 0:
            r: np.ndarray = TypeOps.rand_bits(n, k + param.NBIT_V, field=self.primes[fid])
            rbits: np.ndarray = TypeOps.num_to_bits(r, k)

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
        
        E: np.ndarray = self.boolean.prefix_or(ep, self.primes[fid])

        tpneg: np.ndarray = zeros((n, param.NBIT_K))
        if self.pid > 0:
            for i in range(n):
                for j in range(param.NBIT_K):
                    tpneg[i][j] = (int(E[i][j]) - (1 - ebits[i][j]) * rbits[i][j]) % field
        
        Tneg: np.ndarray = self.boolean.prefix_or(tpneg, self.primes[fid])
        half_len: int = param.NBIT_K // 2

        efir: np.ndarray = zeros((n, param.NBIT_K))
        rfir: np.ndarray = zeros((n, param.NBIT_K))
        if self.pid > 0:
            efir = mul_func(ebits, Tneg)
        rfir = self.arithmetic.multiply(rbits, Tneg, True, self.primes[fid])

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
        x1sign: np.ndarray = self.boolean.is_positive(x1, self.primes_bits[0], self.primes[2])

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
