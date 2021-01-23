import sys
import time
import random
import math

from functools import partial, reduce
from copy import deepcopy

import numpy as np

import param
from c_socket import CSocket
from connect import connect, open_channel
from custom_types import TypeOps, zeros, ones, random_ndarray, add_mod, mul_mod, matmul_mod
from utils import bytes_to_arr, rand_int

Zp = None
Vector = None
Matrix = None

class MPCEnv:
    def __init__(self: 'MPCEnv'):
        self.sockets: dict = dict()
        self.prg_states: dict = dict()
        self.pid: int = None
        self.pascal_cache: dict = dict()
        self.table_cache: dict = dict()
        self.table_type_modular: dict = dict()
        self.lagrange_cache: dict = dict()
        self.table_field_index: dict = dict()
        self.primes: dict = {0: param.BASE_P, 1: 31, 2: 17}  # Temp hardcoded. Needs to be calcualted on init.
        self.primes_bits: dict = {k: math.ceil(math.log2(v)) for k, v in self.primes.items()}
        self.primes_bytes: dict = {k: (v + 7) // 8 for k, v in self.primes_bits.items()}
        self.invpow_cache: dict = dict()
        self.or_lagrange_cache: dict = dict()
    
    def initialize(self: 'MPCEnv', pid: int, pairs: list) -> bool:
        self.pid = pid

        if (not self.setup_channels(pairs)):
            raise ValueError("MPCEnv::Initialize: failed to initialize communication channels")
            
        if (not self.setup_prgs()):
            raise ValueError("MPCEnv::Initialize: failed to initialize PRGs")

        self.setup_tables()

        return True

    def setup_tables(self: 'MPCEnv'):
        # Lagrange cache
        # Table 0
        table = zeros((1, 2))
        if (self.pid > 0):
            table[0][0] = 1
            table[0][1] = 0

        self.table_type_modular[0] = True
        self.table_cache[0] = table
        self.table_field_index[0] = 2

        # Table 1
        half_len: int = param.NBIT_K // 2
        table = zeros((2, half_len + 1))
        if self.pid > 0:
            for i in range(half_len + 1):
                if i == 0:
                    table[0][i] = 1
                    table[1][i] = 1
                else:
                    table[0][i] = table[0][i - 1] * 2
                    table[1][i] = table[1][i - 1] * 4

        self.table_type_modular[1] = True
        self.table_cache[1] = table
        self.table_field_index[1] = 1

        # Table 2: parameters (intercept, slope) for piecewise-linear approximation
        # of negative log-sigmoid function
        table = zeros((2, 64))
        if self.pid > 0:
            with open('sigmoid_approx.txt') as f:
                for i in range(table.shape[1]):
                    intercept, slope = f.readline().split()
                    fp_intercept: int = self.double_to_fp(
                        float(intercept), param.NBIT_K, param.NBIT_F, fid=0)
                    fp_slope: int = self.double_to_fp(float(slope), param.NBIT_K, param.NBIT_F, fid=0)

                    table[0][i] = fp_intercept
                    table[1][i] = fp_slope

        self.table_type_modular[2] = False
        self.table_cache[2] = table
        self.table_field_index[2] = 0

        for cid in range(len(self.table_cache)):
            nrow: int = self.table_cache[cid].shape[0]
            ncol: int = self.table_cache[cid].shape[1]
            self.lagrange_cache[cid] = zeros((nrow, (2 if self.table_type_modular[cid] else 1) * ncol))

            if self.pid > 0:
                for i in range(nrow):
                    x = [0] * (ncol * (2 if self.table_type_modular[cid] else 1))
                    y = zeros(ncol * (2 if self.table_type_modular[cid] else 1))
                    
                    for j in range(ncol):
                        x[j] = j + 1
                        y[j] = self.table_cache[cid][i][j]
                        if (self.table_type_modular[cid]):
                            x[j + ncol] = x[j] + self.primes[self.table_field_index[cid]]
                            y[j + ncol] = self.table_cache[cid][i][j]
                    
                    self.lagrange_cache[cid][i][:] = self.lagrange_interp(x, y, fid=0)
            
        # End of Lagrange cache
    
    def lagrange_interp(self: 'MPCEnv', x: np.ndarray, y: np.ndarray, fid: int) -> np.ndarray:
        n: int = len(y)

        inv_table = dict()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                key: int = abs(x[i] - x[j])
                if key not in inv_table:
                    inv_table[key] = TypeOps.mod_inv(key, self.primes[fid])
        
        # Initialize numer and denom_inv
        numer = zeros((n, n))
        denom_inv = ones(n)
        numer[0][:] = np.mod(y, self.primes[fid])

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                for k in range(n - 1, -1, -1):
                    numer[k][j] = ((0 if k == 0 else int(numer[k - 1][j])) - int(numer[k][j]) * int(x[i])) % self.primes[fid]
                denom_inv[i] = (int(denom_inv[i]) * (1 if x[i] > x[j] else -1) * int(inv_table[abs(x[i] - x[j])])) % self.primes[fid]

        numer_dot = mul_mod(numer, denom_inv, self.primes[fid])
        numer_sum = zeros(n)

        for i in range(n):
            for e in numer_dot[i]:
                numer_sum[i] = add_mod(numer_sum[i], e, self.primes[fid])

        return numer_sum
    
    def setup_channels(self: 'MPCEnv', pairs: list) -> bool:
        for pair in pairs:
            p_1, p_2 = pair

            if (p_1 != self.pid and p_2 != self.pid):
                continue

            port: int = 8000
            if (p_1 == 0 and p_2 == 1):
                port = param.PORT_P0_P1
            elif (p_1 == 0 and p_2 == 2):
                port = param.PORT_P0_P2
            elif (p_1 == 1 and p_2 == 2):
                port = param.PORT_P1_P2
            elif (p_1 == 1 and p_2 == 3):
                port = param.PORT_P1_P3
            elif (p_1 == 2 and p_2 == 3):
                port = param.PORT_P2_P3

            pother: int = p_1 + p_2 - self.pid
            self.sockets[pother] = CSocket(self.pid)

            if (p_1 == self.pid):
                open_channel(self.sockets[pother], port)
            elif (not connect(self.sockets[pother], port)):
                raise ValueError(f"Failed to connect with P{pother}")

        return True

    def setup_prgs(self: 'MPCEnv') -> bool:
        np.random.seed()
        self.prg_states[self.pid] = np.random.get_state() 
        self.import_seed(-1, hash('global'))
        
        for other_pid in set(range(3)) - {self.pid}:
            self.import_seed(other_pid)
        
        self.switch_seed(self.pid)

        return True
    
    def import_seed(self: 'MPCEnv', pid: int, seed: int = None):
        seed: int = hash((min(self.pid, pid), max(self.pid, pid))) if seed is None else seed
        seed %= (1 << 32)
        np.random.seed(seed)
        self.prg_states[pid] = np.random.get_state()

    def receive_bool(self: 'MPCEnv', from_pid: int) -> bool:
        return bool(int(self.sockets[from_pid].receive(msg_len=1)))

    def send_bool(self: 'MPCEnv', flag: bool, to_pid: int):
        self.sockets[to_pid].send(str(int(flag)).encode('utf-8'))

    def send_elem(self: 'MPCEnv', elem: np.ndarray, to_pid: int) -> int:
        return self.sockets[to_pid].send(TypeOps.to_bytes(elem))
    
    def receive_elem(self: 'MPCEnv', from_pid: int, msg_len: int) -> np.ndarray:
        return np.array(int(self.sockets[from_pid].receive(msg_len=msg_len)))

    def receive_vector(self: 'MPCEnv', from_pid: int, msg_len: int, shape: tuple) -> np.ndarray:
        received_vec: np.ndarray = zeros(shape)

        for i, elem in enumerate(bytes_to_arr(self.sockets[from_pid].receive(msg_len=msg_len))):
            received_vec[i] = elem

        return received_vec
    
    def receive_matrix(self: 'MPCEnv', from_pid: int, msg_len: int, shape: tuple) -> np.ndarray:
        matrix: np.ndarray = zeros(shape)
        row_values = self.sockets[from_pid].receive(msg_len=msg_len).split(b';')

        for i, row_value in enumerate(row_values):
            for j, elem in enumerate(bytes_to_arr(row_value)):
                matrix[i][j] = elem
        
        return matrix
    
    def receive_ndarray(self: 'MPCEnv', from_pid: int, msg_len: int, ndim: int, shape: tuple) -> np.ndarray:
        if ndim == 2:
            return self.receive_matrix(from_pid, msg_len, shape)
        
        if ndim == 1:
            return self.receive_vector(from_pid, msg_len, shape)
        
        if ndim == 0:
            return self.receive_elem(from_pid, msg_len)
        
        raise ValueError(f'Invalid dimension expected: {ndim}. Should be either 0, 1 or 2.')

    def clean_up(self: 'MPCEnv'):
        for socket in self.sockets.values():
            socket.close()
  
    def reveal_sym(self: 'MPCEnv', elem: np.ndarray, fid: int = 0) -> np.ndarray:
        if self.pid == 0:
            return elem
        
        msg_len = TypeOps.get_bytes_len(elem)

        received_elem: np.ndarray = None
        if self.pid == 1:
            sent_data = self.send_elem(elem, 3 - self.pid)
            assert sent_data == msg_len, f'Sent {sent_data} bytes but expected {msg_len}'
            received_elem = self.receive_ndarray(3 - self.pid, msg_len=msg_len, ndim=elem.ndim, shape=elem.shape)
        else:
            received_elem = self.receive_ndarray(3 - self.pid, msg_len=msg_len, ndim=elem.ndim, shape=elem.shape)
            sent_data = self.send_elem(elem, 3 - self.pid)
            assert sent_data == msg_len, f'Sent {sent_data} bytes but expected {msg_len}'
            
        return add_mod(elem, received_elem, self.primes[fid])
    
    def switch_seed(self: 'MPCEnv', pid: int):
        self.prg_states[self.pid] = np.random.get_state()
        np.random.set_state(self.prg_states[pid])
    
    def restore_seed(self: 'MPCEnv', pid: int):
        self.prg_states[pid] = np.random.get_state()
        np.random.set_state(self.prg_states[self.pid])

    def beaver_partition(self: 'MPCEnv', x: np.ndarray, fid: int) -> tuple:
        x_: np.ndarray = np.mod(x, self.primes[fid])

        x_r: np.ndarray = zeros(x_.shape)
        r: np.ndarray = zeros(x_.shape)

        if self.pid == 0:
            self.switch_seed(1)
            r_1: np.ndarray = random_ndarray(base=self.primes[fid], shape=x_.shape)
            self.restore_seed(1)

            self.switch_seed(2)
            r_2: np.ndarray = random_ndarray(base=self.primes[fid], shape=x_.shape)
            self.restore_seed(2)

            r: np.ndarray = add_mod(r_1, r_2, self.primes[fid])
        else:
            self.switch_seed(0)
            r: np.ndarray = random_ndarray(base=self.primes[fid], shape=x_.shape)
            self.restore_seed(0)
            
            x_r = (x_ - r) % self.primes[fid]
            x_r = self.reveal_sym(x_r, fid=fid)
        
        return x_r, r
    
    def mul_elem(self: 'MPCEnv', v_1: Vector, v_2: Vector) -> Vector:
        return v_1 * v_2

    def rand_mat(self: 'MPCEnv', m: int, n: int, fid: int) -> Matrix:
        return Matrix(m, n, randomise=True, base=self.primes[fid])

    def get_pascal_matrix(self: 'MPCEnv', power: int) -> np.ndarray:
        if power not in self.pascal_cache:
            pascal_matrix: np.ndarray = self.calculate_pascal_matrix(power)
            self.pascal_cache[power] = pascal_matrix

        return self.pascal_cache[power]
    
    def calculate_pascal_matrix(self: 'MPCEnv', pow: int) -> np.ndarray:
        t = zeros((pow + 1, pow + 1))
        for i in range(pow + 1):
            for j in range(pow + 1):
                if j > i:
                    t[i][j] = 0
                elif j == 0 or j == i:
                    t[i][j] = 1
                else:
                    t[i][j] = t[i - 1][j - 1] + t[i - 1][j]
        
        return t

    def powers(self: 'MPCEnv', x: np.ndarray, power: int, fid: int) -> np.ndarray:
        assert power >= 1, f'Invalid exponent: {power}'

        n: int = len(x)
        b: np.ndarray = np.array([], dtype=np.int64)
        
        if power == 1:
            b.resize((2, n))
            if self.pid > 0:
                if self.pid == 1:
                    b[0] += ones(n)
                b[1][:] = x
        else:  # power > 1
            x_r, r = self.beaver_partition(x, fid)

            if self.pid == 0:
                r_pow: np.ndarray = zeros((power - 1, n))
                r_pow[0][:] = mul_mod(r, r, self.primes[fid])
                
                for p in range(1, r_pow.shape[0]):
                    r_pow[p][:] = mul_mod(r_pow[p - 1], r, self.primes[fid])

                self.switch_seed(1)
                r_: np.ndarray = random_ndarray(base=self.primes[fid], shape=(power - 1, n))
                self.restore_seed(1)

                r_pow = (r_pow - r_) % self.primes[fid]
                self.send_elem(r_pow, 2)

                b.resize((power + 1, n))
            else:
                r_pow: np.ndarray = None
                if self.pid == 1:
                    self.switch_seed(0)
                    r_pow = random_ndarray(base=self.primes[fid], shape=(power - 1, n))
                    self.restore_seed(0)
                else:
                    r_pow = self.receive_matrix(
                        0, msg_len=TypeOps.get_mat_len(power - 1, n),
                        shape=(power - 1, n))

                x_r_pow: np.ndarray = zeros((power - 1, n))
                x_r_pow[0][:] = mul_mod(x_r, x_r, self.primes[fid])
                
                for p in range(1, x_r_pow.shape[0]):
                    x_r_pow[p][:] = mul_mod(x_r_pow[p - 1], x_r, self.primes[fid])

                pascal_matrix: np.ndarray = self.get_pascal_matrix(power)

                b.resize((power + 1, n))

                if self.pid == 1:
                    b[0][:] = add_mod(b[0], ones(n), self.primes[fid])
                b[1][:] = x

                for p in range(2, power + 1):
                    if self.pid == 1:
                        b[p][:] = x_r_pow[p - 2]

                    if p == 2:
                        b[p] = add_mod(
                            b[p],
                            mul_mod(mul_mod(x_r, r, self.primes[fid]), pascal_matrix[p][1], self.primes[fid]),
                            self.primes[fid])
                    else:
                        b[p] = add_mod(
                            b[p],
                            mul_mod(mul_mod(x_r_pow[p - 3], r, self.primes[fid]), pascal_matrix[p][1], self.primes[fid]),
                            self.primes[fid])

                        for j in range(2, p - 1):
                            b[p] = add_mod(
                                b[p],
                                mul_mod(mul_mod(x_r_pow[p - 2 - j], r_pow[j - 2], self.primes[fid]), pascal_matrix[p][j], self.primes[fid]),
                                self.primes[fid])
                        
                        b[p] = add_mod(
                            b[p],
                            mul_mod(mul_mod(x_r, r_pow[p - 3], self.primes[fid]), pascal_matrix[p][p - 1], self.primes[fid]),
                            self.primes[fid])

                    b[p] = add_mod(b[p], r_pow[p - 2], self.primes[fid])
        
        return b
    
    def evaluate_poly(self: 'MPCEnv', x: np.ndarray, coeff: np.ndarray, fid: int) -> np.ndarray:
        n: int = len(x)
        npoly: int = coeff.shape[0]
        deg: int = coeff.shape[1] - 1

        pows: np.ndarray = self.powers(x, deg, fid)

        if self.pid > 0:
            return matmul_mod(coeff, pows, self.primes[fid])
        
        return zeros((npoly, n))
    
    def add_public(self: 'MPCEnv', x: np.ndarray, a: np.ndarray, fid: int) -> np.ndarray:
        if self.pid == 1:
            return add_mod(x, a, self.primes[fid])
        return x
    
    def beaver_mult(
            self: 'MPCEnv', x_r: np.ndarray, r_1: np.ndarray,
            y_r: np.ndarray, r_2: np.ndarray, elem_wise: bool, fid: int) -> np.ndarray:
        mul_func: callable = partial(mul_mod if elem_wise else matmul_mod, field=self.primes[fid])
        
        if self.pid == 0:
            return mul_func(r_1, r_2)

        xy = mul_func(x_r, r_2)
        xy = add_mod(xy, mul_func(r_1, y_r), self.primes[fid])
        if self.pid == 1:
            xy = add_mod(xy, mul_func(x_r, y_r), self.primes[fid])

        return xy

    def beaver_reconstruct(self: 'MPCEnv', elem: np.ndarray, fid: int) -> np.ndarray:
            msg_len: int = TypeOps.get_bytes_len(elem)
            
            if self.pid == 0:
                self.switch_seed(1)
                mask: np.ndarray = random_ndarray(base=self.primes[fid], shape=elem.shape)
                self.restore_seed(1)

                mm: np.ndarray = np.mod(elem - mask, self.primes[fid])
                self.send_elem(mm, 2)
                
                return mm
            else:
                rr: np.ndarray = None
                if self.pid == 1:
                    self.switch_seed(0)
                    rr = random_ndarray(base=self.primes[fid], shape=elem.shape)
                    self.restore_seed(0)
                else:
                    rr = self.receive_ndarray(
                        from_pid=0,
                        msg_len=msg_len,
                        ndim=elem.ndim,
                        shape=elem.shape)
                    
                return add_mod(elem, rr, self.primes[fid])


    def multiply(self: 'MPCEnv', a: np.ndarray, b: np.ndarray, elem_wise: bool, fid: int) -> np.ndarray:
        x_1_r, r_1 = self.beaver_partition(a, fid)
        x_2_r, r_2 = self.beaver_partition(b, fid)
        
        c = self.beaver_mult(x_1_r, r_1, x_2_r, r_2, elem_wise, fid)
        c = self.beaver_reconstruct(c, fid)
        
        return c

    def fp_to_double(self: 'MPCEnv', a: np.ndarray, k: int, f: int, fid: int) -> np.ndarray:
        twokm1: int = TypeOps.left_shift(1, k - 1)

        sn: np.ndarray = np.where(a > twokm1, -1, 1)
        x: np.ndarray = np.where(a > twokm1, self.primes[fid] - a, a)
        x_trunc: np.ndarray = TypeOps.trunc_elem(x, k - 1)
        x_int: np.ndarray = TypeOps.right_shift(x_trunc, f)

        # TODO: consider better ways of doing this?
        x_frac = np.zeros(a.shape)
        for bi in range(f):
            x_frac = np.where(TypeOps.bit(x_trunc, bi) > 0, x_frac + 1, x_frac)
            x_frac /= 2

        return sn * (x_int + x_frac)

    def print_fp(self: 'MPCEnv', mat: np.ndarray, fid: int) -> np.ndarray:
        if self.pid == 0:
            return None
        revealed_mat: np.ndarray = self.reveal_sym(mat, fid=fid)
        mat_float: np.ndarray = self.fp_to_double(revealed_mat, param.NBIT_K, param.NBIT_F, fid=fid)

        if self.pid == 2:
            print(f'{self.pid}: {mat_float}')
        
        return mat_float

    def double_to_fp(self: 'MPCEnv', x: float, k: int, f: int, fid: int) -> int:
        sn: int = 1
        if x < 0:
            x = -x
            sn = -sn

        az: int = int(x)

        az_shift: int = TypeOps.left_shift(az, f)
        az_trunc: int = TypeOps.trunc_elem(az_shift, k - 1)

        xf: float = x - az  # remainder
        for fbit in range(f - 1, -1, -1):
            xf *= 2
            if (xf >= 1):
                xf -= int(xf)
                az_trunc = TypeOps.set_bit(az_trunc, fbit)
        
        return (az_trunc * sn) % self.primes[fid]
    
    def table_lookup(self: 'MPCEnv', x: np.ndarray, table_id: int, fid: int) -> np.ndarray:
        return self.evaluate_poly(x, self.lagrange_cache[table_id], fid=fid)
    
    def rand_bits(self: 'MPCEnv', shape: tuple, num_bits: int, fid: int) -> np.ndarray:
        upper_limit: int = self.primes[fid] - 1
        if num_bits > 63:
            print(f'Warning: Number of bits too big for numpy: {num_bits}')
        else:
            upper_limit = (1 << num_bits) - 1
    
        return random_ndarray(upper_limit, shape=shape) % self.primes[fid]

    def trunc(self: 'MPCEnv', a: np.ndarray, k: int = param.NBIT_K + param.NBIT_F, m: int = param.NBIT_F, fid: int = 0):
        msg_len: int = TypeOps.get_bytes_len(a)
        
        if self.pid == 0:
            r: np.ndarray = self.rand_bits(a.shape, k + param.NBIT_V, fid=fid)
            r_low: np.ndarray = zeros(a.shape)
            
            r_low = np.mod(r & ((1 << m) - 1), self.primes[fid])

            self.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(base=self.primes[fid], shape=a.shape)
            r_low_mask: np.ndarray = random_ndarray(base=self.primes[fid], shape=a.shape)
            self.restore_seed(1)

            r = np.mod(r - r_mask, self.primes[fid])
            r_low = np.mod(r_low - r_low_mask, self.primes[fid])

            self.send_elem(r, 2)
            self.send_elem(r_low, 2)
        elif self.pid == 2:
            r: np.ndarray = self.receive_ndarray(
                from_pid=0, msg_len=msg_len, ndim=a.ndim, shape=a.shape)
            r_low: np.ndarray = self.receive_ndarray(
                from_pid=0, msg_len=msg_len, ndim=a.ndim, shape=a.shape)
        else:
            self.switch_seed(0)
            r: np.ndarray = random_ndarray(base=self.primes[fid], shape=a.shape)
            r_low: np.ndarray = random_ndarray(base=self.primes[fid], shape=a.shape)
            self.restore_seed(0)

        c = add_mod(a, r, self.primes[fid]) if self.pid > 0 else zeros(a.shape)
        c = self.reveal_sym(c, fid=fid)

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
    
    def lagrange_interp_simple(self: 'MPCEnv', y: np.ndarray, fid: int) -> np.ndarray:
        n: int = len(y)
        x = np.arange(1, n + 1)

        return self.lagrange_interp(x, y, fid)

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
            coeff_param = self.lagrange_interp_simple(y, fid) # OR function
            self.or_lagrange_cache[key] = coeff_param

        coeff[0][:] = self.or_lagrange_cache[key]
        bmat = self.evaluate_poly(a_sum, coeff, fid)
        
        return bmat[0]
    
    def reshape(self: 'MPCEnv', a: Matrix, nrows: int, ncols: int):
        if self.pid == 0:
            assert a.shape[0] * a.shape[1] == nrows * ncols
            a.set_dims(nrows, ncols)
        else:
            a.reshape(nrows, ncols)

    def beaver_partition_bulk(self: 'Matrix', x: list, fid: int) -> tuple:
        # TODO: Do this in parallel
        partitions = [self.beaver_partition(e, fid) for e in x]
        x_r = [p[0] for p in partitions]
        r = [p[1] for p in partitions]
        return x_r, r
    
    def beaver_reconstruct_bulk(self: 'Matrix', x: list, fid: int) -> tuple:
        # TODO: Do this in parallel
        return [self.beaver_reconstruct(e, fid) for e in x]

    def mult_aux_parallel(self: 'MPCEnv', a: list, b: list, elem_wise: bool, fid: int) -> list:
        # TODO: Vectorize this method. Make it parallel by having a and b as ndarrays.
        assert len(a) == len(b)
        nmat: int = len(a)

        out_rows = zeros(nmat)
        out_cols = zeros(nmat)

        for k in range(nmat):
            if elem_wise:
                assert a[k].shape == b[k].shape
            else:
                assert a[k].shape[1] == b[k].shape[0]

            out_rows[k] = a[k].shape[0]
            out_cols[k] = a[k].shape[1] if elem_wise else b[k].shape[1]

        ar, am = self.beaver_partition_bulk(a, fid)
        br, bm = self.beaver_partition_bulk(b, fid)

        c = [self.beaver_mult(ar[k], am[k], br[k], bm[k], elem_wise, fid)
             for k in range(nmat)]
        
        return self.beaver_reconstruct_bulk(c, fid)

    def mult_mat_parallel(self: 'MPCEnv', a: list, b: list, fid: int) -> list:
        # TODO: Vectorise/parallelize this method
        return self.mult_aux_parallel(a, b, False, fid)

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

        c = self.mult_mat_parallel(f, tmp, fid)  # c is a concatenation of n 1-by-L matrices

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

        s = self.mult_mat_parallel(f, bdot, fid)

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

        b_scaled: np.ndarray = self.multiply(b, s, True, fid=fid)
        b_scaled = self.trunc(b_scaled, param.NBIT_K, param.NBIT_K - param.NBIT_F, fid=fid)

        b_scaled_sq: np.ndarray = self.multiply(b_scaled, b_scaled, True, fid=fid)
        b_scaled_sq = self.trunc(b_scaled_sq, fid=fid)

        scaled_est = zeros(n)
        if self.pid != 0:
            scaled_est = np.mod(
                mul_func(b_scaled_sq, 5) - mul_func(b_scaled, 10), field)
            if self.pid == 1:
                coeff: int = self.double_to_fp(5.9430, param.NBIT_K, param.NBIT_F, fid=fid)
                scaled_est = add_func(scaled_est, coeff)

        w: np.ndarray = self.multiply(scaled_est, s, True, fid=fid)
        # scaled_est has bit length <= NBIT_F + 2, and s has bit length <= NBIT_K
        # so the bit length of w is at most NBIT_K + NBIT_F + 2
        w = self.trunc(w, param.NBIT_K + param.NBIT_F + 2, param.NBIT_K - param.NBIT_F, fid=fid)

        x: np.ndarray = self.multiply(w, b, True, fid=fid)
        x = self.trunc(x, fid=fid)

        one: int = self.int_to_fp(1, param.NBIT_K, param.NBIT_F, fid=fid)

        x = np.mod(-x, field)
        if self.pid == 1:
            x = add_func(x, one)
        
        y: np.ndarray = self.multiply(a, w, True, fid=fid)
        y = self.trunc(y, fid=fid)

        for _ in range(niter):
            xr, xm = self.beaver_partition(x, fid=fid)
            yr, ym = self.beaver_partition(y, fid=fid)

            xpr = xr.copy()
            if self.pid > 0:
                xpr = add_func(xpr, one)

            y = self.beaver_mult(yr, ym, xpr, xm, True, fid=0)
            x = self.beaver_mult(xr, xm, xr, xm, True, fid=0)

            x = self.beaver_reconstruct(x, fid=fid)
            y = self.beaver_reconstruct(y, fid=fid)

            x = self.trunc(x, fid=fid)
            y = self.trunc(y, fid=fid)

        if self.pid == 1:
            x = add_func(x, one)
            
        c: np.ndarray = self.multiply(y, x, True, fid=fid)
        return self.trunc(c, fid=fid)

    def less_than_bits_public(self: 'MPCEnv', a: np.ndarray, b_pub: np.ndarray, fid: int) -> np.ndarray:
        return self.less_than_bits_aux(a, b_pub, 2, fid)

    def num_to_bits(self: 'MPCEnv', a: np.ndarray, bitlen: int) -> np.ndarray:
        b = zeros((len(a), bitlen))
    
        for i in range(len(a)):
            for j in range(bitlen):
                b[i][j] = int(TypeOps.bit(int(a[i]), bitlen - 1 - j))
    
        return b
    
    def share_random_bits(self: 'MPCEnv', k: int, n: int, fid: int) -> tuple:
        if self.pid == 0:
            r: np.ndarray = self.rand_bits(n, k + param.NBIT_V, fid=fid)
            rbits: np.ndarray = self.num_to_bits(r, k)

            self.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(self.primes[fid], n)
            rbits_mask: np.ndarray = random_ndarray(self.primes[fid], (n, k))
            self.restore_seed(1)

            r -= r_mask
            r %= self.primes[fid]

            rbits -= rbits_mask
            rbits %= self.primes[fid]

            self.send_elem(r, 2)
            self.send_elem(rbits, 2)
        elif self.pid == 2:
            r: np.ndarray = self.receive_vector(0, msg_len=TypeOps.get_vec_len(n), shape=(n, ))
            rbits: np.ndarray = self.receive_matrix(0, msg_len=TypeOps.get_mat_len(n, k), shape=(n, k))
        else:
            self.switch_seed(0)
            r: np.ndarray = random_ndarray(self.primes[fid], n)
            rbits: np.ndarray = random_ndarray(self.primes[fid], (n, k))
            self.restore_seed(0)
        
        return r, rbits

    def normalizer_even_exp(self: 'MPCEnv', a: np.ndarray) -> tuple:
        n: int = len(a)
        fid: int = 1
        field: int = self.primes[fid]
        add_func = partial(add_mod, field=field)
        mul_func = partial(mul_mod, field=field)

        r, rbits = self.share_random_bits(param.NBIT_K, n, fid)

        # Warning: a + r might overflow in numpy.
        e = zeros(n) if self.pid == 0 else a + r
        e = self.reveal_sym(e, fid=0)

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
        rfir = self.multiply(rbits, Tneg, True, fid)

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

        diff = self.multiply(double_flag, diff, True, fid)
        
        chosen_bit_sum = zeros(n)
        if self.pid != 0:
            chosen_bit_sum = add_func(even_bit_sum, diff)
        
        b_mat: np.ndarray = self.table_lookup(chosen_bit_sum, 1, fid=0)

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
            x: np.ndarray = self.multiply(a, b, True, fid)
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

        c_arr: list = self.mult_mat_parallel(f_arr, b_arr, fid)
        
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
        s, s_sqrt = self.normalizer_even_exp(a)

        a_scaled: np.ndarray = self.multiply(a, s, elem_wise=True, fid=fid)
        a_scaled = self.trunc(a_scaled, param.NBIT_K, param.NBIT_K - param.NBIT_F, fid=fid)

        a_scaled_sq: np.ndarray = self.multiply(a_scaled, a_scaled, elem_wise=True, fid=fid)
        a_scaled_sq = self.trunc(a_scaled_sq, fid=fid)

        scaled_est = zeros(n)
        
        if self.pid != 0:
            scaled_est = add_mod(
                mul_mod(-a_scaled, 4, field), add_mod(a_scaled_sq, a_scaled_sq, field), field)
            if self.pid == 1:
                coeff: int = self.double_to_fp(2.9581, param.NBIT_K, param.NBIT_F, fid=0)
                scaled_est = add_mod(scaled_est, coeff, field)

        # TODO: Make h_and_g a ndarray
        h_and_g: list = [zeros((1, n)) for _ in range(2)]

        h_and_g[0][0][:] = self.multiply(scaled_est, s_sqrt, elem_wise=True, fid=0)
        # Our scaled initial approximation (scaled_est) has bit length <= NBIT_F + 2
        # and s_sqrt is at most NBIT_K/2 bits, so their product is at most NBIT_K/2 +
        # NBIT_F + 2
        h_and_g[0] = self.trunc(
            h_and_g[0], param.NBIT_K // 2 + param.NBIT_F + 2, (param.NBIT_K - param.NBIT_F) // 2 + 1, fid=0)

        h_and_g[1][0][:] = add_mod(h_and_g[0][0], h_and_g[0][0], field)
        h_and_g[1][0][:] = self.multiply(h_and_g[1][0], a, elem_wise=True, fid=0)
        h_and_g[1] = self.trunc(h_and_g[1], k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, fid=0)

        onepointfive: int = self.double_to_fp(1.5, param.NBIT_K, param.NBIT_F, fid=0)

        for _ in range(niter):
            r: np.ndarray = self.multiply(h_and_g[0], h_and_g[1], elem_wise=True, fid=0)
            r = self.trunc(r, k = param.NBIT_K + param.NBIT_F, m = param.NBIT_F, fid=0)
            r = np.mod(-r, field)
            if self.pid == 1:
                r[0][:] = add_mod(r[0], onepointfive, field)

            r_dup: list = [r, r]

            h_and_g: list = self.mult_aux_parallel(h_and_g, r_dup, True, fid=0)
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
        
        xr, xm = self.beaver_partition(x, fid=0)

        xdot: np.ndarray = self.beaver_inner_prod(xr, xm, fid=0)
        xdot = np.array([xdot], dtype=np.int64)
        xdot = self.beaver_reconstruct(xdot, fid=0)
        xdot = self.trunc(xdot)

        xnorm, _ = self.fp_sqrt(xdot)

        x1 = np.array([x[0]], dtype=np.int64)
        x1sign: np.ndarray = self.is_positive(x1)

        x1sign = add_func(x1sign, x1sign)
        if self.pid == 1:
            x1sign[0] = (x1sign[0] - 1) % field

        shift: np.ndarray = self.multiply(xnorm, x1sign, True, fid=0)

        sr, sm = self.beaver_partition(shift[0], fid=0)

        xr_0: np.ndarray = np.expand_dims(xr[0], axis=0)
        xm_0: np.ndarray = np.expand_dims(xm[0], axis=0)
        dot_shift: np.ndarray = self.beaver_mult(xr_0, xm_0, sr, sm, True, fid=0)
        dot_shift = self.beaver_reconstruct(dot_shift, fid=0)
        dot_shift = self.trunc(dot_shift, fid=0)

        vdot = zeros(1)
        if self.pid > 0:
            vdot = mul_func(add_func(xdot, dot_shift), 2)

        _, vnorm_inv = self.fp_sqrt(vdot)

        invr, invm = self.beaver_partition(vnorm_inv[0], fid=0)

        vr = zeros(n)
        if self.pid > 0:
            vr = xr.copy()
            vr[0] = add_func(vr[0], sr)
        vm = xm.copy()
        vm[0] = add_func(vm[0], sm)

        v: np.ndarray = self.beaver_mult(vr, vm, invr, invm, True, fid=0)
        v = self.beaver_reconstruct(v, fid=0)
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

            self.switch_seed(1)
            r_mask: np.ndarray = random_ndarray(base=self.primes[0], shape=n)
            r_bits_mask: np.ndarray = random_ndarray(base=field, shape=(n, nbits))
            self.restore_seed(1)

            r -= r_mask
            r_bits -= r_bits_mask
            r %= self.primes[0]
            r_bits %= field

            self.send_elem(r, 2)
            self.send_elem(r_bits, 2)
        elif self.pid == 2:
            r: Vector = self.receive_vector(0, msg_len=TypeOps.get_vec_len(n), shape=n)
            r_bits: Matrix = self.receive_matrix(0, msg_len=TypeOps.get_mat_len(n, nbits), shape=(n, nbits))
        else:
            self.switch_seed(0)
            r: np.ndarray = random_ndarray(base=self.primes[0], shape=n)
            r_bits: np.ndarray = random_ndarray(base=field, shape=(n, nbits))
            self.restore_seed(0)

        c: np.ndarray = zeros(1)
        if self.pid != 0:
            c = add_mod(add_mod(a, a, self.primes[0]), r, self.primes[0])

        c = self.reveal_sym(c, fid=0)

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
        
        lsb: np.ndarray = self.multiply(c_xor_r, no_overflow, True, fid)
        if self.pid > 0:
            lsb = add_mod(lsb, lsb, field)
            lsb -= add_mod(no_overflow, c_xor_r, field)
            if self.pid == 1:
                lsb = add_mod(lsb, 1, field)
        lsb %= field

        # 0, 1 -> 1, 2
        if self.pid == 1:
            lsb = add_mod(lsb, 1, field)
        
        b_mat: np.ndarray = self.table_lookup(lsb, 0, fid=0)

        return b_mat[0]
    
    def beaver_inner_prod(self: 'MPCEnv', ar: Vector, am: Vector, fid: int) -> int:
        mul_func = partial(mul_mod, field=self.primes[fid])
        add_func = partial(add_mod, field=self.primes[fid])
        
        ab: np.ndarray = None
        if self.pid == 0:
            ab = mul_func(am, am)
        else:
            temp: np.ndarray = mul_func(ar, am)
            ab = add_func(temp, temp)
            
            if self.pid == 1:
                ab = add_func(ab, mul_func(ar, ar))

        return reduce(add_func, ab, 0)
    
    def beaver_inner_prod_pair(
            self: 'MPCEnv', ar: Vector, am: Vector, br: Vector, bm: Vector, fid: int) -> Zp:
        ab = Zp(0, self.primes[fid])
        
        for i in range(len(ar)):
            if self.pid == 0:
                ab += am[i] * bm[i]
            else:
                ab += ar[i] * bm[i]
                ab += br[i] * am[i]
                if self.pid == 1:
                    ab += ar[i] * br[i]

        return ab.set_field(self.primes[fid])

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

        one: int = self.double_to_fp(1, param.NBIT_K, param.NBIT_F, fid=0)

        for i in range(n - 1):
            v = np.expand_dims(self.householder(Ap[0]), axis=0)
            vt = v.T
            
            P: np.ndarray = self.multiply(vt, v, False, fid=0)
            P = self.trunc(P, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            if self.pid > 0:
                P = np.mod(-add_func(P, P), self.primes[0])
                if self.pid == 1:
                    np.fill_diagonal(P, add_func(P.diagonal(), one))
            
            B = zeros((n - i, n - i))
            if i == 0:
                Q = P
                B = self.multiply(Ap, P, False, fid=0)
                B = self.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            else:
                Qsub = zeros((n - i, n))
                if self.pid > 0:
                    for j in range(n - i):
                        Qsub[j] = Q[j + i]

                left: list = [P, Ap]
                right: list = [Qsub, P]

                prod: list = self.mult_mat_parallel(left, right, fid=0)
                # TODO: parallelize Trunc
                prod[0] = self.trunc(prod[0], param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
                prod[1] = self.trunc(prod[1], param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

                if self.pid > 0:
                    for j in range(n - i):
                        Q[j + i] = prod[0][j]
                    B = prod[1]
            
            Ap = zeros((n - i - 1, n - i - 1))
            if self.pid > 0:
                for j in range(n - i):
                    R[i + j][i] = B[j][0]
                if i == n - 2:
                    R[n - 1][n - 1] = B[1][1]

                for j in range(n - i - 1):
                    for k in range(n - i - 1):
                        Ap[j][k] = B[j + 1][k + 1]
            
        return Q, R

    def tridiag(self: 'MPCEnv', A: np.ndarray) -> tuple:
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] > 2

        n: int = A.shape[0]
        one: int = self.double_to_fp(1, param.NBIT_K, param.NBIT_F, fid=0)
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
            x = zeros((n - 1))
            if self.pid > 0:
                x[:n - 1] = Ap[0][1:]

            v = np.expand_dims(self.householder(x), axis=0)
            vt = v.T

            vv: np.ndarray = self.multiply(vt, v, False, fid=0)
            vv = self.trunc(vv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            P = zeros(Ap.shape)
            if self.pid > 0:
                P[1:n, 1:n] = np.mod(-add_func(vv[0:n-1, 0:n-1], vv[0:n-1, 0:n-1]), self.primes[0])
                if self.pid == 1:
                    np.fill_diagonal(P, add_func(P.diagonal(), one))

            # TODO: parallelize? (minor improvement)
            PAp: np.ndarray = self.multiply(P, Ap, False, fid=0)
            PAp = self.trunc(PAp, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            B = self.multiply(PAp, P, False, fid=0)
            B = self.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            Qsub = zeros((n, n - i))
            if self.pid > 0:
                Qsub[:, :n - i] = Q[:, i:n]

            Qsub: np.ndarray = self.multiply(Qsub, P, False, fid=0)
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

            Ap = zeros((n - 1, n - 1))
            if self.pid > 0:
                Ap[:n - 1, :n - 1] = B[1:, 1:]

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

                Ap = self.multiply(Q, R, False, fid=0)
                Ap = self.trunc(Ap, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

                if self.pid > 0:
                    np.fill_diagonal(Ap, add_func(Ap.diagonal(), shift))

                Vsub = zeros((i + 1, n))
                if self.pid > 0:
                    Vsub[:i + 1] = V[:i + 1]

                Vsub = self.multiply(Q, Vsub, False, fid=0)
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

    def orthonormal_basis(self: 'MPCEnv', A: Matrix) -> Matrix:
        assert A.shape[1] >= A.shape[0]

        c: int = A.shape[0]
        n: int = A.shape[1]

        v_list: list = []

        Ap = Matrix(c, n)
        if self.pid != 0:
            Ap = Matrix().from_value(deepcopy(A))

        one: Zp = self.double_to_fp(1, param.NBIT_K, param.NBIT_F, fid=0)

        for i in range(c):
            v = Matrix(1, Ap.shape[1])
            v[0] = self.householder(Ap[0])

            if self.pid == 0:
                v_list.append(Vector([Zp(0, base=self.primes[0]) for _ in range(Ap.shape[1])]))
            else:
                v_list.append(Vector(v[0].value))

            vt = Matrix(Ap.shape[1], 1)
            if self.pid != 0:
                vt = v.transpose(inplace=False)

            Apv = self.mult_mat_parallel([Ap], [vt], fid=0)[0]
            self.trunc(Apv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            B = self.mult_mat_parallel([Apv], [v], fid=0)[0]
            self.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            if self.pid > 0:
                B *= -2
                B += Ap

            Ap = Matrix(B.shape[0] - 1, B.shape[1] - 1)
            if self.pid > 0:
                for j in range(B.shape[0] - 1):
                    for k in range(B.shape[1] - 1):
                        Ap[j][k] = Zp(B[j + 1][k + 1].value, B[j + 1][k + 1].base)

        Q = Matrix(c, n)
        if self.pid > 0:
            if self.pid == 1:
                for i in range(c):
                    Q[i][i] = Zp(one.value, one.base)

        for i in range(c - 1, -1, -1):
            v = Matrix(1, len(v_list[i]))
            if self.pid > 0:
                v[0] = Vector(v_list[i].value)

            vt = Matrix(v.shape[1], 1)
            if self.pid != 0:
                vt = v.transpose(inplace=False)

            Qsub = Matrix(c, n - i)
            if self.pid > 0:
                for j in range(c):
                    for k in range(n - i):
                        Qsub[j][k] = Zp(Q[j][k + i].value, Q[j][k + i].base)

            Qv = self.mult_mat_parallel([Qsub], [vt], fid=0)[0]
            self.trunc(Qv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            Qvv = self.mult_mat_parallel([Qv], [v], fid=0)[0]
            self.trunc(Qvv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            if self.pid > 0:
                Qvv *= -2

            if self.pid > 0:
                for j in range(c):
                    for k in range(n - i):
                        Q[j][k + i] += Qvv[j][k]

        return Q

    def read_matrix(self: 'MPCEnv', f, nrows: int, ncols: int, fid: int) -> Matrix:
        a = list()
        
        for _ in range(nrows):
            a.append(self.read_vector(f, ncols, fid))
        
        return Matrix().from_value(Vector(a))
    
    def read_vector(self: 'MPCEnv', f, n: int, fid: int) -> Matrix:
        a: list = list()
        
        for _ in range(n):
            a.append(Zp(int(f.read(self.primes_bytes[0])), base=self.primes[fid]))
        
        return Vector(a)

    def filter(self: 'MPCEnv', v: Vector, mask: Vector) -> Vector:
        return Vector([e for i, e in enumerate(v.value) if mask[i].value == 1])
    
    def filter_rows(self: 'MPCEnv', mat: Matrix, mask: Vector) -> Matrix:
        return Matrix().from_value(self.filter(mat, mask))
    
    def inner_prod(self: 'MPCEnv', a: Matrix, fid: int) -> Matrix:
        ar, am = self.beaver_partition(a, fid)

        c = Vector([Zp(0, base=param.BASE_P) for _ in range(a.shape[0])])
        for i in range(a.shape[0]):
            c[i] = self.beaver_inner_prod(ar[i], am[i], fid)

        return self.beaver_reconstruct(c, fid)

    def parallel_logistic_regression(
        self: 'MPCEnv', xr: Matrix, xm: Matrix, vr: Matrix,
        vm: Matrix, yr: Vector, ym: Vector, max_iter: int) -> tuple:
        n: int = vr.shape[1]
        p: int = vr.shape[0]
        c: int = xr.shape[0]
        assert vm.shape[0] == p
        assert vm.shape[1] == n
        assert xm.shape[0] == c
        assert xm.shape[1] == n
        assert xr.shape[1] == n
        assert len(yr) == n
        assert len(ym) == n

        b0 = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])
        bv = Matrix(c, p)
        bx = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])

        yneg_r = -yr
        yneg_m = -ym
        if self.pid > 0:
            for i in range(n):
                yneg_r[i] += 1

        yneg = deepcopy(yneg_m)
        if self.pid == 1:
            for i in range(n):
                yneg[i] += yneg_r[i]

        fp_memory: Zp = self.double_to_fp(0.5, param.NBIT_K, param.NBIT_F, fid=0)
        fp_one: Zp = self.double_to_fp(1, param.NBIT_K, param.NBIT_F, fid=0)
        eta: float = 0.3

        step0 = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])
        stepv = Matrix(c, p)
        stepx = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])

        nbatch: int = 10
        batch_size: int = (n + nbatch - 1) // nbatch

        for it in range(max_iter):
            print(f'Logistic regression iteration {it} initialized')
            batch_index: int = it % nbatch
            start_ind: int = batch_size * batch_index
            end_ind: int = start_ind + batch_size
            if end_ind > n:
                end_ind = n
            cur_bsize: int = end_ind - start_ind

            xr_batch = Matrix(c, cur_bsize)
            xm_batch = Matrix(c, cur_bsize)
            vr_batch = Matrix(p, cur_bsize)
            vm_batch = Matrix(p, cur_bsize)
            yn_batch = Vector([Zp(0, base=param.BASE_P) for _ in range(cur_bsize)])
            ynr_batch = Vector([Zp(0, base=param.BASE_P) for _ in range(cur_bsize)])
            ynm_batch = Vector([Zp(0, base=param.BASE_P) for _ in range(cur_bsize)])

            for j in range(c):
                for i in range(cur_bsize):
                    xr_batch[j][i].value = xr[j][start_ind + i].value
                    xm_batch[j][i].value = xm[j][start_ind + i].value

            for j in range(p):
                for i in range(cur_bsize):
                    vr_batch[j][i].value = vr[j][start_ind + i].value
                    vm_batch[j][i].value = vm[j][start_ind + i].value

            for i in range(cur_bsize):
                yn_batch[i].value = yneg[start_ind + i].value
                ynr_batch[i].value = yneg_r[start_ind + i].value
                ynm_batch[i].value = yneg_m[start_ind + i].value

            fp_bsize_inv: Zp = self.double_to_fp(eta * (1 / cur_bsize), param.NBIT_K, param.NBIT_F, fid=0)

            bvr, bvm = self.beaver_partition(bv, fid=0)
            bxr, bxm = self.beaver_partition(bx, fid=0)

            h: Matrix = self.beaver_mult(bvr, bvm, vr_batch, vm_batch, False, fid=0)
            for j in range(c):
                xrvec = xr_batch[j] * fp_one
                xmvec = xm_batch[j] * fp_one
                h[j] += self.beaver_mult_vec(xrvec, xmvec, bxr[j], bxm[j], fid=0)
            h: Matrix = self.beaver_reconstruct(h, fid=0)
            self.trunc(h, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            for j in range(c):
                h[j] += b0[j]

            hvec = Matrix().from_value(h).flatten()
            _, s_grad_vec = self.neg_log_sigmoid(hvec, fid=0)

            s_grad = Matrix().from_value(Vector([s_grad_vec], deep_copy=True))
            s_grad.reshape(c, cur_bsize)

            d0 = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])
            dv = Matrix(c, p)
            dx = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])

            for j in range(c):
                s_grad[j] += yn_batch * fp_one
                d0[j] = sum(s_grad[j], Zp(0, base=param.BASE_P))

            s_grad_r, s_grad_m = self.beaver_partition(s_grad, fid=0)

            for j in range(c):
                dx[j] = self.beaver_inner_prod_pair(
                    xr_batch[j], xm_batch[j], s_grad_r[j], s_grad_m[j], fid=0)
            dx = self.beaver_reconstruct(dx, fid=0)

            vr_batch.transpose(inplace=True)
            vm_batch.transpose(inplace=True)
            dv: Matrix = self.beaver_mult(s_grad_r, s_grad_m, vr_batch, vm_batch, False, fid=0)
            dv: Matrix = self.beaver_reconstruct(dv, fid=0)
            self.trunc(dv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            step0: Vector = step0 * fp_memory - d0 * fp_bsize_inv
            stepv: Matrix = stepv * fp_memory - dv * fp_bsize_inv
            stepx: Vector = stepx * fp_memory - dx * fp_bsize_inv
            self.trunc_vec(step0, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            self.trunc(stepv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            self.trunc_vec(stepx, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            b0: Vector = b0 + step0
            bv: Matrix = Matrix().from_value(bv + stepv)
            bx: Vector = bx + stepx
    
        return b0, bv, bx

    def neg_log_sigmoid(self: 'MPCEnv', a: Vector, fid: int) -> tuple:
        n: int = len(a)
        depth: int = 6
        step: float = 4
        cur: Vector = deepcopy(a)
        a_ind = Vector([Zp(0, base=self.primes[fid]) for _ in range(len(a))])

        for i in range(depth):
            cur_sign: Vector = self.is_positive(cur)
            index_step = Zp(1 << (depth - 1 - i), base=self.primes[fid])

            for j in range(n):
                a_ind[j] += cur_sign[j] * index_step

            cur_sign *= 2
            if self.pid == 1:
                for j in range(n):
                    cur_sign[j] -= 1

            step_fp: Zp = self.double_to_fp(
                step, param.NBIT_K, param.NBIT_F, fid=fid)

            for j in range(n):
                cur[j] -= step_fp * cur_sign[j]

            step //= 2

        if self.pid == 1:
            for j in range(n):
                a_ind[j] += 1

        params: Matrix = self.table_lookup(a_ind, 2, fid=0)

        b: Vector = self.mult_vec(params[1], a, fid=fid)
        self.trunc_vec(b)

        if self.pid > 0:
            for j in range(n):
                b[j] += params[0][j]

        b_grad = deepcopy(params[1])

        return b, b_grad
