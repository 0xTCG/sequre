import sys
import time
import random
import math

from functools import partial
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

    def receive_vector(self: 'MPCEnv', from_pid: int, msg_len: int, shape: tuple) -> Vector:
        received_vec: np.ndarray = zeros(shape)

        for i, elem in enumerate(bytes_to_arr(self.sockets[from_pid].receive(msg_len=msg_len))):
            received_vec[i] = elem

        return received_vec
    
    def receive_matrix(self: 'MPCEnv', from_pid: int, msg_len: int, shape: tuple) -> Matrix:
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
    
    def table_lookup(self: 'MPCEnv', x: Vector, table_id: int, fid: int) -> Matrix:
        return self.evaluate_poly(x, self.lagrange_cache[table_id], fid=fid)
    
    def rand_mat_bits(self: 'MPCEnv', shape: tuple, num_bits: int, fid: int) -> np.ndarray:
        assert num_bits < 64, f'Number of bits too big for numpy: {num_bits}'
        return random_ndarray(((1 << num_bits) - 1), shape=shape) % self.primes[fid]

    def trunc(self: 'MPCEnv', a: np.ndarray, k: int = param.NBIT_K + param.NBIT_F, m: int = param.NBIT_F, fid: int = 0):
        msg_len: int = TypeOps.get_bytes_len(a)
        
        if self.pid == 0:
            r: np.ndarray = self.rand_mat_bits(a.shape, k + param.NBIT_V, fid=fid)
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
    
    def lagrange_interp_simple(self: 'MPCEnv', y: Vector, fid: int) -> Vector:
        n: int = len(y)
        x = list(range(1, n + 1))

        return self.lagrange_interp(x, y, fid)

    def fan_in_or(self: 'MPCEnv', a: Matrix, fid: int) -> Vector:
        n: int = a.shape[0]
        d: int = a.shape[1]
        a_sum = [0] * n

        if self.pid > 0:
            for i in range(n):
                a_sum[i] = int(self.pid == 1)
                for j in range(d):
                    a_sum[i] += int(a[i][j])
        a_sum = Vector(a_sum).set_field(self.primes[fid])

        coeff = Matrix(1, d + 1, t=int)

        key: tuple = (d + 1, fid)
        if key not in self.or_lagrange_cache:
            y = Vector([int(i != 0) for i in range(d + 1)])
            coeff_param = self.lagrange_interp_simple(y, fid) # OR function
            self.or_lagrange_cache[key] = coeff_param

        coeff[0] = deepcopy(self.or_lagrange_cache[key])

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
        # TODO: Make it parallel by having a and b as ndarrays
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

    def mult_mat_parallel(self: 'MPCEnv', a: list, b: list, fid: int) -> Vector:
        mults: list = self.mult_aux_parallel(a, b, False, fid)
        return [Matrix().from_value(mult) for mult in mults]

    def prefix_or(self: 'MPCEnv', a: Matrix, fid: int) -> Matrix:
        n: int = a.shape[0]

        # Find next largest squared integer
        L: int = int(math.ceil(math.sqrt(a.shape[1])))
        L2: int = L * L

        # Zero-pad to L2 bits
        a_padded = Matrix(n, L2)
        
        if self.pid > 0:
            for i in range(n):
                for j in range(L2):
                    if j < L2 - a.shape[1]:
                        a_padded[i][j] = 0
                    else:
                        a_padded[i][j] = a[i][j - L2 + a.shape[1]]

        self.reshape(a_padded, n * L, L)

        x: Vector = self.fan_in_or(a_padded, fid)

        xpre = Matrix(n * L, L)
        
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    xpi: int = L * i + j
                    for k in range(L):
                        xpre[xpi][k] = x[L * i + k] * Zp(int(k <= j), base=self.primes[fid])
        
        y: Vector = self.fan_in_or(xpre, fid)

        f = [Matrix(1, L) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    if j == 0:
                        f[i][0][j] = x[L * i]
                    else:
                        f[i][0][j] = y[L * i + j] - y[L * i + j - 1]
                f[i].set_field(self.primes[fid])

        tmp = [Matrix(L, L) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    tmp[i][j] = Vector(a_padded[L * i + j])
                tmp[i].set_field(self.primes[fid])

        c = self.mult_mat_parallel(f, tmp, fid)  # c is a concatenation of n 1-by-L matrices

        cpre = Matrix(n * L, L).set_field(self.primes[fid])
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    cpi: int = L * i + j
                    for k in range(L):
                        cpre[cpi][k] = c[i][0][k] * Zp(int(k <= j), base=self.primes[fid])

        bdot_vec: Vector = self.fan_in_or(cpre, fid)
        
        bdot = [Matrix(1, L).set_field(self.primes[fid]) for _ in range(n)]
        if self.pid > 0:
            for i in range(n):
                for j in range(L):
                    bdot[i][0][j] = Zp(bdot_vec[L * i + j].value, base=self.primes[fid])

        for i in range(n):
            f[i].reshape(L, 1)

        s = self.mult_mat_parallel(f, bdot, fid)

        b = Matrix(n, a.shape[1]).set_field(self.primes[fid])
        if self.pid > 0:
            for i in range(n):
                for j in range(a.shape[1]):
                    j_pad: int = L2 - a.shape[1] + j

                    il: int = j_pad // L
                    jl: int = j_pad - il * L

                    b[i][j] = s[i][il][jl] + y[L * i + il] - f[i][il][0]
        
        return b
    
    def int_to_fp(self: 'MPCEnv', a: int, k: int, f: int, fid: int) -> Zp:
        sn = 1 if a >= 0 else -1

        az_shift: int = TypeOps.left_shift(a, f)
        az_trunc: int = TypeOps.trunc_elem(az_shift, k - 1)

        return Zp(az_trunc * sn, base=self.primes[fid])

    def fp_div(self: 'MPCEnv', a: Vector, b: Vector, fid: int) -> Vector:
        assert len(a) == len(b)

        n: int = len(a)
        if n > param.DIV_MAX_N:
            nbatch: int = math.ceil(n / param.DIV_MAX_N)
            c = Vector([Zp(0, base=self.primes[fid]) for _ in range(n)])
            for i in range(nbatch):
                start: int = param.DIV_MAX_N * i
                end: int = start + param.DIV_MAX_N
                if end > n:
                    end = n
                batch_size: int = end - start

                a_copy = None
                b_copy = None
                for j in range(batch_size):
                    a_copy.append(Vector(a[start + j], deep_copy=True))
                    b_copy.append(Vector(b[start + j], deep_copy=True))

                c_copy: Vector = self.fp_div(a_copy, b_copy, fid=fid)
                for j in range(batch_size):
                    c[start + j] = Vector(c_copy[j], deep_copy=True)
            return c

        niter: int = 2 * math.ceil(math.log2(param.NBIT_K / 3.5)) + 1

        # Initial approximation: 1 / x_scaled ~= 5.9430 - 10 * x_scaled + 5 * x_scaled^2
        s, _ = self.normalizer_even_exp(b)

        b_scaled: Vector = self.mult_vec(b, s, fid=fid)

        self.trunc_vec(b_scaled, param.NBIT_K, param.NBIT_K - param.NBIT_F, fid=fid)

        b_scaled_sq: Vector = self.mult_vec(b_scaled, b_scaled, fid=fid)
        self.trunc_vec(b_scaled_sq, fid=fid)

        scaled_est = Vector([Zp(0, base=self.primes[fid]) for _ in range(n)])
        if self.pid != 0:
            scaled_est = -b_scaled * Zp(10, base=self.primes[fid]) + b_scaled_sq * Zp(5, base=self.primes[fid])
            if self.pid == 1:
                coeff: Zp = self.double_to_fp(5.9430, param.NBIT_K, param.NBIT_F, fid=fid)
                scaled_est += coeff

        w: Vector = self.mult_vec(scaled_est, s, fid=fid)
        # scaled_est has bit length <= NBIT_F + 2, and s has bit length <= NBIT_K
        # so the bit length of w is at most NBIT_K + NBIT_F + 2
        self.trunc_vec(w, param.NBIT_K + param.NBIT_F + 2, param.NBIT_K - param.NBIT_F, fid=fid)

        x: Vector = self.mult_vec(w, b, fid=fid)
        self.trunc_vec(x, fid=fid)

        one: Zp = self.int_to_fp(1, param.NBIT_K, param.NBIT_F, fid=fid)

        x = -x
        if self.pid == 1:
            for i in range(len(x)):
                x[i] += one
        
        y: Vector = self.mult_vec(a, w, fid=fid)
        self.trunc_vec(y, fid=fid)

        for _ in range(niter):
            xr, xm = self.beaver_partition(x, fid=fid)
            yr, ym = self.beaver_partition(y, fid=fid)

            xpr = deepcopy(xr)
            if self.pid > 0:
                xpr += one

            y: Vector = self.beaver_mult_vec(yr, ym, xpr, xm, fid=0)
            x: Vector = self.beaver_mult_vec(xr, xm, xr, xm, fid=0)

            x: Vector = self.beaver_reconstruct(x, fid=fid)
            y: Vector = self.beaver_reconstruct(y, fid=fid)

            self.trunc_vec(x, fid=fid)
            self.trunc_vec(y, fid=fid)

        if self.pid == 1:
            for i in range(len(x)):
                x[i] += one
            
        c: Vector = self.mult_vec(y, x, fid=fid)
        self.trunc_vec(c, fid=fid)

        return c

    def less_than_bits_public(self: 'MPCEnv', a: Matrix, b_pub: Matrix, fid: int) -> Matrix:
        return self.less_than_bits_aux(a, b_pub, 2, fid)

    def num_to_bits(self: 'MPCEnv', a: Vector, bitlen: int) -> Vector:
        b = Matrix(len(a), bitlen, t=int)
    
        for i in range(len(a)):
            for j in range(bitlen):
                b[i][j] = TypeOps.bit(int(a[i]), bitlen - 1 - j)
    
        return b
    
    def rand_vec_bits(self: 'MPCEnv', n: int, bitlen: int, fid: int) -> Vector:
        am: Matrix = self.rand_mat_bits(1, n, bitlen, fid=fid)
        return am[0]
    
    def share_random_bits(self: 'MPCEnv', k: int, n: int, fid: int) -> tuple:
        if self.pid == 0:
            r: Vector = self.rand_vec_bits(n, k + param.NBIT_V, fid=fid)
            rbits: Matrix = self.num_to_bits(r, k)

            self.switch_seed(1)
            r_mask: Vector = self.rand_vector(n, fid)
            rbits_mask: Matrix = self.rand_mat(n, k, fid).to_int()
            self.restore_seed(1)

            r -= r_mask

            rbits -= rbits_mask
            rbits.set_field(self.primes[fid])
            rbits.to_int()

            self.send_elem(r, 2)
            self.send_elem(Matrix().from_value(rbits), 2)
        elif self.pid == 2:
            r: Vector = self.receive_vector(0, msg_len=TypeOps.get_vec_len(n), fid=fid)
            rbits: Matrix = self.receive_matrix(0, msg_len=TypeOps.get_mat_len(n, k), fid=fid).to_int()
        else:
            self.switch_seed(0)
            r: Vector = self.rand_vector(n, fid)
            rbits: Matrix = self.rand_mat(n, k, fid).to_int()
            self.restore_seed(0)
        
        return r, rbits

    def normalizer_even_exp(self: 'MPCEnv', a: np.ndarray) -> tuple:
        n: int = len(a)
        fid: int = 1
        field: int = self.primes[fid]

        r, rbits = self.share_random_bits(param.NBIT_K, n, fid)

        e = zeros(n) if self.pid == 0 else add_mod(a, r, field)
        e = self.reveal_sym(e, fid=0)

        ebits: np.ndarray = zeros(
            (n, param.NBIT_K)) if self.pid == 0 else self.num_to_bits(e, param.NBIT_K)

        c: np.ndarray = self.less_than_bits_public(rbits, ebits, fid)
        if self.pid > 0:
            c = np.mod(-c, field)
            if self.pid == 1:
                c = add_mod(c, 1, field)
        
        ep: np.ndarray = zeros((n, param.NBIT_K + 1))
        if self.pid > 0:
            for i in range(n):
                ep[i][0] = c[i]
                for j in range(1, param.NBIT_K + 1):
                    temp: np.ndarray = mul_mod(ebits[i][j - 1], 2, field)
                    ep[i][j] = mul_mod(
                        np.mod(-add_mod(temp, -1, field), field), rbits[i][j - 1], field) 
                    if self.pid == 1:
                        ep[i][j] = add_mod(ep[i][j], ebits[i][j - 1], field)

        E: np.ndarray = self.prefix_or(ep, fid)

        tpneg: np.ndarray = zeros((n, param.NBIT_K))
        if self.pid > 0:
            for i in range(n):
                for j in range(param.NBIT_K):
                    tpneg[i][j] = E[i][j] - rbits[i][j] * (1 - ebits[i][j])
        tpneg.set_field(self.primes[fid]).to_int()
        
        Tneg: Matrix = self.prefix_or(tpneg, fid).to_int()
        half_len: int = param.NBIT_K // 2

        efir = Matrix(n, param.NBIT_K, t=int)
        rfir = Matrix(n, param.NBIT_K, t=int)
        if self.pid > 0:
            efir = self.mul_elem(ebits, Tneg)
        efir.set_field(self.primes[fid])
        rfir = self.mult_elem(
            rbits.set_field(self.primes[fid]),
            Tneg.set_field(self.primes[fid]), fid)

        efir = Matrix().from_value(efir).set_field(self.primes[fid])
        rfir = Matrix().from_value(rfir).set_field(self.primes[fid])

        double_flag: Vector = self.less_than_bits(efir, rfir, fid)

        odd_bits = Matrix(n, half_len, t=int)
        even_bits = Matrix(n, half_len, t=int)
        Tneg.to_int()
        if self.pid > 0:
            for i in range(n):
                for j in range(half_len):
                    odd_bits[i][j] = (1 - Tneg[i][2 * j + 1]) if self.pid == 1 else -Tneg[i][2 * j + 1]
                    if ((2 * j + 2) < param.NBIT_K):
                        even_bits[i][j] = (1 - Tneg[i][2 * j + 2]) if self.pid == 1 else -Tneg[i][2 * j + 2]
                    else:
                        even_bits[i][j] = 0
        odd_bits.set_field(self.primes[fid]).to_int()
        even_bits.set_field(self.primes[fid]).to_int()

        odd_bit_sum = Vector([0] * n)
        even_bit_sum = Vector([0] * n)
        for i in range(n):
            for j in range(half_len):
                odd_bit_sum[i] += odd_bits[i][j]
                even_bit_sum[i] += even_bits[i][j]

            if self.pid == 1:
                odd_bit_sum[i] += 1
                even_bit_sum[i] += 1
        odd_bit_sum.set_field(self.primes[fid]).to_int()
        even_bit_sum.set_field(self.primes[fid]).to_int()

        # If double_flag = true, then use odd_bits, otherwise use even_bits

        diff = Vector([0] * n)
        if self.pid != 0:
            diff: Vector = odd_bit_sum - even_bit_sum
            diff.set_field(self.primes[fid]).to_int()

        diff: Vector = self.mult_vec(
            double_flag.set_field(self.primes[fid]),
            diff.set_field(self.primes[fid]), fid).to_int()
        
        chosen_bit_sum = Vector([0] * n)
        if self.pid != 0:
            chosen_bit_sum = even_bit_sum + diff
            chosen_bit_sum.set_field(self.primes[fid])
        
        b_mat: Matrix = self.table_lookup(chosen_bit_sum, 1, fid=0)

        if self.pid > 0:
            b_sqrt: Vector = b_mat[0]
            b: Vector = b_mat[1]
            return b, b_sqrt
        
        return Vector([Zp(0, base=self.primes[fid]) for _ in range(n)]), Vector([Zp(0, base=self.primes[fid]) for _ in range(n)])

    def less_than_bits(self: 'MPCEnv', a: Matrix, b: Matrix, fid: int) -> Vector:
        return self.less_than_bits_aux(a, b, 0, fid)
    
    def less_than_bits_aux(self: 'MPCEnv', a: Matrix, b: Matrix, public_flag: int, fid: int) -> Vector:
        assert a.shape[0] == b.shape[0]
        assert a.shape[1] == b.shape[1]

        n: int = a.shape[0]
        L: int = a.shape[1]

        # Calculate XOR
        x = Matrix(n, L).set_field(self.primes[fid])

        if public_flag == 0:
            x: Matrix = self.mult_elem(a, b, fid)
            if self.pid > 0:
                x = a + b - x * Zp(2, base=self.primes[fid])
                x.set_field(self.primes[fid])
        elif self.pid > 0:
            x = self.mul_elem(a, b)
            x = a + b - x * 2
            if self.pid == 2:
                x -= a if public_flag == 1 else b
            x.set_field(self.primes[fid])
        
        f: Matrix = self.prefix_or(x, fid)

        if self.pid > 0:
            for i in range(n):
                for j in range(L - 1, 0, -1):
                    f[i][j] -= f[i][j - 1]
            f.set_field(self.primes[fid])
        
        f.to_int()

        if public_flag == 2:
            c = Vector([0] * n)
            if self.pid > 0:
                for i in range(n):
                    c[i] = 0
                    for j in range(L):
                        c[i] += f[i][j] * b[i][j]
                c.set_field(self.primes[fid]).to_int()
            
            return c
        
        # TODO: optimize
        f_arr = [Matrix(1, L).set_field(self.primes[fid]) for _ in range(n)]
        b_arr = [Matrix(L, 1).set_field(self.primes[fid]) for _ in range(n)]

        if self.pid > 0:
            for i in range(n):
                f_arr[i][0] = Vector(f[i].set_field(self.primes[fid]))
                for j in range(L):
                    b_arr[i][j][0] = b[i][j]

        c_arr: list = self.mult_mat_parallel(f_arr, b_arr, fid)

        return Vector([int(c_arr[i][0][0]) if self.pid > 0 else 0 for i in range(n)])

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
    
    def householder(self: 'MPCEnv', x: Vector) -> Vector:
        n: int = len(x)

        xr, xm = self.beaver_partition(x, fid=0)

        xdot = Vector([self.beaver_inner_prod(xr, xm, fid=0)])
        xdot = self.beaver_reconstruct(xdot, fid=0)
        self.trunc_vec(xdot)

        xnorm, _ = self.fp_sqrt(xdot)

        x1 = Vector([Zp(x[0].value, base=x[0].base)])

        x1sign: Vector = self.is_positive(x1)

        x1sign *= 2
        if self.pid == 1:
            x1sign[0] -= 1

        shift: Vector = self.mult_vec(xnorm, x1sign, fid=0)

        sr, sm = self.beaver_partition(shift[0], fid=0)

        dot_shift = self.beaver_mult_vec(xr[0], xm[0], sr, sm, fid=0)
        dot_shift = self.beaver_reconstruct(dot_shift, fid=0)
        self.trunc_elem(dot_shift, fid=0)

        vdot = Vector([Zp(0, base=self.primes[0])])
        if self.pid > 0:
            vdot[0] = (xdot[0] + dot_shift) * 2

        _, vnorm_inv = self.fp_sqrt(vdot)

        invr, invm = self.beaver_partition(vnorm_inv[0], fid=0)

        vr = Vector([Zp(0, base=self.primes[0]) for _ in range(n)])
        if self.pid > 0:
            vr = Vector(xr)
            vr[0] += sr
        vm = Vector(xm)
        vm[0] += sm

        v: Vector = self.beaver_mult_vec(vr, vm, invr, invm, fid=0)
        v: Vector = self.beaver_reconstruct(v, fid=0)
        self.trunc_vec(v, fid=0)

        return v
    
    def is_positive(self: 'MPCEnv', a: Vector) -> Vector:
        n: int = len(a)
        nbits: int = self.primes_bits[0]
        fid: int = 2

        r = Vector([Zp(0, base=self.primes[0]) for _ in range(n)])
        r_bits = Matrix(n, nbits, t=int)
        if self.pid == 0:
            r: Vector = self.rand_vector(n, fid=0)
            r_bits: Matrix = self.num_to_bits(r, nbits)

            self.switch_seed(1)
            r_mask: Vector = self.rand_vector(n, fid=0)
            r_bits_mask: Matrix = self.rand_mat(n, nbits, fid=fid).to_int()
            self.restore_seed(1)

            r -= r_mask
            r_bits -= r_bits_mask
            r_bits.set_field(field=self.primes[fid])
            r_bits = Matrix().from_value(r_bits).to_int()

            self.send_elem(r, 2)
            self.send_elem(r_bits, 2)
        elif self.pid == 2:
            r: Vector = self.receive_vector(0, msg_len=TypeOps.get_vec_len(n), fid=0)
            r_bits: Matrix = self.receive_matrix(0, msg_len=TypeOps.get_mat_len(n, nbits), fid=fid).to_int()
        else:
            self.switch_seed(0)
            r: Vector = self.rand_vector(n, fid=0)
            r_bits: Matrix = self.rand_mat(n, nbits, fid=fid).to_int()
            self.restore_seed(0)

        c = Vector([Zp(0, base=self.primes[0])])
        if self.pid != 0:
            c = a * 2 + r

        c = self.reveal_sym(c, fid=0)

        c_bits = Matrix(n, nbits, t=int)
        if self.pid != 0:
            c_bits = self.num_to_bits(c, nbits)

        # Incorrect result if r = 0, which happens with probaility 1 / BASE_P
        no_overflow: Vector = self.less_than_bits_public(r_bits, c_bits, fid=fid)

        c_xor_r = Vector([0] * n)
        if self.pid > 0:
            for i in range(n):
                c_xor_r[i] = r_bits[i][nbits - 1] - 2 * c_bits[i][nbits - 1] * r_bits[i][nbits - 1]
                if self.pid == 1:
                    c_xor_r[i] += c_bits[i][nbits - 1]
            c_xor_r.set_field(self.primes[fid]).to_int()
        
        lsb: Vector = self.mult_vec(c_xor_r, no_overflow, fid).to_int()
        if self.pid > 0:
            lsb *= 2
            for i in range(n):
                lsb[i] -= no_overflow[i] + c_xor_r[i]
                if self.pid == 1:
                    lsb[i] += 1
            lsb.set_field(self.primes[fid]).to_int()

        # 0, 1 -> 1, 2
        if self.pid == 1:
            for i in range(n):
                lsb[i] += 1
        
        lsb.set_field(self.primes[fid])
        b_mat: Matrix = self.table_lookup(lsb, 0, fid=0)

        return b_mat[0]
    
    def beaver_inner_prod(self: 'MPCEnv', ar: Vector, am: Vector, fid: int) -> Zp:
        ab = Zp(0, self.primes[fid])
        
        for i in range(len(ar)):
            if self.pid == 0:
                ab += am[i] * am[i]
            else:
                ab += ar[i] * am[i] * 2
                if self.pid == 1:
                    ab += ar[i] * ar[i]

        return ab.set_field(self.primes[fid])
    
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

    def qr_fact_square(self: 'MPCEnv', A: Matrix) -> Matrix:
        assert A.shape[0] == A.shape[1]

        n: int = A.shape[0]
        R = Matrix(n, n)
        Q = Matrix(n, n)

        Ap = Matrix(n, n)
        if self.pid != 0:
            Ap = deepcopy(A)

        one: Zp = self.double_to_fp(1, param.NBIT_K, param.NBIT_F, fid=0)

        for i in range(n - 1):
            v = Matrix(1, Ap.shape[1])
            v[0] = self.householder(Ap[0])

            vt = Matrix(Ap.shape[1], 1)
            if self.pid != 0:
                vt = Matrix().from_value(v.transpose(inplace=False))
            
            P = self.mult_mat_parallel([vt], [v], fid=0)[0]
            self.trunc(P, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            if self.pid > 0:
                P *= -2
                if self.pid == 1:
                    for j in range(P.shape[1]):
                        P[j][j] += one
            
            B = Matrix(n - i, n - i)
            if i == 0:
                Q = deepcopy(P)
                B = self.mult_mat_parallel([Ap], [P], fid=0)[0]
                self.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            else:
                Qsub = Matrix(n - i, n)
                if self.pid > 0:
                    for j in range(n - i):
                        Qsub[j] = Q[j + i]

                left: list = [P, Ap]
                right: list = [Qsub, P]

                prod: list = self.mult_mat_parallel(left, right, fid=0)
                # TODO: parallelize Trunc
                self.trunc(prod[0], param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
                self.trunc(prod[1], param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

                if self.pid > 0:
                    for j in range(n - i):
                        Q[j + i] = Vector(prod[0][j].value)
                    B = deepcopy(prod[1])
            
            Ap = Matrix(n - i - 1, n - i - 1)
            if self.pid > 0:
                for j in range(n - i):
                    R[i + j][i] = Zp(B[j][0].value, B[j][0].base)
                if i == n - 2:
                    R[n - 1][n - 1] = Zp(B[1][1].value, B[1][1].base)

                for j in range(n - i - 1):
                    for k in range(n - i - 1):
                        Ap[j][k] = Zp(B[j + 1][k + 1].value, B[j + 1][k + 1].base)
            
        return Q, R

    def tridiag(self: 'MPCEnv', A: Matrix) -> tuple:
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] > 2

        n: int = A.shape[0]
        one: Zp = self.double_to_fp(1, param.NBIT_K, param.NBIT_F, fid=0)

        Q = Matrix(n, n)
        T = Matrix(n, n)
        if self.pid > 0:
            if self.pid == 1:
                for i in range(n):
                    Q[i][i] = Zp(one.value, one.base)

        Ap = Matrix(n, n)
        if self.pid != 0:
            Ap = Matrix().from_value(deepcopy(A))

        for i in range(n - 2):
            x = Vector([Zp(0, base=self.primes[0]) for _ in range(Ap.shape[1] - 1)])
            if self.pid > 0:
                for j in range(Ap.shape[1] - 1):
                    x[j] = Zp(Ap[0][j + 1].value, Ap[0][j + 1].base)

            v = Matrix(1, len(x))
            v[0] = self.householder(x)

            vt = Matrix(len(x), 1)
            if self.pid != 0:
                vt = Matrix().from_value(v.transpose(inplace=False))

            vv = self.mult_mat_parallel([vt], [v], fid=0)[0]
            self.trunc(vv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            P = Matrix(Ap.shape[1], Ap.shape[1])
            if self.pid > 0:
                P[0][0] = Zp(one.value, one.base) if self.pid == 1 else Zp(0, one.base)
                for j in range(1, Ap.shape[1]):
                    for k in range(1, Ap.shape[1]):
                        P[j][k] = vv[j - 1][k - 1] * -2
                        if self.pid == 1 and j == k:
                            P[j][k] += one

            # TODO: parallelize? (minor improvement)
            PAp = self.mult_mat_parallel([P], [Ap], fid=0)[0]
            self.trunc(PAp, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            B = self.mult_mat_parallel([PAp], [P], fid=0)[0]
            self.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            Qsub = Matrix(n, n - i)
            if self.pid > 0:
                for j in range(n):
                    for k in range(n - i):
                        Qsub[j][k] = Zp(Q[j][k + i].value, Q[j][k + i].base)

            Qsub = self.mult_mat_parallel([Qsub], [P], fid=0)[0]
            self.trunc(Qsub, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            if self.pid > 0:
                for j in range(n):
                    for k in range(n - i):
                        Q[j][k + i] = Zp(Qsub[j][k].value, Qsub[j][k].base)

            if self.pid > 0:
                T[i][i] = Zp(B[0][0].value, B[0][0].base)
                T[i + 1][i] = Zp(B[1][0].value, B[1][0].base)
                T[i][i + 1] = Zp(B[0][1].value, B[0][1].base)
                if i == n - 3:
                    T[i + 1][i + 1] = Zp(B[1][1].value, B[1][1].base)
                    T[i + 1][i + 2] = Zp(B[1][2].value, B[1][2].base)
                    T[i + 2][i + 1] = Zp(B[2][1].value, B[2][1].base)
                    T[i + 2][i + 2] = Zp(B[2][2].value, B[2][2].base)

            Ap = Matrix(B.shape[0] - 1, B.shape[1] - 1)
            if self.pid > 0:
                for j in range(B.shape[0] - 1):
                    for k in range(B.shape[1] - 1):
                       Ap[j][k] = Zp(B[j + 1][k + 1].value, B[j + 1][k + 1].base)

        return T, Q

    def eigen_decomp(self: 'MPCEnv', A: Matrix) -> tuple:
        assert A.shape[0] == A.shape[1]
        n: int = A.shape[0]

        L = Vector([Zp(0, base=self.primes[0]) for _ in range(n)])

        Ap, Q = self.tridiag(A)

        V = Matrix(n, n)
        if self.pid != 0:
            V = Q.transpose(inplace=False)

        for i in range(n - 1, 0, -1):
            for _ in range(param.ITER_PER_EVAL):
                shift = Zp(Ap[i][i].value, Ap[i][i].base)
                if self.pid > 0:
                    for j in range(Ap.shape[1]):
                        Ap[j][j] -= shift

                Q, R = self.qr_fact_square(Ap)

                Ap = self.mult_mat_parallel([Q], [R], fid=0)[0]
                self.trunc(Ap, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

                if self.pid > 0:
                    for j in range(Ap.shape[1]):
                        Ap[j][j] += shift

                Vsub = Matrix(i + 1, n)
                if self.pid > 0:
                    for j in range(i + 1):
                        Vsub[j] = Vector(V[j].value)

                Vsub = self.mult_mat_parallel([Q], [Vsub], fid=0)[0]
                self.trunc(Vsub, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

                if self.pid > 0:
                    for j in range(i + 1):
                        V[j] = Vector(Vsub[j].value)

            L[i] = Zp(Ap[i][i].value, Ap[i][i].base)
            if i == 1:
                L[0] = Zp(Ap[0][0].value, Ap[0][0].base)

            Ap_copy: Matrix = deepcopy(Ap)
            Ap = Matrix(i, i)
            if self.pid > 0:
                for j in range(i):
                    for k in range(i):
                        Ap[j][k] = Ap_copy[j][k]
        
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
