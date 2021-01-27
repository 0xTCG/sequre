import numpy as np

import utils.param as param

from mpc.prg import PRG
from mpc.comms import Comms
from mpc.arithmetic import Arithmetic
from utils.custom_types import add_mod, mul_mod, matmul_mod, zeros, ones
from utils.type_ops import TypeOps
from utils.utils import random_ndarray


class Polynomial:
    def __init__(self: 'Polynomial', pid: int, primes: dict, prg: PRG, comms: Comms, arithmetic: Arithmetic):
        self.pid = pid
        self.primes = primes
        self.prg = prg
        self.comms = comms
        self.arithmetic = arithmetic

        self.pascal_cache: dict = dict()
        self.table_cache: dict = dict()
        self.table_type_modular: dict = dict()
        self.table_field_index: dict = dict()
        self.lagrange_cache: dict = dict()

        self.__setup_tables()
    
    def lagrange_interp_simple(self: 'Polynomial', y: np.ndarray, field: int) -> np.ndarray:
        n: int = len(y)
        x = np.arange(1, n + 1)

        return self.lagrange_interp(x, y, field)

    def table_lookup(self: 'Polynomial', x: np.ndarray, table_id: int, field: int = param.BASE_P) -> np.ndarray:
        return self.evaluate_poly(x, self.lagrange_cache[table_id], field=field)
    
    def lagrange_interp(self: 'Polynomial', x: np.ndarray, y: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        n: int = len(y)

        inv_table = dict()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                key: int = abs(x[i] - x[j])
                if key not in inv_table:
                    inv_table[key] = TypeOps.mod_inv(key, field)
        
        # Initialize numer and denom_inv
        numer = zeros((n, n))
        denom_inv = ones(n)
        numer[0][:] = np.mod(y, field)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                for k in range(n - 1, -1, -1):
                    numer[k][j] = ((0 if k == 0 else int(numer[k - 1][j])) - int(numer[k][j]) * int(x[i])) % field
                denom_inv[i] = (int(denom_inv[i]) * (1 if x[i] > x[j] else -1) * int(inv_table[abs(x[i] - x[j])])) % field

        numer_dot = mul_mod(numer, denom_inv, field)
        numer_sum = zeros(n)

        for i in range(n):
            for e in numer_dot[i]:
                numer_sum[i] = add_mod(numer_sum[i], e, field)

        return numer_sum
 
    def get_pascal_matrix(self: 'Polynomial', power: int) -> np.ndarray:
        if power not in self.pascal_cache:
            pascal_matrix: np.ndarray = self.calculate_pascal_matrix(power)
            self.pascal_cache[power] = pascal_matrix

        return self.pascal_cache[power]
    
    def calculate_pascal_matrix(self: 'Polynomial', pow: int) -> np.ndarray:
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

    def powers(self: 'Polynomial', x: np.ndarray, power: int, field: int = param.BASE_P) -> np.ndarray:
        assert power >= 1, f'Invalid exponent: {power}'

        n: int = len(x)
        b: np.ndarray = zeros((power + 1, n))
        
        if power == 1:
            if self.pid > 0:
                if self.pid == 1:
                    b[0] += ones(n)
                b[1][:] = x
        else:  # power > 1
            x_r, r = self.arithmetic.beaver_partition(x, field=field)

            if self.pid == 0:
                r_pow: np.ndarray = zeros((power - 1, n))
                r_pow[0][:] = mul_mod(r, r, field)
                
                for p in range(1, r_pow.shape[0]):
                    r_pow[p][:] = mul_mod(r_pow[p - 1], r, field)

                self.prg.switch_seed(1)
                r_: np.ndarray = random_ndarray(base=field, shape=(power - 1, n))
                self.prg.restore_seed(1)

                r_pow = (r_pow - r_) % field
                self.comms.send_elem(r_pow, 2)
            else:
                r_pow: np.ndarray = None
                if self.pid == 1:
                    self.prg.switch_seed(0)
                    r_pow = random_ndarray(base=field, shape=(power - 1, n))
                    self.prg.restore_seed(0)
                else:
                    r_pow = self.comms.receive_matrix(
                        0, msg_len=TypeOps.get_mat_len(power - 1, n),
                        shape=(power - 1, n))

                x_r_pow: np.ndarray = zeros((power - 1, n))
                x_r_pow[0][:] = mul_mod(x_r, x_r, field)
                
                for p in range(1, x_r_pow.shape[0]):
                    x_r_pow[p][:] = mul_mod(x_r_pow[p - 1], x_r, field)

                pascal_matrix: np.ndarray = self.get_pascal_matrix(power)

                if self.pid == 1:
                    b[0][:] = add_mod(b[0], ones(n), field)
                b[1][:] = x

                for p in range(2, power + 1):
                    if self.pid == 1:
                        b[p][:] = x_r_pow[p - 2]

                    if p == 2:
                        b[p] = add_mod(
                            b[p],
                            mul_mod(mul_mod(x_r, r, field), pascal_matrix[p][1], field),
                            field)
                    else:
                        b[p] = add_mod(
                            b[p],
                            mul_mod(mul_mod(x_r_pow[p - 3], r, field), pascal_matrix[p][1], field),
                            field)

                        for j in range(2, p - 1):
                            b[p] = add_mod(
                                b[p],
                                mul_mod(mul_mod(x_r_pow[p - 2 - j], r_pow[j - 2], field), pascal_matrix[p][j], field),
                                field)
                        
                        b[p] = add_mod(
                            b[p],
                            mul_mod(mul_mod(x_r, r_pow[p - 3], field), pascal_matrix[p][p - 1], field),
                            field)

                    b[p] = add_mod(b[p], r_pow[p - 2], field)
        
        return b
    
    def evaluate_poly(self: 'Polynomial', x: np.ndarray, coeff: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        n: int = len(x)
        npoly: int = coeff.shape[0]
        deg: int = coeff.shape[1] - 1

        pows: np.ndarray = self.powers(x, deg, field)

        if self.pid > 0:
            return matmul_mod(coeff, pows, field)
        
        return zeros((npoly, n))
    
    def __setup_tables(self: 'Polynomial'):
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
            with open(param.SIGMOID_APPROX_PATH) as f:
                for i in range(table.shape[1]):
                    intercept, slope = f.readline().split()
                    fp_intercept: int = TypeOps.double_to_fp(
                        float(intercept), param.NBIT_K, param.NBIT_F)
                    fp_slope: int = TypeOps.double_to_fp(float(slope), param.NBIT_K, param.NBIT_F)

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
                    
                    self.lagrange_cache[cid][i][:] = self.lagrange_interp(x, y)
