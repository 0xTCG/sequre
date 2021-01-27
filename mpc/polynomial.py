import numpy as np

from mpc.arithmetic import Arithmetic
from mpc.prg import PRG
from mpc.comms import Comms
from utils.type_ops import TypeOps
from utils.custom_types import zeros, ones, mul_mod, add_mod, matmul_mod
from utils.utils import random_ndarray


class Polynomial:
    def __init__(self: 'Polynomial', pid: int, prg: PRG, comms: Comms, arithmetic: Arithmetic):
        self.pid = pid
        self.arithmetic = arithmetic
        self.prg = prg
        self.comms = comms
        self.pascal_cache: dict = dict()
        
    def lagrange_interp(self: 'Polynomial', x: np.ndarray, y: np.ndarray, field: int) -> np.ndarray:
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

    def powers(self: 'Polynomial', x: np.ndarray, power: int, field: int) -> np.ndarray:
        assert power >= 1, f'Invalid exponent: {power}'

        n: int = len(x)
        b: np.ndarray = zeros((power + 1, n))
        
        if power == 1:
            if self.pid > 0:
                if self.pid == 1:
                    b[0] += ones(n)
                b[1][:] = x
        else:  # power > 1
            x_r, r = self.arithmetic.beaver_partition(x, field)

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
    
    def evaluate_poly(self: 'Polynomial', x: np.ndarray, coeff: np.ndarray, field: int) -> np.ndarray:
        n: int = len(x)
        npoly: int = coeff.shape[0]
        deg: int = coeff.shape[1] - 1

        pows: np.ndarray = self.powers(x, deg, field)

        if self.pid > 0:
            return matmul_mod(coeff, pows, field)
        
        return zeros((npoly, n))
    
    def lagrange_interp_simple(self: 'MPCEnv', y: np.ndarray, field: int) -> np.ndarray:
        n: int = len(y)
        x = np.arange(1, n + 1)

        return self.lagrange_interp(x, y, field)
 