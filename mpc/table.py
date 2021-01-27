import numpy as np

import utils.param as param

from mpc.polynomial import Polynomial
from utils.custom_types import zeros
from utils.type_ops import TypeOps


class Table:
    def __init__(self: 'Table', pid: int, primes: dict, polynomial: Polynomial):
        self.pid = pid
        self.primes = primes
        self.polynomial = polynomial
        
        self.table_cache: dict = dict()
        self.table_type_modular: dict = dict()
        self.lagrange_cache: dict = dict()
        self.table_field_index: dict = dict()
        
        self.__setup_tables()

    def __setup_tables(self: 'MPCEnv'):
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
            with open(param.SIGMOID_APPROX_PATH) as f:
                for i in range(table.shape[1]):
                    intercept, slope = f.readline().split()
                    fp_intercept: int = TypeOps.double_to_fp(
                        float(intercept), param.NBIT_K, param.NBIT_F, field=self.primes[0])
                    fp_slope: int = TypeOps.double_to_fp(float(slope), param.NBIT_K, param.NBIT_F, field=self.primes[0])

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
                    
                    self.lagrange_cache[cid][i][:] = self.polynomial.lagrange_interp(x, y, field=self.primes[0])
            
        # End of Lagrange cache
    
    def table_lookup(self: 'MPCEnv', x: np.ndarray, table_id: int, field: int) -> np.ndarray:
        return self.polynomial.evaluate_poly(x, self.lagrange_cache[table_id], field=field)
 