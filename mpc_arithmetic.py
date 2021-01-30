from functools import partial

import numpy as np

import utils.param as param

from mpc.prg import PRG
from mpc.comms import Comms
from utils.custom_types import zeros, add_mod, mul_mod, matmul_mod
from utils.utils import random_ndarray
from utils.type_ops import TypeOps

class MPCArithmetic:
    def __init__(self: 'MPCArithmetic', pid: int, prg: PRG, comms: Comms):
        self.pid = pid
        self.prg = prg
        self.comms = comms

    def add_public(self: 'MPCArithmetic', x: np.ndarray, a: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        if self.pid == 1:
            return add_mod(x, a, field)
        return x
    
    def beaver_mult(
            self: 'MPCArithmetic', x_r: np.ndarray, r_1: np.ndarray,
            y_r: np.ndarray, r_2: np.ndarray, elem_wise: bool, field: int = param.BASE_P) -> np.ndarray:
        mul_func: callable = partial(mul_mod if elem_wise else matmul_mod, field=field)
        
        if self.pid == 0:
            return mul_func(r_1, r_2)

        xy = mul_func(x_r, r_2)
        xy = add_mod(xy, mul_func(r_1, y_r), field)
        if self.pid == 1:
            xy = add_mod(xy, mul_func(x_r, y_r), field)

        return xy

    def beaver_reconstruct(self: 'MPCArithmetic', elem: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
            msg_len: int = TypeOps.get_bytes_len(elem)
            
            if self.pid == 0:
                self.prg.switch_seed(1)
                mask: np.ndarray = random_ndarray(base=field, shape=elem.shape)
                self.prg.restore_seed(1)

                mm: np.ndarray = np.mod(elem - mask, field)
                self.comms.send_elem(mm, 2)
                
                return mm
            else:
                rr: np.ndarray = None
                if self.pid == 1:
                    self.prg.switch_seed(0)
                    rr = random_ndarray(base=field, shape=elem.shape)
                    self.prg.restore_seed(0)
                else:
                    rr = self.comms.receive_ndarray(
                        from_pid=0,
                        msg_len=msg_len,
                        ndim=elem.ndim,
                        shape=elem.shape)
                    
                return add_mod(elem, rr, field)

    def multiply(self: 'MPCArithmetic', a: np.ndarray, b: np.ndarray, elem_wise: bool, field: int = param.BASE_P) -> np.ndarray:
        x_1_r, r_1 = self.beaver_partition(a, field)
        x_2_r, r_2 = self.beaver_partition(b, field)
        
        c = self.beaver_mult(x_1_r, r_1, x_2_r, r_2, elem_wise, field)
        c = self.beaver_reconstruct(c, field)
        
        return c

    
    def beaver_partition(self: 'MPCArithmetic', x: np.ndarray, field: int = param.BASE_P) -> tuple:
        x_: np.ndarray = np.mod(x, field)

        x_r: np.ndarray = zeros(x_.shape)
        r: np.ndarray = zeros(x_.shape)

        if self.pid == 0:
            self.prg.switch_seed(1)
            r_1: np.ndarray = random_ndarray(base=field, shape=x_.shape)
            self.prg.restore_seed(1)

            self.prg.switch_seed(2)
            r_2: np.ndarray = random_ndarray(base=field, shape=x_.shape)
            self.prg.restore_seed(2)

            r: np.ndarray = add_mod(r_1, r_2, field)
        else:
            self.prg.switch_seed(0)
            r: np.ndarray = random_ndarray(base=field, shape=x_.shape)
            self.prg.restore_seed(0)
            
            x_r = (x_ - r) % field
            x_r = self.comms.reveal_sym(x_r, field=field)
        
        return x_r, r
    
    def beaver_partition_bulk(self: 'MPCArithmetic', x: list, field: int = param.BASE_P) -> tuple:
        # TODO: Do this in parallel
        partitions = [self.beaver_partition(e, field) for e in x]
        x_r = [p[0] for p in partitions]
        r = [p[1] for p in partitions]
        return x_r, r
    
    def beaver_reconstruct_bulk(self: 'MPCArithmetic', x: list, field: int = param.BASE_P) -> tuple:
        # TODO: Do this in parallel
        return [self.beaver_reconstruct(e, field) for e in x]

    def mult_aux_parallel(self: 'MPCArithmetic', a: list, b: list, elem_wise: bool, field: int = param.BASE_P) -> list:
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

        ar, am = self.beaver_partition_bulk(a, field)
        br, bm = self.beaver_partition_bulk(b, field)

        c = [self.beaver_mult(ar[k], am[k], br[k], bm[k], elem_wise, field)
             for k in range(nmat)]
        
        return self.beaver_reconstruct_bulk(c, field)

    def mult_mat_parallel(self: 'MPCArithmetic', a: list, b: list, field: int = param.BASE_P) -> list:
        # TODO: Vectorise/parallelize this method
        return self.mult_aux_parallel(a, b, False, field)
    
    def beaver_inner_prod(self: 'MPCEnv', ar: np.ndarray, am: np.ndarray, field: int = param.BASE_P) -> int:
        mul_func = partial(mul_mod, field=field)
        add_func = partial(add_mod, field=field)
        
        ab: np.ndarray = None
        if self.pid == 0:
            ab = mul_func(am, am)
        else:
            temp: np.ndarray = mul_func(ar, am)
            ab = add_func(temp, temp)
            
            if self.pid == 1:
                ab = add_func(ab, mul_func(ar, ar))

        cum_sum: int = 0
        for e in ab: cum_sum = add_func(cum_sum, e)

        return cum_sum
    
    def beaver_inner_prod_pair(
            self: 'MPCEnv', ar: np.ndarray, am: np.ndarray, br: np.ndarray, bm: np.ndarray, field: int = param.BASE_P) -> int:
        # TODO: Test this method. Check if it is redundant.
        ab: int = 0
        
        for i in range(len(ar)):
            # TODO: Do modular arithmetic
            if self.pid == 0:
                ab += am[i] * bm[i]
            else:
                ab += ar[i] * bm[i]
                ab += br[i] * am[i]
                if self.pid == 1:
                    ab += ar[i] * br[i]

        return ab
    
    def inner_prod(self: 'MPCEnv', a: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        # TODO: Test this metod.
        ar, am = self.beaver_partition(a, field)

        c = zeros(a.shape[0])
        for i in range(a.shape[0]):
            c[i] = self.beaver_inner_prod(ar[i], am[i], field)

        return self.beaver_reconstruct(c, field)
