import sys
import time
import random
import math

from functools import partial
from copy import deepcopy

import param
from c_socket import CSocket
from connect import connect, open_channel
from custom_types import Zp, Vector, Matrix, TypeOps


class MPCEnv:
    def __init__(self: 'MPCEnv'):
        self.sockets: dict = dict()
        self.prg_states: dict = dict()
        self.pid: int = None
        self.pascal_cache: dict = dict()
        self.table_cache: dict = dict()
        self.lagrange_cache: dict = dict()
        self.table_field_index: dict = dict()
        self.primes: dict = {0: param.BASE_P, 1: 31, 2: 17}  # Temp hardcoded. Needs to be calcualted on init.
        self.invpow_cache: dict = dict()
        self.or_lagrange_cache: dict = dict()
    
    def initialize(self: 'MPCEnv', pid: int, pairs: list) -> bool:
        self.pid = pid

        if (not self.setup_channels(pairs)):
            raise ValueError("MPCEnv::Initialize: failed to initialize communication channels")
            
        if (not self.setup_prgs()):
            raise ValueError("MPCEnv::Initialize: failed to initialize PRGs")

        self.setup_tables()

        if self.pid == 0:  # TODO: Fix with new sockets.
            time.sleep(2 ** 4)
        return True

    def setup_tables(self: 'MPCEnv'):
        # Lagrange cache
        # Table 0
        table = Matrix(1, 2, t=int)
        if (self.pid > 0):
            table[0][0] = 1
            table[0][1] = 0

        table.type_ = int  # table_type_ZZ[0] = true;
        self.table_cache[0] = table
        self.table_field_index[0] = 2

        # Table 1
        half_len: int = param.NBIT_K // 2
        table = Matrix(2, half_len + 1, t=int)
        if (self.pid > 0):
            for i in range(0, half_len + 1):
                if (i == 0):
                    table[0][i] = 1
                    table[1][i] = 1
                else:
                    table[0][i] = table[0][i - 1] * 2
                    table[1][i] = table[1][i - 1] * 4

        table.type_ = int  # table_type_ZZ[1] = true;
        self.table_cache[1] = table
        self.table_field_index[1] = 1

        # Table 2: parameters (intercept, slope) for piecewise-linear approximation
        # of negative log-sigmoid function
        table = Matrix(2, 64)
        if (self.pid > 0):
            with open('sigmoid_approx.txt') as f:
                for i in range(0, table.num_cols()):
                    intercept, slope = f.readline().split()
                    fp_intercept: Zp = self.double_to_fp(
                        float(intercept), param.NBIT_K, param.NBIT_F)
                    fp_slope: Zp = self.double_to_fp(float(slope), param.NBIT_K, param.NBIT_F)

                    table[0][i] = fp_intercept
                    table[1][i] = fp_slope

        table.type_ = Zp  # table_type_ZZ[2] = false;
        self.table_cache[2] = table
        self.table_field_index[2] = 0

        for cid in range(0, len(self.table_cache)):
            nrow: int = self.table_cache[cid].num_rows()
            ncol: int = self.table_cache[cid].num_cols()
            index_by_ZZ: bool = self.table_cache[cid].type_ == int
            self.lagrange_cache[cid] = Matrix(nrow, (2 if index_by_ZZ else 1) * ncol)

            if (self.pid > 0):
                for i in range(0, nrow):
                    x = Vector([0] * ncol * (2 if index_by_ZZ else 1))
                    y = Vector([Zp(0)] * ncol * (2 if index_by_ZZ else 1))
                    
                    for j in range(0, ncol):
                        x[j] = j + 1
                        y[j] = self.table_cache[cid][i][j]
                        if (index_by_ZZ):
                            x[j + ncol] = x[j] + int(self.primes[self.table_field_index[cid]])
                            y[j + ncol] = self.table_cache[cid][i][j]
                    
                    self.lagrange_cache[cid][i] = self.lagrange_interp(x, y)
            
        # End of Lagrange cache
    
    def lagrange_interp(self: 'MPCEnv', x: Vector, y: Vector, fid: int = 0) -> Vector:
        n: int = len(y)

        inv_table = dict()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                key: int = abs(x[i] - x[j])
                if key not in inv_table:
                    inv_table[key] = Zp(key).inv(self.primes[fid])
        
        # Initialize numer and denom_inv
        numer = Matrix(n, n).set_field(self.primes[fid])
        denom_inv = Vector([1] * n)
        numer[0] = Vector(y).set_field(self.primes[fid])

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                for k in range(n - 1, -1, -1):
                    numer[k][j] = ((Zp(0, base=self.primes[fid]) if k == 0 else numer[k - 1][j]) - 
                                    Zp(x[i], base=self.primes[fid]) * numer[k][j])
                denom_inv[i] *= (1 if x[i] > x[j] else -1) * int(inv_table[abs(x[i] - x[j])])

        denom_inv.set_field(self.primes[fid])
        
        numer = Vector(
            [sum(row * denom_inv, Zp(0, base=self.primes[fid]))
             for row in numer.value])

        return numer
    
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
        random.seed()
        self.prg_states[self.pid] = random.getstate() 
        
        for other_pid in set(range(3)) - {self.pid}:
            random.seed(hash((min(self.pid, other_pid), max(self.pid, other_pid))))
            self.prg_states[other_pid] = random.getstate()
        
        return True

    def receive_bool(self: 'MPCEnv', from_pid: int) -> bool:
        return self.sockets[from_pid].receive()

    def send_bool(self: 'MPCEnv', flag: bool, to_pid: int):
        self.sockets[to_pid].send(bytes(flag), sys.getsizeof(flag))

    def send_elem(self: 'MPCEnv', elem: Zp, to_pid: int):
        self.sockets[to_pid].send(elem.to_bytes(), sys.getsizeof(elem.value))
    
    def receive_elem(self: 'MPCEnv', from_pid: int, fid: int = 0) -> Zp:
        return Zp(int.from_bytes(self.sockets[from_pid].receive(), 'big'), base=self.primes[fid])
    
    def receive_vector(self: 'MPCEnv', from_pid: int, fid: int = 0) -> Vector:
        values = self.sockets[from_pid].receive().decode("utf-8").split('.')
        return Vector([Zp(int(v), base=self.primes[fid]) for v in values])
    
    def receive_matrix(self: 'MPCEnv', from_pid: int, fid: int = 0) -> Matrix:
        row_values = self.sockets[from_pid].receive().decode("utf-8").split(';')
        matrix: Matrix = Matrix(len(row_values))

        for i, row_value in enumerate(row_values):
            decoded_vector = Vector([Zp(int(e), base=self.primes[fid]) for e in row_value.split('.')])
            matrix[i] = decoded_vector
        
        return matrix

    def clean_up(self: 'MPCEnv'):
        for socket in self.sockets.values():
            socket.close()
  
    def reveal_sym(self: 'MPCEnv', elem: Zp) -> Zp:
        if (self.pid == 0):
            return deepcopy(elem)
        
        receive_func = None
        if isinstance(elem, Zp):
            receive_func = self.receive_elem
        elif isinstance(elem, Matrix):
            receive_func = self.receive_matrix
        elif isinstance(elem, Vector):
            receive_func = self.receive_vector

        received_elem: Zp = None
        if (self.pid == 1):
            self.send_elem(elem, 3 - self.pid)
            received_elem = receive_func(3 - self.pid)
        else:
            received_elem = receive_func(3 - self.pid)
            self.send_elem(elem, 3 - self.pid)
            
        return elem + received_elem
    
    def switch_seed(self: 'MPCEnv', pid: int):
        self.prg_states[self.pid] = random.getstate()
        random.setstate(self.prg_states[pid])
    
    def restore_seed(self: 'MPCEnv', pid: int):
        self.prg_states[pid] = random.getstate()
        random.setstate(self.prg_states[self.pid])

    def rand_elem(self: 'MPCEnv', fid: int = 0) -> Zp:
        return Zp.randzp(self.primes[fid])
    
    def rand_vector(self: 'MPCEnv', size: int) -> Vector:
        return Vector([self.rand_elem() for _ in range(size)])

    def beaver_partition(self: 'MPCEnv', x: object, fid: int = 0) -> tuple:
        x_ = x.to_field(self.primes[fid])
        type_ = type(x)
        rand_func = None
        if isinstance(x, Zp):
            rand_func = self.rand_elem
        elif isinstance(x, Matrix):
            rand_func = partial(Matrix, m=x_.num_rows(), n=x_.num_cols(), randomise=True)
        elif isinstance(x, Vector):
            rand_func = partial(self.rand_vector, size=len(x_))
        x_r = type_()
        r = type_()
        if self.pid == 0:
            self.switch_seed(1)
            r_1: Zp = rand_func()
            self.restore_seed(1)

            self.switch_seed(2)
            r_2: Zp = rand_func()
            self.restore_seed(2)

            r: Zp = r_1 + r_2
            r.set_field(self.primes[fid])
        else:
            self.switch_seed(0)
            r: Zp = rand_func()
            self.restore_seed(0)
            r.set_field(self.primes[fid])
            
            if isinstance(x, Matrix):  # Temp hack. Will be removed in .seq
                x_r = self.reveal_sym(Matrix().from_value(x_ - r))
            else:
                x_r = self.reveal_sym(x_ - r)
        
        x_r.set_field(self.primes[fid])
        r.set_field(self.primes[fid])
        return x_r, r
    
    def mul_elem(self: 'MPCEnv', v_1: Vector, v_2: Vector) -> Vector:
        return v_1 * v_2

    def rand_mat(self: 'MPCEnv', m: int, n: int, fid: int = 0) -> Matrix:
        m = Matrix(m, n, randomise=True)
        m.set_field(self.primes[fid])
        return m

    def get_pascal_matrix(self: 'MPCEnv', pow: int) -> Matrix:
        if pow not in self.pascal_cache:
            pascal_matrix: Matrix = self.calculate_pascal_matrix(pow)
            self.pascal_cache[pow] = pascal_matrix

        return self.pascal_cache[pow]
    
    def calculate_pascal_matrix(self: 'MPCEnv', pow: int) -> Matrix:
        t = Matrix(pow + 1, pow + 1)
        for i in range(pow + 1):
            for j in range(pow + 1):
                if (j > i):
                    t[i][j] = Zp(0)
                elif (j == 0 or j == i):
                    t[i][j] = Zp(1)
                else:
                    t[i][j] = t[i - 1][j - 1] + t[i - 1][j]
        
        return t

    def powers(self: 'MPCEnv', x: Vector, pow: int, fid: int = 0) -> Matrix:
        assert pow >= 1

        n: int = len(x)
        b = Matrix().set_field(self.primes[fid])
        
        if (pow == 1):
            b.set_dims(2, n, base=self.primes[fid])
            if (self.pid > 0):
                if (self.pid == 1):
                    b[0] += Vector([Zp(1, base=self.primes[fid])] * n)
                b[1] = Vector(x)
        else:  # pow > 1
            x_r, r = self.beaver_partition(x, fid)

            if (self.pid == 0):
                r_pow = Matrix(pow - 1, n).set_field(self.primes[fid])
                r_pow[0] = self.mul_elem(r, r)
                
                for p in range(1, r_pow.num_rows()):
                    r_pow[p] = self.mul_elem(r_pow[p - 1], r)
                    r_pow[p].set_field(self.primes[fid])

                self.switch_seed(1)
                r_ = self.rand_mat(pow - 1, n, fid)
                self.restore_seed(1)

                r_pow -= r_
                r_pow.set_field(self.primes[fid])

                self.send_elem(r_pow, 2)

                b.set_dims(pow + 1, n, base=self.primes[fid])
            else:
                r_pow = Matrix()
                if (self.pid == 1):
                    self.switch_seed(0)
                    r_pow = self.rand_mat(pow - 1, n, fid)
                    self.restore_seed(0)
                else:  # pid == 2
                    r_pow = self.receive_matrix(0)

                x_r_pow = Matrix(pow - 1, n).set_field(self.primes[fid])
                x_r_pow[0] = self.mul_elem(x_r, x_r)
                for p in range(1, x_r_pow.num_rows()):
                    x_r_pow[p] = self.mul_elem(x_r_pow[p - 1], x_r)
                    x_r_pow[p].set_field(self.primes[fid])

                pascal_matrix = self.get_pascal_matrix(pow)

                b.set_dims(pow + 1, n, base=self.primes[fid])

                if (self.pid == 1):
                    b[0] += Vector([Zp(1, base=self.primes[fid])] * n)
                b[1] = x * Vector([Zp(1, base=self.primes[fid])] * n)

                for p in range(2, pow + 1):
                    if (self.pid == 1):
                        b[p] = Vector(x_r_pow[p - 2])

                    if (p == 2):
                        b[p] += self.mul_elem(x_r, r) * Vector([pascal_matrix[p][1]] * n)
                    else:
                        b[p] += self.mul_elem(x_r_pow[p - 3], r) * Vector([pascal_matrix[p][1]] * n)

                        for j in range(2, p - 1):
                            b[p] += self.mul_elem(x_r_pow[p - 2 - j], r_pow[j - 2]) * Vector([pascal_matrix[p][j]] * n)
                        
                        b[p] += self.mul_elem(x_r, r_pow[p - 3]) * Vector([pascal_matrix[p][p - 1]] * n)

                    b[p] += r_pow[p - 2]
        b.set_field(self.primes[fid])
        
        return b
    
    def evaluate_poly(self: 'MPCEnv', x: Vector, coeff: Matrix, fid: int = 0) -> Matrix:
        n: int = len(x)
        npoly: int = coeff.num_rows()
        deg: int = coeff.num_cols() - 1

        pows: Matrix = self.powers(x, deg, fid)

        if (self.pid > 0):
            evaluated_poly = coeff.mult(pows)
            evaluated_poly.set_field(self.primes[fid])
            return Matrix().from_value(evaluated_poly)
        
        return Matrix(npoly, n)
    
    def add_public(self: 'MPCEnv', x: object, a: object) -> object:
        if self.pid == 1:
            return x + a
        return x
    
    def beaver_mult_vec(self: 'MPCEnv', ar: Vector, am: Vector, br: Vector,
                        bm: Vector, fid: int) -> Vector:
        ab = Vector([Zp(0, base=self.primes[fid])] * len(am))
        if self.pid == 0:
            ab += self.mul_elem(am, bm)
        else:
            ab += ar * bm
            ab += am * br
            if self.pid == 1:
                ab += ar * br

        return ab.set_field(self.primes[fid])

    def beaver_mult(self: 'MPCEnv', x_r: Matrix, r_1: Matrix,
                    y_r: Matrix, r_2: Matrix, elem_wise: bool, fid: int) -> Matrix:
        x_r_ = x_r.to_field(self.primes[fid])
        r_1_ = r_1.to_field(self.primes[fid])
        y_r_ = y_r.to_field(self.primes[fid])
        r_2_ = r_2.to_field(self.primes[fid])
        xy = Matrix(r_1_.num_rows(), r_2_.num_rows()
                    if isinstance(r_2_.value[0], Zp) or isinstance(r_2_.value[0], int) 
                    else r_2_.num_cols()).set_field(self.primes[fid])
        if self.pid == 0:
            r_1_r_2 = self.mul_elem(r_1_, r_2_) if elem_wise else r_1_.mult(r_2_)
            xy += r_1_r_2
        else:
            if elem_wise:
                for i in range(xy.num_rows()):
                    for j in range(xy.num_cols()):
                        xy[i][j] += x_r_[i][j] * r_2_[i][j]
                        xy[i][j] += r_1_[i][j] * y_r_[i][j]
                    if self.pid == 1:
                        xy[i][j] += x_r_[i][j] * y_r_[i][j]
            else:
                xy += x_r_.mult(r_2_)
                xy += r_1_.mult(y_r_)
                if self.pid == 1:
                    xy += x_r_.mult(y_r_)

        xy.set_field(self.primes[fid])
        return xy

    def beaver_mult_elem(self: 'MPCEnv', x_1_r: Matrix, r_1: Matrix, x_2_r: Matrix, r_2: Matrix, fid: int = 0) -> Matrix:
        return self.beaver_mult(x_1_r, r_1, x_2_r, r_2, True, fid)
    
    def beaver_reconstruct(self: 'MPCEnv', elem: object, fid: int = 0) -> Matrix:
            elem_ = elem.to_field(self.primes[fid])
            receive_func = None
            if isinstance(elem, Zp):
                receive_func = self.receive_elem
            elif isinstance(elem, Matrix):
                receive_func = self.receive_matrix
            elif isinstance(elem, Vector):
                receive_func = self.receive_vector
            
            rand_func = None
            if isinstance(elem, Zp):
                rand_func = self.rand_elem
            elif isinstance(elem, Matrix):
                rand_func = partial(Matrix, m=elem.num_rows(), n=elem.num_cols(), randomise=True)
            elif isinstance(elem, Vector):
                rand_func = partial(self.rand_vector, size=len(elem))

            if self.pid == 0:
                self.switch_seed(1)
                mask = rand_func()
                self.restore_seed(1)
                mask.set_field(self.primes[fid])

                mm = Matrix().from_value(elem_ - mask) if isinstance(elem, Matrix) else elem_ - mask
                mm.set_field(self.primes[fid])
                time.sleep(2)  # TODO: Fix sockets and remove this wait!
                self.send_elem(mm, 2)
                return mm
            else:
                rr = None
                if self.pid == 1:
                    self.switch_seed(0)
                    rr = rand_func()
                    self.restore_seed(0)
                else:
                    rr = receive_func(0)
                    
                rr.set_field(self.primes[fid])

                return (elem_ + rr).set_field(self.primes[fid])
            
    def mult_elem(self: 'MPCEnv', a: Matrix, b: Matrix, fid: int = 0) -> Matrix:
        x_1_r, r_1 = self.beaver_partition(a, fid)
        x_2_r, r_2 = self.beaver_partition(b, fid)
        
        c = self.beaver_mult_elem(x_1_r, r_1, x_2_r, r_2, fid)
        c = self.beaver_reconstruct(c, fid)
        
        return c
    
    def mult_vec(self: 'MPCEnv', a: Vector, b: Vector, fid: int = 0) -> Vector:
        x_1_r, r_1 = self.beaver_partition(a, fid)
        x_2_r, r_2 = self.beaver_partition(b, fid)
        
        c = self.beaver_mult_vec(x_1_r, r_1, x_2_r, r_2, fid)
        c = self.beaver_reconstruct(c, fid)

        return c
    
    def fp_to_double_elem(self: 'MPCEnv', a: Zp, k: int, f: int) -> float:
        mat = Matrix(1, 1)
        mat[0][0] = a
        return self.fp_to_double(mat, k, f)[0][0]
    
    def fp_to_double(self: 'MPCEnv', a: Matrix, k: int, f: int) -> Matrix:
        base = a[0][0].base
        b = Matrix(a.num_rows(), a.num_cols(), t=float)
        twokm1: int = TypeOps.left_shift(1, k - 1)

        for i in range(0, a.num_rows()):
            for j in range(0, a.num_cols()):
                x = int(a[i][j])
                sn = 1
                if x > twokm1:  # negative number
                    x = base - x
                    sn = -1

                x_trunc = TypeOps.trunc_elem(x, k - 1)
                x_int = TypeOps.right_shift(x_trunc, f)

                # TODO: consider better ways of doing this?
                x_frac = 0
                for bi in range(0, f):
                    if TypeOps.bit(x_trunc, bi) > 0:
                        x_frac += 1
                    x_frac /= 2

                b[i][j] = sn * (x_int + x_frac)
        
        return b

    def print_fp_elem(self: 'MPCEnv', elem: Zp) -> float:
        if self.pid == 0:
            return None
        revealed_elem: Zp = self.reveal_sym(elem)
        elem_float: float = self.fp_to_double_elem(revealed_elem, param.NBIT_K, param.NBIT_F)

        if self.pid == 2:
            print(f'{self.pid}: {elem_float}')
        
        return elem_float

    def print_fp(self: 'MPCEnv', mat: Matrix) -> Matrix:
        if self.pid == 0:
            return None
        revealed_mat: Vector = self.reveal_sym(mat)
        mat_float: Matrix = self.fp_to_double(revealed_mat, param.NBIT_K, param.NBIT_F)

        if self.pid == 2:
            print(f'{self.pid}: {mat_float}')
        
        return mat_float

    def double_to_fp(self: 'MPCEnv', x: float, k: int, f: int) -> Zp:
        sn: int = 1
        if (x < 0):
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
        
        return Zp(az_trunc * sn)
    
    def table_lookup(self: 'MPCEnv', x: Vector, table_id: int) -> Matrix:
        return self.evaluate_poly(x, self.lagrange_cache[table_id])
    
    def rand_mat_bits(self: 'MPCEnv', num_rows: int, num_cols: int, num_bits: int) -> Matrix:
        rand_mat = Matrix(num_rows, num_cols)

        for i in range(num_rows):
            for j in range(num_cols):
                rand_mat[i][j] = Zp(Zp.randzp(base=(2 ** num_bits - 1)).value)

        return rand_mat

    def trunc(self: 'MPCEnv', a: Matrix, k: int, m: int):
        r = Matrix()
        r_low = Matrix()
        if (self.pid == 0):
            r = self.rand_mat_bits(a.num_rows(), a.num_cols(), k + param.NBIT_V)
            r_low.set_dims(a.num_rows(), a.num_cols())
            
            for i in range(0, a.num_rows()):
                for j in range(0, a.num_cols()):
                    r_low[i][j] = Zp(int(r[i][j]) & (2 ** m - 1))
                    # r_low[i][j] = conv<ZZ_p>(trunc_ZZ(rep(r[i][j]), m));

            self.switch_seed(1)
            r_mask = self.rand_mat(a.num_rows(), a.num_cols())
            r_low_mask = self.rand_mat(a.num_rows(), a.num_cols())
            self.restore_seed(1)

            r -= r_mask
            r_low -= r_low_mask

            time.sleep(3)  # TODO: Fix with new sockets.
            self.send_elem(Matrix().from_value(r), 2)
            self.send_elem(Matrix().from_value(r_low), 2)
        elif self.pid == 2:
            r = self.receive_matrix(0)
            r_low = self.receive_matrix(0)
        else:
            self.switch_seed(0)
            r = self.rand_mat(a.num_rows(), a.num_cols())
            r_low = self.rand_mat(a.num_rows(), a.num_cols())
            self.restore_seed(0)

        c = Matrix().from_value(a + r) if self.pid > 0 else Matrix(a.num_rows(), a.num_cols())
        c = self.reveal_sym(c)

        c_low = Matrix(a.num_rows(), a.num_cols())
        if (self.pid > 0):
            for i in range(0, a.num_rows()):
                for j in range(0, a.num_cols()):
                    c_low[i][j] = Zp(int(c[i][j]) & ((1 << m) - 1))

        if (self.pid > 0):
            a += r_low
            if (self.pid == 1):
                a -= c_low

            if m not in self.invpow_cache:
                twoinv = Zp(2).inv()
                twoinvm = twoinv ** m
                self.invpow_cache[m] = twoinvm
                
            a *= self.invpow_cache[m]
    
    def lagrange_interp_simple(self: 'MPCEnv', y: Vector, fid: int) -> Vector:
        n: int = len(y)
        x = Vector(list(range(1, n + 1)))

        return self.lagrange_interp(x, y, fid)

    def fan_in_or(self: 'MPCEnv', a: Matrix, fid: int) -> Vector:
        n: int = a.num_rows()
        d: int = a.num_cols()
        a_sum = Vector([0] * n)

        if self.pid > 0:
            for i in range(n):
                a_sum[i] = int(self.pid == 1)
                for j in range(d):
                    a_sum[i] += int(a[i][j])
        a_sum.set_field(self.primes[fid])

        coeff = Matrix(1, d + 1, t=int)

        key: tuple = (d + 1, fid)
        if key not in self.or_lagrange_cache:
            y = Vector([int(i != 0) for i in range(d + 1)])
            coeff_param = self.lagrange_interp_simple(y.set_field(self.primes[fid]), fid) # OR function
            self.or_lagrange_cache[key] = coeff_param

        coeff[0] = deepcopy(self.or_lagrange_cache[key])

        bmat = self.evaluate_poly(a_sum, coeff, fid)
        return bmat[0]
    
    def reshape(self: 'MPCEnv', a: Matrix, nrows: int, ncols: int):
        if self.pid == 0:
            assert a.num_rows() * a.num_cols() == nrows * ncols
            a.set_dims(nrows, ncols)
        else:
            a.reshape(nrows, ncols)

    def beaver_partition_bulk(self: 'Matrix', x: list, fid: int = 0) -> tuple:
        # TODO: Do this in parallel
        partitions = [self.beaver_partition(e, fid) for e in x]
        x_r = [p[0] for p in partitions]
        r = [p[1] for p in partitions]
        return x_r, r
    
    def beaver_reconstruct_bulk(self: 'Matrix', x: list, fid: int = 0) -> tuple:
        # TODO: Do this in parallel
        return [self.beaver_reconstruct(e, fid) for e in x]

    def mult_aux_parallel(self: 'MPCEnv', a: list, b: list, elem_wise: bool, fid: int = 0) -> list:
        assert len(a) == len(b)
        nmat: int = len(a)

        out_rows = Vector([0] * nmat)
        out_cols = Vector([0] * nmat)

        for k in range(nmat):
            if elem_wise:
                assert (a[k].num_rows() == b[k].num_rows() and
                        a[k].num_cols() == b[k].num_cols())
            else:
                assert a[k].num_cols() == b[k].num_rows()

            out_rows[k] = a[k].num_rows()
            out_cols[k] = a[k].num_cols() if elem_wise else b[k].num_cols()


        ar, am = self.beaver_partition_bulk(a, fid)
        br, bm = self.beaver_partition_bulk(b, fid)

        c = [self.beaver_mult(ar[k], am[k], br[k], bm[k], elem_wise, fid)
             for k in range(nmat)]
        
        return self.beaver_reconstruct_bulk(c, fid)

    def mult_mat_parallel(self: 'MPCEnv', a: list, b: list, fid: int = 0) -> Vector:
        return self.mult_aux_parallel(a, b, False, fid)

    def prefix_or(self: 'MPCEnv', a: Matrix, fid: int) -> Matrix:
        n: int = a.num_rows()

        # Find next largest squared integer
        L: int = int(math.ceil(math.sqrt(a.num_cols())))
        L2: int = L * L

        # Zero-pad to L2 bits
        a_padded = Matrix(n, L2)
        
        if self.pid > 0:
            for i in range(n):
                for j in range(L2):
                    if j < L2 - a.num_cols():
                        a_padded[i][j] = 0
                    else:
                        a_padded[i][j] = a[i][j - L2 + a.num_cols()]

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

        b = Matrix(n, a.num_cols()).set_field(self.primes[fid])
        if self.pid > 0:
            for i in range(n):
                for j in range(a.num_cols()):
                    j_pad: int = L2 - a.num_cols() + j

                    il: int = j_pad // L
                    jl: int = j_pad - il * L

                    b[i][j] = s[i][il][jl] + y[L * i + il] - f[i][il][0]
        
        return b
    
    def int_to_fp(self: 'MPCEnv', a: int, k: int, f: int) -> Zp:
        sn = 1 if a >= 0 else -1

        az_shift: int = TypeOps.left_shift(a, f)
        az_trunc: int = TypeOps.trunc_elem(az_shift, k - 1)

        return Zp(az_trunc * sn)

    def fp_div(self: 'MPCEnv', a: Vector, b: Vector) -> Vector:
        assert len(a) == len(b)

        n: int = len(a)
        if n > param.DIV_MAX_N:
            nbatch: int = math.ceil(n / param.DIV_MAX_N)
            c = Vector([Zp(0)] * n)
            for i in range(nbatch):
                start: int = param.DIV_MAX_N * i
                end: int = start + param.DIV_MAX_N
                if end > n:
                    end = n
                batch_size: int = end - start

                a_copy = Vector([Zp(0)] * batch_size)
                b_copy = Vector([Zp(0)] * batch_size)
                for j in range(batch_size):
                    a_copy[j] = Vector(a[start + j])
                    b_copy[j] = Vector(b[start + j])

                c_copy: Vector = self.fp_div(a_copy, b_copy)
                for j in range(batch_size):
                    c[start + j] = Vector(c_copy[j])
            return c

        niter: int = 2 * math.ceil(math.log2(param.NBIT_K / 3.5)) + 1

        # Initial approximation: 1 / x_scaled ~= 5.9430 - 10 * x_scaled + 5 * x_scaled^2
        s, _ = self.normalizer_even_exp(b)
        # if self.pid == 0: print(f'FINAL S at {self.pid}', s)
        # if self.pid == 0: print(f'FINAL A at {self.pid}', a)
        # if self.pid == 0: print(f'FINAL B at {self.pid}', b)
        
        # raise Exception()
        # s = Vector([Zp(0)] * 2)
        # if (self.pid == 1):
        #     s[0] = Zp(159482573418184356372497469311518688338292085995)
        #     s[1] = Zp(581503681566990113299555434737691398678538636139)
        # if (self.pid == 2):
        #     s[0] = Zp(1302019063912718561831187363405052561693792168678)
        #     s[1] = Zp(879997955763912804904129397978592746877300749414)
        # if self.pid == 2: print(f'FINAL A at {self.pid}', a)
        # if self.pid == 2: print(f'FINAL B at {self.pid}', b)
        print(f'FINAL S at {self.pid}', s)

        b_scaled: Vector = self.mult_vec(b, s)

        self.trunc_vec(b_scaled, param.NBIT_K, param.NBIT_K - param.NBIT_F)

        b_scaled_sq: Vector = self.mult_vec(b_scaled, b_scaled)
        self.trunc_vec(b_scaled_sq)

        scaled_est = Vector([Zp(0)] * n)
        if self.pid != 0:
            scaled_est = -b_scaled * Zp(10) + b_scaled_sq * Zp(5)
            if self.pid == 1:
                coeff: Zp = self.double_to_fp(5.9430, param.NBIT_K, param.NBIT_F)
                scaled_est += coeff

        w: Vector = self.mult_vec(scaled_est, s)
        # scaled_est has bit length <= NBIT_F + 2, and s has bit length <= NBIT_K
        # so the bit length of w is at most NBIT_K + NBIT_F + 2
        self.trunc_vec(w, param.NBIT_K + param.NBIT_F + 2, param.NBIT_K - param.NBIT_F)

        x: Vector = self.mult_vec(w, b)
        self.trunc_vec(x)

        one: Zp = self.int_to_fp(1, param.NBIT_K, param.NBIT_F)

        x = -x
        if self.pid == 1:
            for i in range(len(x)):
                x[i] += one
        
        y: Vector = self.mult_vec(a, w)
        self.trunc_vec(y)

        print('FINAL X prefin at', self.pid, x)
        print('FINAL Y prefin at', self.pid, y)

        for _ in range(niter):
            xr, xm = self.beaver_partition(x)
            yr, ym = self.beaver_partition(y)

            xpr = deepcopy(xr)
            if self.pid > 0:
                xpr += one

            y: Vector = self.beaver_mult_vec(yr, ym, xpr, xm, fid=0)
            x: Vector = self.beaver_mult_vec(xr, xm, xr, xm, fid=0)

            x: Vector = self.beaver_reconstruct(x)
            y: Vector = self.beaver_reconstruct(y)

            self.trunc_vec(x)
            self.trunc_vec(y)

        print('FINAL X fin at', self.pid, x)
        print('FINAL Y fin at', self.pid, y)

        if self.pid == 1:
            for i in range(len(x)):
                x[i] += one
            
        print('FINAL X fin alty at', self.pid, x)

        c: Vector = self.mult_vec(y, x)
        self.trunc_vec(c)

        print('FINAL C fin at', self.pid, c)

        return c

    def trunc_vec(self: 'MPCEnv', v: Vector, k: int = param.NBIT_K + param.NBIT_F, m: int = param.NBIT_F):
        am = Matrix(1, len(v))
        am[0] = v
        self.trunc(am, k, m)
        v.value = am[0].value
    
    def less_than_bits_public(self: 'MPCEnv', a: Matrix, b_pub: Matrix, fid: int) -> Matrix:
        return self.less_than_bits_aux(a, b_pub, 2, fid)

    def num_to_bits(self: 'MPCEnv', a: Vector, bitlen: int) -> Vector:
        b = Matrix(len(a), bitlen)
    
        for i in range(len(a)):
            for j in range(bitlen):
                b[i][j] = TypeOps.bit(int(a[i]), bitlen - 1 - j)
    
        return b
    
    def rand_vec_bits(self: 'MPCEnv', n: int, bitlen: int) -> Vector:
        am: Matrix = self.rand_mat_bits(1, n, bitlen)
        return am[0]
    
    def share_random_bits(self: 'MPCEnv', k: int, n: int, fid: int) -> tuple:
        if self.pid == 0:
            r: Vector = self.rand_vec_bits(n, k + param.NBIT_V)
            rbits: Matrix = self.num_to_bits(r, k)

            self.switch_seed(1)
            r_mask: Vector = self.rand_vector(n)
            rbits_mask: Matrix = self.rand_mat(n, k, fid).to_int()
            self.restore_seed(1)

            r -= r_mask

            rbits -= rbits_mask
            rbits.set_field(self.primes[fid])
            rbits.to_int()

            self.send_elem(r, 2)
            self.send_elem(rbits, 2)
        elif self.pid == 2:
            r: Vector = self.receive_vector(0)
            rbits: Matrix = self.receive_matrix(0).to_int()
        else:
            self.switch_seed(0)
            r: Vector = self.rand_vector(n)
            rbits: Matrix = self.rand_mat(n, k, fid).to_int()
            self.restore_seed(0)
        
        return r, rbits

    def normalizer_even_exp(self: 'MPCEnv', a: Vector) -> tuple:
        n: int = len(a)
        fid: int = 1

        r, rbits = self.share_random_bits(param.NBIT_K, n, fid)

        e = Vector([Zp(0)] * n) if self.pid == 0 else a + r
        e = self.reveal_sym(e)

        ebits = Matrix(n, param.NBIT_K, t=int) if self.pid == 0 else self.num_to_bits(e, param.NBIT_K)

        c: Matrix = self.less_than_bits_public(rbits, ebits, fid)
        if self.pid > 0:
            c = -c
            if self.pid == 1:
                for i in range(n):
                    c[i] += 1
            c.set_field(self.primes[fid])
        
        ep = Matrix(n, param.NBIT_K + 1).set_field(self.primes[fid])
        if self.pid > 0:
            for i in range(n):
                ep[i][0] = Zp(c[i].value, base=self.primes[fid])
                for j in range(1, param.NBIT_K + 1):
                    ep[i][j] = Zp(
                        (1 - 2 * ebits[i][j - 1]) * rbits[i][j - 1],
                        base=self.primes[fid])
                    if self.pid == 1:
                        ep[i][j] += Zp(ebits[i][j - 1], base=self.primes[fid])

        E: Matrix = self.prefix_or(ep, fid).to_int()

        tpneg = Matrix(n, param.NBIT_K, t=int)
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

        print('SEMI efir at ', self.pid, efir)
        print('SEMI rfir at ', self.pid, rfir)

        efir = Matrix().from_value(efir).set_field(self.primes[fid])
        rfir = Matrix().from_value(rfir).set_field(self.primes[fid])

        double_flag: Vector = self.less_than_bits(efir, rfir, fid)

        print('SEMI double_flag at ', self.pid, double_flag)

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

        print('SEMI even_bits at ', self.pid, odd_bits)

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

        print('SEMI odd_bit_sum at ', self.pid, odd_bit_sum)
        print('SEMI even_bit_sum at ', self.pid, even_bit_sum)

        # If double_flag = true, then use odd_bits, otherwise use even_bits

        diff = Vector([0] * n)
        if self.pid != 0:
            diff: Vector = odd_bit_sum - even_bit_sum
            diff.set_field(self.primes[fid]).to_int()

        diff: Vector = self.mult_vec(
            double_flag.set_field(self.primes[fid]),
            diff.set_field(self.primes[fid]), fid).to_int()
        
        print('SEMI diff at ', self.pid, diff)

        chosen_bit_sum = Vector([0] * n)
        if self.pid != 0:
            chosen_bit_sum = even_bit_sum + diff
            chosen_bit_sum.set_field(self.primes[fid])
        
        print('SEMI chosen_bit_sum at ', self.pid, chosen_bit_sum)

        b_mat: Matrix = self.table_lookup(chosen_bit_sum, 1)

        print('SEMI b_mat at ', self.pid, b_mat)

        if self.pid > 0:
            b_sqrt: Vector = b_mat[0]
            b: Vector = b_mat[1]
            return b, b_sqrt
        
        return Vector([Zp(0)] * n), Vector([Zp(0)] * n)

    def less_than_bits(self: 'MPCEnv', a: Matrix, b: Matrix, fid: int) -> Vector:
        return self.less_than_bits_aux(a, b, 0, fid)
    
    def less_than_bits_aux(self: 'MPCEnv', a: Matrix, b: Matrix, public_flag: int, fid: int) -> Vector:
        assert a.num_rows() == b.num_rows()
        assert a.num_cols() == b.num_cols()

        n: int = a.num_rows()
        L: int = a.num_cols()

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

        # print('LessThankBitsAux f at ', self.pid, f)

        if self.pid > 0:
            for i in range(n):
                for j in range(L - 1, 0, -1):
                    f[i][j] -= f[i][j - 1]
            f.set_field(self.primes[fid])
        
        f.to_int()

        # print('LessThankBitsAux f alt at ', self.pid, f)
        
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
    