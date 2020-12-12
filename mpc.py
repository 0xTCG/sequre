import sys
import time
import random

from functools import partial

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
        self.primes: dict = {0: param.BASE_P, 1: 31, 2: 17}
        self.invpow_cache: dict = dict()
    
    def initialize(self: 'MPCEnv', pid: int, pairs: list) -> bool:
        self.pid = pid

        if (not self.setup_channels(pairs)):
            raise ValueError("MPCEnv::Initialize: failed to initialize communication channels")
            
        if (not self.setup_prgs()):
            raise ValueError("MPCEnv::Initialize: failed to initialize PRGs")

        # Lagrange cache
        # Table 0
        table = Matrix(1, 2)
        if (self.pid > 0):
            table[0][0] = Zp(1)
            table[0][1] = Zp(0)

        table.type_ = int  # table_type_ZZ[0] = true;
        self.table_cache[0] = table
        self.table_field_index[0] = 2

        # Table 1
        half_len: int = param.NBIT_K // 2
        table = Matrix(2, half_len + 1)
        if (self.pid > 0):
            for i in range(0, half_len + 1):
                if (i == 0):
                    table[0][i] = Zp(1)
                    table[1][i] = Zp(1)
                else:
                    table[0][i] = table[0][i - 1] * Zp(2)
                    table[1][i] = table[1][i - 1] * Zp(4)

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

        if self.pid == 0:  # TODO: Fix with new sockets.
            time.sleep(8)
        return True
    
    def lagrange_interp(self: 'MPCEnv', x: Vector, y: Vector, fid: int = 0) -> Vector:
        n: int = len(y)

        inv_table = dict()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                key: int = abs(x[i] - x[j])
                if key not in inv_table:
                    inv_table[key] = Zp(key).inv()  # fid used here

        # Initialize numer and denom_inv
        numer = Matrix(n, n)
        denom_inv = Vector([1] * n)
        numer[0] = y * Zp(1)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                for k in range(n - 1, -1, -1):
                    numer[k][j] = (Zp(0) if k == 0 else numer[k - 1][j]) - Zp(x[i]) * numer[k][j]
                    # Mod(numer[k][j], fid);
                denom_inv[i] *= (1 if x[i] > x[j] else -1) * int(inv_table[abs(x[i] - x[j])])
                # Mod(denom_inv[i], fid);

        denom_inv = Vector([Zp(e) for e in denom_inv])
        return numer.mult(denom_inv)
        # Mod(c, fid);
    
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
    
    def receive_elem(self: 'MPCEnv', from_pid: int) -> Zp:
        return Zp(int.from_bytes(self.sockets[from_pid].receive(), 'big'))
    
    def receive_vector(self: 'MPCEnv', from_pid: int) -> Vector:
        values = self.sockets[from_pid].receive().decode("utf-8").split('.')
        return Vector([Zp(int(v)) for v in values])
    
    def receive_matrix(self: 'MPCEnv', from_pid: int) -> Matrix:
        # print(f'Gotta receive from ', from_pid)
        row_values = self.sockets[from_pid].receive().decode("utf-8").split(';')
        # print(f'Received row values from ', from_pid, row_values)
        matrix: Matrix = Matrix(len(row_values))

        for i, row_value in enumerate(row_values):
            decoded_vector = Vector([Zp(int(e)) for e in row_value.split('.')])
            matrix[i] = decoded_vector
        
        # print(f'Received matrix ', matrix)
        return matrix

    def clean_up(self: 'MPCEnv'):
        for socket in self.sockets.values():
            socket.close()
  
    def reveal_sym(self: 'MPCEnv', elem: Zp) -> Zp:
        if (self.pid == 0):
            return None

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

    def rand_elem(self: 'MPCEnv') -> Zp:
        return Zp.randzp()
    
    def rand_vector(self: 'MPCEnv', size: int) -> Vector:
        return Vector([self.rand_elem() for _ in range(size)])

    def beaver_partition(self: 'MPCEnv', x: object, fid: int = 0) -> tuple:
        type_ = type(x)
        rand_func = None
        if isinstance(x, Zp):
            rand_func = self.rand_elem
        elif isinstance(x, Matrix):
            rand_func = partial(Matrix, m=x.num_rows(), n=x.num_cols(), randomise=True)
        elif isinstance(x, Vector):
            rand_func = partial(self.rand_vector, size=len(x))
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
        else:
            self.switch_seed(0)
            r: Zp = rand_func()
            self.restore_seed(0)
            
            if isinstance(x, Matrix):  # Temp hack. Will be removed in .seq
                x_r = self.reveal_sym(Matrix().from_value(x - r))
            else:
                x_r = self.reveal_sym(x - r)
        
        return x_r, r
    
    def mul_elem(self: 'MPCEnv', v_1: Vector, v_2: Vector) -> Vector:
        return v_1 * v_2

    def rand_mat(self: 'MPCEnv', m: int, n: int) -> Matrix:
        return Matrix(m, n, randomise=True)

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

    def powers(self: 'MPCEnv', x: Vector, pow: int) -> Matrix:
        assert pow >= 1

        n: int = len(x)
        b = Matrix()
        
        if (pow == 1):
            b.set_dims(2, n)
            if (self.pid > 0):
                if (self.pid == 1):
                    b[0] += Vector([Zp(1)] * n)
                b[1] = x * Vector([Zp(1)] * n)
        else:  # pow > 1
            x_r, r = self.beaver_partition(x)

            if (self.pid == 0):
                r_pow = Matrix(pow - 1, n)
                r_pow[0] = self.mul_elem(r, r)
                
                for p in range(1, r_pow.num_rows()):
                    r_pow[p] = self.mul_elem(r_pow[p - 1], r)

                self.switch_seed(1)
                r_ = self.rand_mat(pow - 1, n)
                self.restore_seed(1)

                r_pow -= r_

                self.send_elem(r_pow, 2)

                b.set_dims(pow + 1, n)
            else:
                r_pow = Matrix()
                if (self.pid == 1):
                    self.switch_seed(0)
                    r_pow = self.rand_mat(pow - 1, n)
                    self.restore_seed(0)
                else:  # pid == 2
                    r_pow = self.receive_matrix(0)

                x_r_pow = Matrix(pow - 1, n)
                x_r_pow[0] = self.mul_elem(x_r, x_r)
                for p in range(1, x_r_pow.num_rows()):
                    x_r_pow[p] = self.mul_elem(x_r_pow[p - 1], x_r)

                pascal_matrix = self.get_pascal_matrix(pow)

                b.set_dims(pow + 1, n)

                if (self.pid == 1):
                    b[0] += Vector([Zp(1)] * n)
                b[1] = x * Vector([Zp(1)] * n)

                for p in range(2, pow + 1):
                    if (self.pid == 1):
                        b[p] = x_r_pow[p - 2] * Vector([Zp(1)] * n)

                    if (p == 2):
                        b[p] += Vector([pascal_matrix[p][1]] * n) * self.mul_elem(x_r, r)
                    else:
                        b[p] += Vector([pascal_matrix[p][1]] * n) * self.mul_elem(x_r_pow[p - 3], r)

                        for j in range(2, p - 1):
                            b[p] += Vector([pascal_matrix[p][j]] * n) * self.mul_elem(x_r_pow[p - 2 - j], r_pow[j - 2])
                        
                        b[p] += Vector([pascal_matrix[p][p - 1]] * n) * self.mul_elem(x_r, r_pow[p - 3])

                    b[p] += r_pow[p - 2]
        
        return b
    
    def evaluate_poly(self: 'MPCEnv', x: Vector, coeff: Matrix) -> Matrix:
        n: int = len(x)
        npoly: int = coeff.num_rows()
        deg: int = coeff.num_cols() - 1

        pows: Matrix = self.powers(x, deg)

        if (self.pid > 0):
            return Matrix().from_value(coeff * pows)
        
        return Matrix(npoly, n)
    
    def add_public(self: 'MPCEnv', x: object, a: object) -> object:
        if self.pid == 1:
            return x + a
        return x
    
    def beaver_mult(self: 'MPCEnv', x_r: Matrix, r_1: Matrix,
                    y_r: Matrix, r_2: Matrix, elem_wise: bool, fid: int) -> Matrix:
        xy = Matrix(r_1.num_rows(), r_1.num_cols())
        if self.pid == 0:
            r_1_r_2 = self.mul_elem(r_1, r_2) if elem_wise else r_1.mult(r_2)
            xy += r_1_r_2
        else:
            if elem_wise:
                for i in range(xy.num_rows()):
                    for j in range(xy.num_cols()):
                        xy[i][j] += x_r[i][j] * r_2[i][j]
                        xy[i][j] += r_1[i][j] * y_r[i][j]
                    if self.pid == 1:
                        xy[i][j] += x_r[i][j] * y_r[i][j]
            else:
                xy += x_r * r_2
                xy += r_1 * y_r
                if self.pid == 1:
                    xy += x_r * y_r

        return xy

    def beaver_mult_elem(self: 'MPCEnv', x_1_r: Matrix, r_1: Matrix, x_2_r: Matrix, r_2: Matrix, fid: int = 0) -> Matrix:
        return self.beaver_mult(x_1_r, r_1, x_2_r, r_2, True, fid)
    
    def beaver_reconstruct(self: 'MPCEnv', elem: object, fid: int = 0) -> Matrix:
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
                mask = rand_func()  # fid was here
                self.restore_seed(1)

                # m -= mask
                # Mod(ab, fid);
                mm = Matrix().from_value(elem - mask) if isinstance(elem, Matrix) else elem - mask
                time.sleep(2)  # TODO: Fix sockets and remove this wait!
                self.send_elem(mm, 2)  # fid was here
                return mm
            else:
                rr = None
                if self.pid == 1:
                    self.switch_seed(0)
                    rr = rand_func()  # fid was here
                    self.restore_seed(0)
                else:
                    rr = receive_func(0)  # fid was here
                    

                return elem + rr
                # Mod(ab, fid);
            
    def mult_elem(self: 'MPCEnv', a: Matrix, b: Matrix, fid: int = 0) -> Matrix:
        x_1_r, r_1 = self.beaver_partition(a, fid)
        x_2_r, r_2 = self.beaver_partition(b, fid)
        
        c = self.beaver_mult_elem(x_1_r, r_1, x_2_r, r_2, fid)
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
        # print('Elem', elem, type(elem))
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
                rand_mat[i][j] = Zp.randzp(base=(2 ** num_bits - 1))

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
            print('Sent ...')
        elif self.pid == 2:
            r = self.receive_matrix(0)
            r_low = self.receive_matrix(0)
            print('Received ...')
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
                    c_low[i][j] = Zp(int(c[i][j]) & (2 ** m - 1))
                    # c_low[i][j] = conv<ZZ_p>(trunc_ZZ(c[i][j]), m);

        if (self.pid > 0):
            a += r_low
            if (self.pid == 1):
                a -= c_low

            if m not in self.invpow_cache:
                twoinv = Zp(2).inv()
                twoinvm = twoinv ** m
                self.invpow_cache[m] = twoinvm
                
            a *= self.invpow_cache[m]
