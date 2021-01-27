import math

from functools import partial

import numpy as np

import utils.param as param

from mpc.arithmetic import Arithmetic
from mpc.fp import FP
from mpc.boolean import Boolean
from utils.custom_types import add_mod, mul_mod, zeros
from utils.type_ops import TypeOps


class LinAlg:
    def __init__(self: 'LinAlg', pid: int, primes: dict, arithmetic: Arithmetic, boolean: Boolean, fp: FP):
        self.pid = pid
        self.primes = primes
        self.arithmetic = arithmetic
        self.boolean = boolean
        self.fp = fp

        self.primes_bits: dict = {k: math.ceil(math.log2(v)) for k, v in self.primes.items()}
        self.primes_bytes: dict = {k: (v + 7) // 8 for k, v in self.primes_bits.items()}


    def householder(self: 'MPCEnv', x: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        n: int = len(x)
        add_func = partial(add_mod, field=field)
        mul_func = partial(mul_mod, field=field)
        
        xr, xm = self.arithmetic.beaver_partition(x)

        xdot: np.ndarray = self.arithmetic.beaver_inner_prod(xr, xm)
        xdot = np.array([xdot], dtype=np.int64)
        xdot = self.arithmetic.beaver_reconstruct(xdot)
        xdot = self.fp.trunc(xdot)

        # Bottleneck
        xnorm, _ = self.fp.fp_sqrt(xdot)

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
        dot_shift = self.fp.trunc(dot_shift)

        vdot = zeros(1)
        if self.pid > 0:
            vdot = mul_func(add_func(xdot, dot_shift), 2)

        # Bottleneck
        _, vnorm_inv = self.fp.fp_sqrt(vdot)

        invr, invm = self.arithmetic.beaver_partition(vnorm_inv[0])

        vr = zeros(n)
        if self.pid > 0:
            vr = xr.copy()
            vr[0] = add_func(vr[0], sr)
        vm = xm.copy()
        vm[0] = add_func(vm[0], sm)

        v: np.ndarray = self.arithmetic.beaver_mult(vr, vm, invr, invm, True)
        v = self.arithmetic.beaver_reconstruct(v)
        v = self.fp.trunc(v)

        return v
    
    def qr_fact_square(self: 'MPCEnv', A: np.ndarray, field: int = param.BASE_P) -> np.ndarray:
        assert A.shape[0] == A.shape[1]

        n: int = A.shape[0]
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
            P = self.fp.trunc(P, param.NBIT_K + param.NBIT_F, param.NBIT_F)

            if self.pid > 0:
                P = np.mod(-add_func(P, P), field)
                if self.pid == 1:
                    np.fill_diagonal(P, add_func(P.diagonal(), one))
            
            B = zeros((n - i, n - i))
            if i == 0:
                Q = P
                B = self.arithmetic.multiply(Ap, P, False)
                B = self.fp.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F)
            else:
                Qsub = zeros((n - i, n))
                if self.pid > 0:
                    Qsub[:n - i] = Q[i: n]

                left: list = [P, Ap]
                right: list = [Qsub, P]

                prod: list = self.arithmetic.mult_mat_parallel(left, right)
                # TODO: parallelize Trunc
                prod[0] = self.fp.trunc(prod[0], param.NBIT_K + param.NBIT_F, param.NBIT_F)
                prod[1] = self.fp.trunc(prod[1], param.NBIT_K + param.NBIT_F, param.NBIT_F)

                if self.pid > 0:
                    Q[i:n] = prod[0][:n - i]
                    B = prod[1]
            
            Ap = zeros((n - i - 1, n - i - 1))
            if self.pid > 0:
                R[i:n, i] = B[:n-i, 0]
                if i == n - 2: R[n - 1][n - 1] = B[1][1]
                Ap[:n - i - 1, :n - i - 1] = B[1:n - i, 1:n - i]
            
        return Q, R

    def tridiag(self: 'MPCEnv', A: np.ndarray, field: int = param.BASE_P) -> tuple:
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] > 2

        n: int = A.shape[0]
        one: int = TypeOps.double_to_fp(1, param.NBIT_K, param.NBIT_F)
        add_func = partial(add_mod, field=field)

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
            vv = self.fp.trunc(vv, param.NBIT_K + param.NBIT_F, param.NBIT_F)

            P = zeros(Ap.shape)
            if self.pid > 0:
                cols_no = Ap.shape[1]
                P[1:cols_no, 1:cols_no] = np.mod(
                    -add_func(vv[0:cols_no-1, 0:cols_no-1], vv[0:cols_no-1, 0:cols_no-1]), field)
                if self.pid == 1:
                    np.fill_diagonal(P, add_func(P.diagonal(), one))

            # TODO: parallelize? (minor improvement)
            PAp: np.ndarray = self.arithmetic.multiply(P, Ap, False)
            PAp = self.fp.trunc(PAp, param.NBIT_K + param.NBIT_F, param.NBIT_F)
            B = self.arithmetic.multiply(PAp, P, False)
            B = self.fp.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F)

            Qsub = zeros((n, n - i))
            if self.pid > 0:
                Qsub[:, :n - i] = Q[:, i:n]

            Qsub: np.ndarray = self.arithmetic.multiply(Qsub, P, False)
            Qsub = self.fp.trunc(Qsub, param.NBIT_K + param.NBIT_F, param.NBIT_F)
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

    def eigen_decomp(self: 'MPCEnv', A: np.ndarray, field: int = param.BASE_P) -> tuple:
        assert A.shape[0] == A.shape[1]
        n: int = A.shape[0]
        add_func = partial(add_mod, field=field)

        L = zeros(n)

        Ap, Q = self.tridiag(A)
        V: np.ndarray = Q.T

        for i in range(n - 1, 0, -1):
            for _ in range(param.ITER_PER_EVAL):
                shift = Ap[i][i]
                if self.pid > 0:
                    np.fill_diagonal(Ap, np.mod(Ap.diagonal() - shift, field))

                Q, R = self.qr_fact_square(Ap)

                Ap = self.arithmetic.multiply(Q, R, False)
                Ap = self.fp.trunc(Ap, param.NBIT_K + param.NBIT_F, param.NBIT_F)

                if self.pid > 0:
                    np.fill_diagonal(Ap, add_func(Ap.diagonal(), shift))

                Vsub = zeros((i + 1, n))
                if self.pid > 0:
                    Vsub[:i + 1] = V[:i + 1]

                Vsub = self.arithmetic.multiply(Q, Vsub, False)
                Vsub = self.fp.trunc(Vsub, param.NBIT_K + param.NBIT_F, param.NBIT_F)

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
        add_func: callable = partial(add_mod, field=field)

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
            Apv = self.fp.trunc(Apv, param.NBIT_K + param.NBIT_F, param.NBIT_F)

            B = self.arithmetic.multiply(Apv, v, False)
            B = self.fp.trunc(B, param.NBIT_K + param.NBIT_F, param.NBIT_F)

            if self.pid > 0:
                B = np.mod(-B, field)
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
            Qv = self.fp.trunc(Qv, param.NBIT_K + param.NBIT_F, param.NBIT_F)

            Qvv = self.arithmetic.multiply(Qv, v, False)
            Qvv = self.fp.trunc(Qvv, param.NBIT_K + param.NBIT_F, param.NBIT_F)
            if self.pid > 0:
                Qvv = np.mod(-Qvv, field)
                Qvv = add_func(Qvv, Qvv)

            if self.pid > 0:
                # TODO: Vectorize
                for j in range(c):
                    for k in range(n - i):
                        Q[j][k + i] = add_func(Q[j][k + i], Qvv[j][k])

        return Q
