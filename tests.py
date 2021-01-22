import time

import numpy as np

import param
from custom_types import TypeOps, add_mod
from mpc import MPCEnv
from param import BASE_P


def assert_values(result, expected):
    assert np.all(result == expected), f'Result: {result}. Expected: {expected}'


def assert_approx(result, expected, error = 10 ** (-1)):
    assert np.all(expected - error < result < expected + error), f'Result: {result}. Expected: {expected}'


def test_all(mpc: MPCEnv = None, pid: int = None):
    if mpc is not None and pid is not None:
        if pid != 0:
            assert_values(mpc.lagrange_cache[2][1][1], 3107018978382642104)
        
        revealed_value: np.ndarray = mpc.reveal_sym(np.array(10) if pid == 1 else np.array(7))
        if pid != 0:
            assert_values(revealed_value, 17)
        
        revealed_value: np.ndarray = mpc.reveal_sym(np.array([10, 11, 12]) if pid == 1 else np.array([7, 8, 9]))
        if pid != 0:
            assert_values(revealed_value, np.array([17, 19, 21]))
        
        revealed_value: np.ndarray = mpc.reveal_sym(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) if pid == 1
            else np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]))
        if pid != 0:
            assert_values(revealed_value, np.array([[11, 13, 15], [17, 19, 21], [23, 25, 27]]))
        
        x_r, r = mpc.beaver_partition(np.array(10) if pid == 1 else np.array(7), fid=0)
        if pid == 0:
            mpc.send_elem(r, 1)
            mpc.send_elem(r, 2)
        else:
            r_0 = mpc.receive_ndarray(0, msg_len=TypeOps.get_bytes_len(x_r), ndim=x_r.ndim, shape=x_r.shape)
            assert_values(r_0, mpc.reveal_sym(r))
            assert_values((x_r + mpc.reveal_sym(r)) % mpc.primes[0], np.array(17))
        
        x_r, r = mpc.beaver_partition(np.array([10, 11, 12]) if pid == 1 else np.array([3, 4, 5]), fid=0)
        if pid == 0:
            mpc.send_elem(r, 1)
            mpc.send_elem(r, 2)
        else:
            r_0 = mpc.receive_ndarray(0, msg_len=TypeOps.get_bytes_len(x_r), ndim=x_r.ndim, shape=x_r.shape)
            assert_values(r_0, mpc.reveal_sym(r))
            assert_values(add_mod(x_r, mpc.reveal_sym(r), mpc.primes[0]), np.array([13, 15, 17]))
        
        x_r, r = mpc.beaver_partition(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) if pid == 1 else
            np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]), fid=0)
        if pid == 0:
            mpc.send_elem(r, 1)
            mpc.send_elem(r, 2)
        else:
            r_0 = mpc.receive_ndarray(0, msg_len=TypeOps.get_bytes_len(x_r), ndim=x_r.ndim, shape=x_r.shape)
            assert_values(r_0, mpc.reveal_sym(r))
            assert_values(add_mod(x_r, mpc.reveal_sym(r), mpc.primes[0]), np.array([[11, 13, 15], [17, 19, 21], [23, 25, 27]]))
        
        p: np.ndarray = mpc.powers(np.array([2, 0 if pid == 1 else 1, 3], dtype=np.int64), 10, fid=0)
        revealed_p: np.ndarray = mpc.reveal_sym(p)
        if pid != 0:
            assert_values(revealed_p[10], np.array([1048576, 1, 60466176], dtype=np.int64))
        
        coeff = np.array([[1] * 3, [2] * 3, [3] * 3], dtype=np.int64)
        p: np.ndarray = mpc.evaluate_poly(np.array([1, 2, 3], dtype=np.int64), coeff, fid=0)
        revealed_p = mpc.reveal_sym(p)
        if pid != 0:
            expected_mat = np.array([
                [7, 21, 43],
                [14, 42, 86],
                [21, 63, 129]], dtype=np.int64)
            assert_values(revealed_p, expected_mat)
        
        a: int = mpc.double_to_fp(2 if pid == 1 else 1.14, param.NBIT_K, param.NBIT_F, fid=0)
        b: int = mpc.double_to_fp(3 if pid == 1 else 2.95, param.NBIT_K, param.NBIT_F, fid=0)
        float_a = mpc.print_fp(np.array(a), fid=0)
        float_b = mpc.print_fp(np.array(b), fid=0)
        if pid != 0:
            assert_approx(float_a, 3.14)
            assert_approx(float_b, 5.95)

        pub: int = mpc.double_to_fp(5.07, param.NBIT_K, param.NBIT_F, fid=0)
        a: np.ndarray = mpc.add_public(np.array(a, dtype=np.int64), np.array(pub, dtype=np.int64), fid=0)
        float_a = mpc.print_fp(a, fid=0)
        if pid != 0:
            assert_approx(float_a, 8.21)

        a_mat = np.array([[int(a)]], dtype=np.int64)
        b_mat = np.array([[int(b)]], dtype=np.int64)
        d: np.ndarray = mpc.multiply(a_mat, b_mat, elem_wise=True, fid=0)
        mpc.print_fp(d, fid=0)
        d = mpc.trunc(d, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
        float_d = mpc.print_fp(d, fid=0)
        if pid != 0:
            assert_approx(float_d, 48.8495)

        a = np.array([9 if pid == 1 else 7, 5 if pid == 1 else 3])
        b = np.array([4, 1])
        p = mpc.multiply(a, b, elem_wise=True, fid=0)
        revealed_p = mpc.reveal_sym(p, fid=0)
        expected_p = np.array([128, 16])
        if pid != 0:
            assert_values(revealed_p, expected_p)
        
        p_or = mpc.reveal_sym(mpc.prefix_or(a_mat, fid=0), fid=0)
        if pid != 0:
            assert_values(p_or, 1599650766643921085)
        
        nee = mpc.normalizer_even_exp(b)
        if pid != 0:
            nee_0 = mpc.print_fp(mpc.reveal_sym(nee[0], 0), 0)[0]
            assert_values(nee_0, 131072)

        # a = np.array([
        #     mpc.double_to_fp(18, param.NBIT_K, param.NBIT_F, 0),
        #     mpc.double_to_fp(128, param.NBIT_K, param.NBIT_F, 0),
        #     mpc.double_to_fp(32, param.NBIT_K, param.NBIT_F, 0),
        #     mpc.double_to_fp(50, param.NBIT_K, param.NBIT_F, 0)], dtype=np.int64)
        # b, b_inv = mpc.fp_sqrt(a)
        # float_b: np.ndarray = mpc.print_fp(b, fid=0)
        # float_b_inv: np.ndarray = mpc.print_fp(b_inv, fid=0)
        # if pid != 0:
        #     assert_approx(float_b, np.array([6, 16, 8, 10]))
        #     assert_approx(float_b_inv, np.array([0.1666666, 0.0625, 0.125, 0.1]))

        # a = Vector([Zp(7, BASE_P), Zp(256, BASE_P), Zp(99, BASE_P), Zp(50, BASE_P)])
        # b = Vector([Zp(6, BASE_P), Zp(16, BASE_P), Zp(3, BASE_P), Zp(40, BASE_P)])
        # d = mpc.fp_div(a, b, fid=0)
        # float_d_1 = mpc.print_fp_elem(d[0], fid=0)
        # float_d_2 = mpc.print_fp_elem(d[1], fid=0)
        # float_d_3 = mpc.print_fp_elem(d[2], fid=0)
        # float_d_4 = mpc.print_fp_elem(d[3], fid=0)
        # if pid != 0:
        #     assert_approx(float_d_1, 1.1666666)
        #     assert_approx(float_d_2, 16)
        #     assert_approx(float_d_3, 33)
        #     assert_approx(float_d_4, 1.25)
        
        # a = Vector([
        #     mpc.double_to_fp(1.5, param.NBIT_K, param.NBIT_F, 0),
        #     mpc.double_to_fp(0.5, param.NBIT_K, param.NBIT_F, 0),
        #     mpc.double_to_fp(2.5, param.NBIT_K, param.NBIT_F, 0)])
        # v: Vector = mpc.householder(a)
        # float_v_1 = mpc.print_fp_elem(v[0], fid=0)
        # float_v_2 = mpc.print_fp_elem(v[1], fid=0)
        # float_v_3 = mpc.print_fp_elem(v[2], fid=0)
        # if pid != 0:
        #     assert_approx(float_v_1, 0.86807)
        #     assert_approx(float_v_2, 0.0973601)
        #     assert_approx(float_v_3, 0.486801)
        
        # mat = Vector([
        #     Vector([mpc.double_to_fp(4, param.NBIT_K, param.NBIT_F, 0) for _ in range(3)]),
        #     Vector([mpc.double_to_fp(4.5, param.NBIT_K, param.NBIT_F, 0) for _ in range(3)]),
        #     Vector([mpc.double_to_fp(5.5, param.NBIT_K, param.NBIT_F, 0) for _ in range(3)])])
        # mat = Matrix().from_value(mat)
        # q, r = mpc.qr_fact_square(mat)
        # result_q = mpc.print_fp(q, fid=0)
        # result_r = mpc.print_fp(r, fid=0)
        # expected_q = Vector([
        #     Vector([-0.57735, -0.57735, -0.57735]),
        #     Vector([-0.57735, 0.788675, -0.211325]),
        #     Vector([-0.57735, -0.211325, 0.788675])])
        # expected_r = Vector([
        #     Vector([-13.85640, 0, 0]),
        #     Vector([-15.58846, 0, 0]),
        #     Vector([-19.05255, 0, 0])])
        # if pid != 0:
        #     assert_approx(result_q, expected_q)
        #     assert_approx(result_r, expected_r)
        
        # mat = Vector([
        #     Vector([mpc.double_to_fp(4, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(3, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(2.5, param.NBIT_K, param.NBIT_F, 0)]),
        #     Vector([mpc.double_to_fp(0.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(4.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(1.5, param.NBIT_K, param.NBIT_F, 0)]),
        #     Vector([mpc.double_to_fp(5.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(2, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(1, param.NBIT_K, param.NBIT_F, 0)])])
        # t, q = mpc.tridiag(mat)
        # result_t = mpc.print_fp(t, fid=0)
        # result_q = mpc.print_fp(q, fid=0)
        # expected_t = Vector([
        #     Vector([8, -7.81025, 0]),
        #     Vector([-7.81025, 9.57377, 3.31148]),
        #     Vector([0, 2.31148, 1.42623])])
        # expected_q = Vector([
        #     Vector([1, 0, 0]),
        #     Vector([0, -0.768221, -0.640184]),
        #     Vector([0, -0.640184, 0.768221])])
        # if pid != 0:
        #     assert_approx(result_t, expected_t)
        #     assert_approx(result_q, expected_q)
        
        # mat = Vector([
        #     Vector([mpc.double_to_fp(4, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(3, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(2.5, param.NBIT_K, param.NBIT_F, 0)]),
        #     Vector([mpc.double_to_fp(0.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(4.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(1.5, param.NBIT_K, param.NBIT_F, 0)]),
        #     Vector([mpc.double_to_fp(5.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(2, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(1, param.NBIT_K, param.NBIT_F, 0)])])
        # v, l = mpc.eigen_decomp(mat)
        # result_v = mpc.print_fp(v, fid=0)
        # result_l = mpc.print_fp(Matrix().from_value(Vector([l])), fid=0)
        # expected_v = Vector([
        #     Vector([0.650711, 0.672083, 0.353383]),
        #     Vector([-0.420729, -0.0682978, 0.904612]),
        #     Vector([0.632109, -0.73732, 0.238322])])
        # expected_l = Vector([
        #     Vector([16.91242, -0.798897, 2.88648])])
        # if pid != 0:
        #     assert_approx(result_v, expected_v)
        #     assert_approx(result_l, expected_l)
        
        # mat = Vector([
        #     Vector([mpc.double_to_fp(4, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(3, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(2.5, param.NBIT_K, param.NBIT_F, 0)]),
        #     Vector([mpc.double_to_fp(0.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(4.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(1.5, param.NBIT_K, param.NBIT_F, 0)]),
        #     Vector([mpc.double_to_fp(5.5, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(2, param.NBIT_K, param.NBIT_F, 0),
        #             mpc.double_to_fp(1, param.NBIT_K, param.NBIT_F, 0)])])
        # q = mpc.orthonormal_basis(mat)
        # result_q = mpc.print_fp(q, fid=0)
        # expected_q = Vector([
        #     Vector([-0.715542, -0.536656, -0.447214]),
        #     Vector([0.595097, -0.803563, 0.0121201]),
        #     Vector([0.365868, 0.257463, -0.894345])])
        # if pid != 0:
        #     assert_approx(result_q, expected_q)

    print(f'All tests passed at {pid}!')


def benchmark(mpc: MPCEnv, pid: int, m: int, n: int):
    import random, math
    
    # mat = Vector([
    #     Vector(
    #         [Zp(0, base=BASE_P) for j in range(n)])
    #         for i in range(m)])
    # mat = Matrix().from_value(mat)
    
    # print('Orthonormal basis ...')
    # mpc.orthonormal_basis(mat)
    # print('QR ...')
    # mpc.qr_fact_square(mat)
    # print('Tridiag ...')
    # mpc.tridiag(mat)
    
    # print('Eigen decomp ...')
    # from profilehooks import profile
    # fn = profile(mpc.eigen_decomp, entries=200) if pid == 1 else mpc.eigen_decomp
    # fn(mat)

    print(f'Benchmarks done at {pid}!')


if __name__ == "__main__":
    test_all()
