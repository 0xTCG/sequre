import time

import param
from custom_types import Zp, Vector, Matrix
from mpc import MPCEnv
from param import BASE_P


def assert_values(result, expected):
    assert result == expected, f'Result: {result}. Expected: {expected}'


def assert_approx(result, expected, error = 10 ** (-5)):
    assert expected - error < result < expected + error, f'Result: {result}. Expected: {expected}'


def test_all(mpc: MPCEnv = None, pid: int = None):
    # Zp
    assert_values(Zp(1, BASE_P) + Zp(1, BASE_P), Zp(2, BASE_P))
    test_val = Zp(1, BASE_P)
    test_val += Zp(1, BASE_P)
    assert_values(test_val, Zp(2, BASE_P))
    assert_values(Zp(2, BASE_P) - Zp(1, BASE_P), Zp(1, BASE_P))
    test_val = Zp(2, BASE_P)
    test_val -= Zp(1, BASE_P)
    assert_values(test_val, Zp(1, BASE_P))
    assert_values(Zp(2, BASE_P) * Zp(3, BASE_P), Zp(6, BASE_P))
    test_val = Zp(2, BASE_P)
    test_val *= Zp(3, BASE_P)
    assert_values(test_val, Zp(6, BASE_P))
    # Vector
    assert_values(Vector([1]) + Vector([1]), Vector([2]))
    test_val = Vector([1])
    test_val += Vector([1])
    assert_values(test_val, Vector([2]))
    assert_values(Vector([2]) - Vector([1]), Vector([1]))
    test_val = Vector([2])
    test_val -= Vector([1])
    assert_values(test_val, Vector([1]))
    assert_values(Vector([2]) * Vector([3]), Vector([6]))
    test_val = Vector([2])
    test_val *= Vector([3])
    assert_values(test_val, Vector([6]))

    if mpc is not None and pid is not None:
        if pid != 0:
            assert_values(mpc.lagrange_cache[2][1][1], Zp(678478060187771812621521424374354386933136592975, BASE_P))
        revealed_value: Zp = mpc.reveal_sym(Zp(10, BASE_P) if pid == 1 else Zp(7, BASE_P), fid=0)
        if pid != 0:
            assert_values(revealed_value, Zp(17, BASE_P))
        
        x_r, r = mpc.beaver_partition(Zp(10, BASE_P) if pid == 1 else Zp(7, BASE_P), fid=0)
        if pid == 0:
            mpc.send_elem(r, 1)
            mpc.send_elem(r, 2)
        else:
            r_0 = mpc.receive_elem(0, fid=0)
            assert_values(r_0, mpc.reveal_sym(r, fid=0))
            assert_values(x_r + mpc.reveal_sym(r, fid=0), Zp(17, BASE_P))
        
        x_r, r = mpc.beaver_partition(Vector([Zp(10, BASE_P), Zp(11, BASE_P), Zp(12, BASE_P)]) if pid == 1 else Vector([Zp(3, BASE_P), Zp(4, BASE_P), Zp(5, BASE_P)]), fid=0)
        if pid == 0:
            mpc.send_elem(r, 1)
            mpc.send_elem(r, 2)
        else:
            r_0 = mpc.receive_vector(0, fid=0)
            assert_values(r_0, mpc.reveal_sym(r, fid=0))
            assert_values(x_r + mpc.reveal_sym(r, fid=0), Vector([Zp(13, BASE_P), Zp(15, BASE_P), Zp(17, BASE_P)]))
        
        p = mpc.powers(Vector([Zp(2, BASE_P), Zp(0, BASE_P) if pid == 1 else Zp(1, BASE_P), Zp(3, BASE_P)]), 10, fid=0)
        revealed_p = Matrix().from_value(mpc.reveal_sym(p, fid=0))
        if pid != 0:
            assert_values(revealed_p[10], Vector([Zp(1048576, BASE_P), Zp(1, BASE_P), Zp(60466176, BASE_P)]))
        
        coeff = Matrix().from_value(Vector([Vector([Zp(1, BASE_P)] * 3), Vector([Zp(2, BASE_P)] * 3), Vector([Zp(3, BASE_P)] * 3)]))
        p = mpc.evaluate_poly(Vector([Zp(1, BASE_P), Zp(2, BASE_P), Zp(3, BASE_P)]), coeff, fid=0)
        revealed_p = mpc.reveal_sym(p, fid=0)
        if pid != 0:
            expected_mat = Vector([
                Vector([Zp(7, BASE_P), Zp(21, BASE_P), Zp(43, BASE_P)]),
                Vector([Zp(14, BASE_P), Zp(42, BASE_P), Zp(86, BASE_P)]),
                Vector([Zp(21, BASE_P), Zp(63, BASE_P), Zp(129, BASE_P)])])
            assert_values(revealed_p, expected_mat)
        
        a: Zp = mpc.double_to_fp(2 if pid == 1 else 1.14, param.NBIT_K, param.NBIT_F, fid=0)
        b: Zp = mpc.double_to_fp(3 if pid == 1 else 2.95, param.NBIT_K, param.NBIT_F, fid=0)
        float_a = mpc.print_fp_elem(a, fid=0)
        float_b = mpc.print_fp_elem(b, fid=0)
        if pid != 0:
            assert_approx(float_a, 3.14)
            assert_approx(float_b, 5.95)

        pub: Zp = mpc.double_to_fp(5.07, param.NBIT_K, param.NBIT_F, fid=0)
        a = mpc.add_public(a, pub)
        float_a = mpc.print_fp_elem(a, fid=0)
        if pid != 0:
            assert_approx(float_a, 8.21)

        a_mat = Matrix(1, 1)
        a_mat[0][0] = a
        b_mat = Matrix(1, 1)
        b_mat[0][0] = b
        d: Matrix = mpc.mult_elem(a_mat, b_mat, fid=0)
        mpc.print_fp_elem(d[0][0], fid=0)
        mpc.trunc(d, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
        float_d = mpc.print_fp_elem(d[0][0], fid=0)
        if pid != 0:
            assert_approx(float_d, 48.8495)

        a = Vector([Zp(9, BASE_P) if pid == 1 else Zp(7, BASE_P), Zp(5, BASE_P) if pid == 1 else Zp(3, BASE_P)])
        b = Vector([Zp(4, BASE_P), Zp(1, BASE_P)])
        p = mpc.mult_vec(a, b, fid=0)
        revealed_p = mpc.reveal_sym(p, fid=0)
        expected_p = Vector([Zp(128, BASE_P), Zp(16, BASE_P)])
        if pid != 0:
            assert_values(revealed_p, expected_p)

        a = Vector([Zp(7, BASE_P), Zp(256, BASE_P), Zp(99, BASE_P), Zp(50, BASE_P)])
        b, b_inv = mpc.fp_sqrt(a)
        float_b_1 = mpc.print_fp_elem(b[0], fid=0)
        float_b_2 = mpc.print_fp_elem(b[1], fid=0)
        float_b_3 = mpc.print_fp_elem(b[2], fid=0)
        float_b_4 = mpc.print_fp_elem(b[3], fid=0)
        float_b_inv_1 = mpc.print_fp_elem(b_inv[0], fid=0)
        float_b_inv_2 = mpc.print_fp_elem(b_inv[1], fid=0)
        float_b_inv_3 = mpc.print_fp_elem(b_inv[2], fid=0)
        float_b_inv_4 = mpc.print_fp_elem(b_inv[3], fid=0)

        a = Vector([Zp(7, BASE_P), Zp(256, BASE_P), Zp(99, BASE_P), Zp(50, BASE_P)])
        b = Vector([Zp(6, BASE_P), Zp(16, BASE_P), Zp(3, BASE_P), Zp(40, BASE_P)])
        d = mpc.fp_div(a, b, fid=0)
        float_d_1 = mpc.print_fp_elem(d[0], fid=0)
        float_d_2 = mpc.print_fp_elem(d[1], fid=0)
        float_d_3 = mpc.print_fp_elem(d[2], fid=0)
        float_d_4 = mpc.print_fp_elem(d[3], fid=0)
        if pid != 0:
            assert_approx(float_d_1, 1.1666666)
            assert_approx(float_d_2, 16)
            assert_approx(float_d_3, 33)
            assert_approx(float_d_4, 1.25)
                
    print(f'All tests passed at {pid}!')


if __name__ == "__main__":
    test_all()
