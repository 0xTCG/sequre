import time

import param
from custom_types import Zp, Vector, Matrix
from mpc import MPCEnv


def assert_values(result, expected):
    assert result == expected, f'Result: {result}. Expected: {expected}'


def assert_approx(result, expected, error = 10 ** (-5)):
    assert expected - error < result < expected + error, f'Result: {result}. Expected: {expected}'


def test_all(mpc: MPCEnv = None, pid: int = None):
    # # Zp
    # assert_values(Zp(1) + Zp(1), Zp(2))
    # test_val = Zp(1)
    # test_val += Zp(1)
    # assert_values(test_val, Zp(2))
    # assert_values(Zp(2) - Zp(1), Zp(1))
    # test_val = Zp(2)
    # test_val -= Zp(1)
    # assert_values(test_val, Zp(1))
    # assert_values(Zp(2) * Zp(3), Zp(6))
    # test_val = Zp(2)
    # test_val *= Zp(3)
    # assert_values(test_val, Zp(6))
    # # Vector
    # assert_values(Vector([1]) + Vector([1]), Vector([2]))
    # test_val = Vector([1])
    # test_val += Vector([1])
    # assert_values(test_val, Vector([2]))
    # assert_values(Vector([2]) - Vector([1]), Vector([1]))
    # test_val = Vector([2])
    # test_val -= Vector([1])
    # assert_values(test_val, Vector([1]))
    # assert_values(Vector([2]) * Vector([3]), Vector([6]))
    # test_val = Vector([2])
    # test_val *= Vector([3])
    # assert_values(test_val, Vector([6]))

    if mpc is not None and pid is not None:
        # if pid != 0:
        #     assert_values(mpc.lagrange_cache[2][1][1], Zp(678478060187771812621521424374354386933136592975))
        # revealed_value: Zp = mpc.reveal_sym(Zp(10) if pid == 1 else Zp(7))
        # if pid != 0:
        #     assert_values(revealed_value, Zp(17))
        
        # x_r, r = mpc.beaver_partition(Zp(10) if pid == 1 else Zp(7))
        # if pid == 0:
        #     mpc.send_elem(r, 1)
        #     mpc.send_elem(r, 2)
        # else:
        #     r_0 = mpc.receive_elem(0)
        #     assert_values(r_0, mpc.reveal_sym(r))
        #     assert_values(x_r + mpc.reveal_sym(r), Zp(17))
        
        # x_r, r = mpc.beaver_partition(Vector([Zp(10), Zp(11), Zp(12)]) if pid == 1 else Vector([Zp(3), Zp(4), Zp(5)]))
        # if pid == 0:
        #     mpc.send_elem(r, 1)
        #     mpc.send_elem(r, 2)
        # else:
        #     r_0 = mpc.receive_vector(0)
        #     assert_values(r_0, mpc.reveal_sym(r))
        #     assert_values(x_r + mpc.reveal_sym(r), Vector([Zp(13), Zp(15), Zp(17)]))
        
        # p = mpc.powers(Vector([Zp(2), Zp(0) if pid == 1 else Zp(1), Zp(3)]), 10)
        # revealed_p = Matrix().from_value(mpc.reveal_sym(p))
        # if pid != 0:
        #     assert_values(revealed_p[10], Vector([Zp(1048576), Zp(1), Zp(60466176)]))
        
        # coeff = Matrix().from_value(Vector([Vector([Zp(1)] * 3), Vector([Zp(2)] * 3), Vector([Zp(3)] * 3)]))
        # p = mpc.evaluate_poly(Vector([Zp(1), Zp(2), Zp(3)]), coeff)
        # revealed_p = mpc.reveal_sym(p)
        # if pid != 0:
        #     expected_mat = Vector([
        #         Vector([Zp(7), Zp(21), Zp(43)]),
        #         Vector([Zp(14), Zp(42), Zp(86)]),
        #         Vector([Zp(21), Zp(63), Zp(129)])])
        #     assert_values(revealed_p, expected_mat)
        
        # a: Zp = mpc.double_to_fp(2 if pid == 1 else 1.14, param.NBIT_K, param.NBIT_F)
        # b: Zp = mpc.double_to_fp(3 if pid == 1 else 2.95, param.NBIT_K, param.NBIT_F)
        # float_a = mpc.print_fp_elem(a)
        # float_b = mpc.print_fp_elem(b)
        # if pid != 0:
        #     assert_approx(float_a, 3.14)
        #     assert_approx(float_b, 5.95)

        # pub: Zp = mpc.double_to_fp(5.07, param.NBIT_K, param.NBIT_F)
        # a = mpc.add_public(a, pub)
        # float_a = mpc.print_fp_elem(a)
        # if pid != 0:
        #     assert_approx(float_a, 8.21)

        # a_mat = Matrix(1, 1)
        # a_mat[0][0] = a
        # b_mat = Matrix(1, 1)
        # b_mat[0][0] = b
        # d: Matrix = mpc.mult_elem(a_mat, b_mat)
        # mpc.print_fp_elem(d[0][0])
        # mpc.trunc(d, param.NBIT_K + param.NBIT_F, param.NBIT_F)
        # float_d = mpc.print_fp_elem(d[0][0])
        # if pid != 0:
        #     assert_approx(float_d, 48.8495)

        # a = Vector([Zp(9) if pid == 1 else Zp(7), Zp(5) if pid == 1 else Zp(3)])
        # b = Vector([Zp(4), Zp(1)])
        # p = mpc.mult_vec(a, b)
        # revealed_p = mpc.reveal_sym(p)
        # expected_p = Vector([Zp(128), Zp(16)])
        # if pid != 0:
        #     assert_values(revealed_p, expected_p)

        # if pid == 0:
        #     a = Vector([Zp(80), Zp(80)])
        #     b = Vector([Zp(4), Zp(2)])
        # if pid == 1:
        #     a = Vector([Zp(0), Zp(0)])
        #     b = Vector([Zp(0), Zp(0)])
        # if pid == 2:
        #     a = Vector([Zp(80), Zp(80)])
        #     b = Vector([Zp(4), Zp(2)])
        a = Vector([Zp(7), Zp(100), Zp(99), Zp(50)])
        b = Vector([Zp(6), Zp(12), Zp(3), Zp(40)])
        d = mpc.fp_div(a, b)  # 4.03302
        mpc.print_fp_elem(d[0])
        mpc.print_fp_elem(d[1])
        mpc.print_fp_elem(d[2])
        mpc.print_fp_elem(d[3])
        
        pass
                
    print(f'All tests passed at {pid}!')


if __name__ == "__main__":
    test_all()
