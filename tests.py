import time

import param
from custom_types import Zp, Vector, Matrix
from mpc import MPCEnv


def assert_values(result, expected):
    assert result == expected, f'Result: {result}. Expected: {expected}'


def assert_approx(result, expected, error = 10 ** (-5)):
    assert expected - error < result < expected + error, f'Result: {result}. Expected: {expected}'


def test_all(mpc: MPCEnv = None, pid: int = None):
    # Zp
    assert_values(Zp(1) + Zp(1), Zp(2))
    test_val = Zp(1)
    test_val += Zp(1)
    assert_values(test_val, Zp(2))
    assert_values(Zp(2) - Zp(1), Zp(1))
    test_val = Zp(2)
    test_val -= Zp(1)
    assert_values(test_val, Zp(1))
    assert_values(Zp(2) * Zp(3), Zp(6))
    test_val = Zp(2)
    test_val *= Zp(3)
    assert_values(test_val, Zp(6))
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
            revealed_value: Zp = mpc.reveal_sym(Zp(10) if pid == 1 else Zp(7))
            assert_values(revealed_value, Zp(17))
        
        x_r, r = mpc.beaver_partition(Zp(10) if pid == 1 else Zp(7))
        if pid == 0:
            mpc.send_elem(r, 1)
            mpc.send_elem(r, 2)
        else:
            r_0 = mpc.receive_elem(0)
            assert_values(r_0, mpc.reveal_sym(r))
            assert_values(x_r + mpc.reveal_sym(r), Zp(17))
        
        x_r, r = mpc.beaver_partition(Vector([Zp(10), Zp(11), Zp(12)]) if pid == 1 else Vector([Zp(3), Zp(4), Zp(5)]))
        if pid == 0:
            mpc.send_elem(r, 1)
            mpc.send_elem(r, 2)
        else:
            r_0 = mpc.receive_vector(0)
            assert_values(r_0, mpc.reveal_sym(r))
            assert_values(x_r + mpc.reveal_sym(r), Vector([Zp(13), Zp(15), Zp(17)]))
        
        p = mpc.powers(Vector([Zp(1), Zp(2), Zp(3)]), 10)
        if pid != 0:
            revealed_p = Matrix().from_value(mpc.reveal_sym(p))
            assert_values(revealed_p[10], Vector([Zp(1024), Zp(1048576), Zp(60466176)]))
        
        coeff = Matrix().from_value(Vector([Vector([Zp(1)] * 3), Vector([Zp(2)] * 3), Vector([Zp(3)] * 3)]))
        p = mpc.evaluate_poly(Vector([Zp(1), Zp(2), Zp(3)]), coeff)
        if pid != 0:
            revealed_p = mpc.reveal_sym(p)
            expected_mat = Vector(
                [Vector([Zp(1)] * 3),
                 Vector([Zp(4), Zp(8), Zp(12)]),
                 Vector([Zp(12), Zp(48), Zp(108)])])
            assert_values(revealed_p, expected_mat)
        
        if pid != 0:
            a: Zp = mpc.double_to_fp(2 if pid == 1 else 1.14, param.NBIT_K, param.NBIT_F)
            b: Zp = mpc.double_to_fp(3 if pid == 1 else 2.95, param.NBIT_K, param.NBIT_F)
            assert_approx(mpc.print_fp_elem(a), 3.14)
            assert_approx(mpc.print_fp_elem(b), 5.95)

            pub: Zp = mpc.double_to_fp(5.07, param.NBIT_K, param.NBIT_F)
            a = mpc.add_public(a, pub)
            assert_approx(mpc.print_fp_elem(a), 8.21)

            d: Zp = mpc.mult_elem(a, b)
            mat = Matrix(1, 1)
            mat[0][0] = d
            mpc.trunc(mat)
            mpc.print_fp(d)
            mpc.print_fp(mat[0][0])
            # assert_values(mpc.print_fp(d), 48.8495) 
                

    print(f'All tests passed at {pid}!')


if __name__ == "__main__":
    test_all()
