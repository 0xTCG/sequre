from eva import *
from eva.ckks import *
from eva.seal import generate_keys
import time
import numpy as np


def l2_distance():
    x1_p = np.random.uniform(0, 10, size=8192)
    x2_p = np.random.uniform(0, 10, size=8192)

    def he_sqrt(x, d=2):
        a = x
        b = x - 1
        for i in range(d):
            b_tmp = (b * 0.5)
            b_tmp = (1 - b_tmp)
            a = a * 1
            a_new = a * b
            if i != (d - 1):
                b = (b * b) * ((b - 3) * 0.25)
        return a_new

    def logaccumulate(x, log2N):
        for i in range(log2N):
            if i == 0:
                accum = x
            else:
                accum += accum >> (1 << i)
        return accum


    poly = EvaProgram('l2_distance', vec_size=len(x1_p))

    with poly:
        x1 = Input('x1')
        x2 = Input('x2') 
        d = x1 - x2
        e = d * d
        e = logaccumulate(e,13)
        res = he_sqrt(e)
        Output('y', res)
    poly.set_output_ranges(30)
    poly.set_input_scales(30)


    compiler = CKKSCompiler()
    compiled_poly, params, signature = compiler.compile(poly)
    print(params.prime_bits)

    public_ctx, secret_ctx = generate_keys(params)

    inputs = { 'x1': x1_p, 'x2': x2_p}
    encInputs = public_ctx.encrypt(inputs, signature)
    input_1 = { 'x1': x1_p}
    for i in range(32):
        encInputs_dum = public_ctx.encrypt(input_1, signature)
        for _ in range(32):
            if i == 0:
                encInputs_dum = public_ctx.encrypt(input_1, signature)
            encOutputs = public_ctx.execute(compiled_poly, encInputs)
            outputs = secret_ctx.decrypt(encOutputs, signature)

    return 0, (params.poly_modulus_degree, params.prime_bits)

if __name__ == "__main__":
    a = time.time()
    try:
        l2_distance()
    except:
        print("RUN FAILED")
    b = time.time()
    print("L2 Execution Time:", b - a)
