from eva import *
from eva.ckks import *
from eva.seal import generate_keys
import time
import numpy as np

def matmul():
    m, n = 128, 8192
    mn = m * n
    matrix = np.random.randint(10, size=mn).reshape(m, n)
    vector = np.random.randint(10, size=n).reshape(n,)


    def logaccumulate(x, log2N):
        for i in range(log2N):
            if i == 0:
                accum = x
            else:
                accum += accum >> (1 << i)
        return accum


    matmult = EvaProgram('mat_mult', vec_size=len(vector))

    with matmult:
        x1 = Input('x1')
        x2 = Input('x2')
        e = x1 * x2
        res = logaccumulate(e, 13)
        res = res + res
        Output('y', res)
    matmult.set_output_ranges(30)
    matmult.set_input_scales(30)


    compiler = CKKSCompiler()
    compiled_poly, params, signature = compiler.compile(matmult)

    
    public_ctx, secret_ctx = generate_keys(params)

    inputs = { 'x1': vector, 'x2': vector}
    encInputs = public_ctx.encrypt(inputs, signature)
    input_1 = { 'x1': vector}
    for i in range(m):
        encInputs_dum = public_ctx.encrypt(input_1, signature)
        for _ in range(m):
            if i == 0:
                encInputs_dum = public_ctx.encrypt(input_1, signature)
            encOutputs = public_ctx.execute(compiled_poly, encInputs)
            outputs = secret_ctx.decrypt(encOutputs, signature)

    return 0, (params.poly_modulus_degree, params.prime_bits)

if __name__ == "__main__":
    a = time.time()
    matmul()
    b = time.time()
    print("Matmul Execution Time:", b - a)
