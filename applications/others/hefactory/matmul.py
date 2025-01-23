import numpy as np
from time import time
from HEFactory.Tapir import CGManager, CGSym, CGArray
from HEFactory.Boar import Boar


def matmul_example():
    m, n = 128, 8192
    mn = m * n
    matrix = np.random.randint(10, size=mn).reshape(m, n)
    vector = np.random.randint(10, size=n).reshape(n,)

    expected_res = vector.dot(vector)
    start_time = time()
    with CGManager(precision=10, performance=0, security=0) as cgm:
        result = []
        encrypted_vector_j = CGArray(cgm, vector)
        for i in range(m):
            encrypted_vector = CGArray(cgm, vector)
            for j in range(m):
                if i == 0:
                    encrypted_vector_j = CGArray(cgm, vector)
                resu = encrypted_vector * encrypted_vector_j
                res = resu.log_accumulate()
                if j == 0:
                    result.append(res)
        cgm.output([result])

    boar = Boar(verbose=True)
    boar.launch()
    results = {k: v for k, v in boar.grab_results().items()}
    for k, v in results.items():
        a = np.array(v)
        print(k, a[np.nonzero(a)])
    end_time = time()
    execution_time = end_time - start_time
    print("Matmul Execution Time:", execution_time, "seconds")
    print("-"*10 + "MATRIX x VECTOR" + "-"*10)
    print("MATRIX:\n", matrix)
    print("VECTOR:\n", vector)
    print("EXPECTED  RES:", expected_res)
    print("ENCRYPTED RES:", res)
    print("-"*(20 + len("MATRIX x VECTOR")))

matmul_example()
