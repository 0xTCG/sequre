import numpy as np
from time import time

from HEFactory.Tapir import CGManager, CGSym, CGArray
from HEFactory.Boar import Boar

def compile():
    a = np.random.uniform(-100, 100, size=8192)
    b = np.random.uniform(-100, 100, size=8192)
    print("expected: ", np.linalg.norm(a - b))

    with CGManager() as cgm:
        result = []
        encrypted_b = CGArray(cgm, b)
        for i in range(128):
            encrypted_a = CGArray(cgm, a)
            for j in range(128):
                if i == 0:
                    encrypted_b = CGArray(cgm, b)
                d = encrypted_a - encrypted_b
                e = d * d
                f = e.log_accumulate()
                #res = f.sqrt(d=2)
                result.append(f)
        cgm.output([result])

    print("L2 Result: ", result)
    return result

def execute(c):
    boar = Boar(verbose=True)
    boar.launch()
    results = {k: v for k, v in boar.grab_results().items()}
    for k, v in results.items():
        a = np.array(v)
        print(k, a[np.nonzero(a)])

def l2_distance():
    start_time = time()
    c = compile()
    execute(c)
    end_time = time()
    execution_time = end_time - start_time
    print("L2 Execution Time:", execution_time, "seconds")

if __name__ == "__main__":
    l2_distance()



