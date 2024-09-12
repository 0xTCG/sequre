import time, numpy as np, sys, tqdm
from mpyc.runtime import mpc
from mpyc.statistics import _fsqrt


async def main(args):
    n = 8192
    mpc.options.sec_param = 0
    raw_x = [-(i % 5) - 1.1 for i in range(n)]
    raw_y = [(i % 5) + 1.1 for i in range(n)]
    np_raw_x = np.array(raw_x)
    np_raw_y = np.array(raw_x)
    secnum = mpc.SecFxp(230, 8, ((1 << 251) - 9))
    print('Using secure fixed-point numbers:', secnum)

    async with mpc:
        if "--micro" in args or "--all" in args:
            # Encryption
            await mpc.transfer(mpc.pid)
            s = time.time()
            x = list(map(secnum, raw_x))
            print(f"{await mpc.output(x[0])} = {raw_x[0]}")
            e = time.time()
            print(f"Encryption took {e - s}s at CP{mpc.pid}.")
            y = list(map(secnum, raw_y))

            # NP Encryption
            await mpc.transfer(mpc.pid)
            s = time.time()
            np_x = secnum.array(np_raw_x)
            print(f"{await mpc.output(np_x[0])} = {raw_x[0]}")
            e = time.time()
            print(f"NP Encryption took {e - s}s at CP{mpc.pid}.")
            np_y = secnum.array(np_raw_y)

            # Plain addition
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = [a + 1.1 for a in x]
            print(f"{await mpc.output(res[0])} = {raw_x[0]} + 1.1")
            e = time.time()
            print(f"Plain addition took {e - s}s at CP{mpc.pid}.")

            # NP Plain addition
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = np_x + 1.1
            print(f"{await mpc.output(res[0])} = {raw_x[0]} + 1.1")
            e = time.time()
            print(f"NP Plain addition took {e - s}s at CP{mpc.pid}.")
            
            # Addition
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = [a + b for a, b in zip(x, y)]
            print(f"{await mpc.output(res[0])} = {raw_x[0]} + {raw_y[0]}")
            e = time.time()
            print(f"Addition took {e - s}s at CP{mpc.pid}.")

            # NP Addition
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = np_x + np_y
            print(f"{await mpc.output(res[0])} = {raw_x[0]} + {raw_y[0]}")
            e = time.time()
            print(f"NP Addition took {e - s}s at CP{mpc.pid}.")

            # Plain multiplication
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = [a * 1.1 for a in x]
            print(f"{await mpc.output(res[0])} = {raw_x[0]} * 1.1")
            e = time.time()
            print(f"Plain multiplication took {e - s}s at CP{mpc.pid}.")

            # NP Plain multiplication
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = np_x * 1.1
            print(f"{await mpc.output(res[0])} = {raw_x[0]} * 1.1")
            e = time.time()
            print(f"NP Plain multiplication took {e - s}s at CP{mpc.pid}.")
            
            # Multiplication
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = mpc.schur_prod(x, y)
            print(f"{await mpc.output(res[0])} = {raw_x[0]} * {raw_y[0]}")
            e = time.time()
            print(f"Multiplication took {e - s}s at CP{mpc.pid}.")

            # NP Multiplication
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = np_x * np_y
            print(f"{await mpc.output(res[0])} = {raw_x[0]} * {raw_y[0]}")
            e = time.time()
            print(f"NP Multiplication took {e - s}s at CP{mpc.pid}.")

            # Rotation
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = x[3:] + x[:3]
            print(f"{await mpc.output(res[0])} = {raw_x[3]}")
            e = time.time()
            print(f"Rotation took {e - s}s at CP{mpc.pid}.")

            # Rotation
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = mpc.np_roll(np_x, 3)
            print(f"{await mpc.output(res[0])} = {raw_x[3]}")
            e = time.time()
            print(f"NP Rotation took {e - s}s at CP{mpc.pid}.")

            # Decryption
            await mpc.transfer(mpc.pid)
            s = time.time()
            rev = await mpc.output(x)
            print(f"{rev[0]} = {raw_x[0]}")
            e = time.time()
            print(f"Decryption took {e - s}s at CP{mpc.pid}.")

            # Decryption
            await mpc.transfer(mpc.pid)
            s = time.time()
            rev = await mpc.output(np_x)
            print(f"{rev[0]} = {raw_x[0]}")
            e = time.time()
            print(f"NP Decryption took {e - s}s at CP{mpc.pid}.")

        raw_x = [[(i % 5) + 1.1 for i in range(n)] for _ in range(128)]
        raw_y = [[(i % 3) + 1.1 for i in range(n)] for _ in range(128)]
        np_raw_x = np.array(raw_x)
        np_raw_y = np.array(raw_y)
        raw_y_t = np_raw_y.T.tolist()
        x = [[secnum(e) for e in row] for row in raw_x]
        y = [[secnum(e) for e in row] for row in raw_y]
        y_t = [[secnum(e) for e in row] for row in raw_y_t]
        np_x = secnum.array(np_raw_x)
        np_y = secnum.array(np_raw_y)
        expected = np_raw_x @ np_raw_y.T

        if "--matmul-via-dot" in args:
            # Matrix multiplication
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = [[mpc.in_prod(a, b) for b in y_t] for a in x]
            print(f"{await mpc.output(res[0][0])} = {expected[0, 0]}")
            e = time.time()
            print(f"Matrix multiplication via dot took {e - s}s at CP{mpc.pid}.")
        
        if "--matmul" in args or "--all" in args:
            # Matrix multiplication
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = mpc.matrix_prod(x, y, True)
            print(f"{await mpc.output(res[0][0])} = {expected[0, 0]}")
            e = time.time()
            print(f"Matrix multiplication took {e - s}s at CP{mpc.pid}.")

        if "--matmul-np" in args or "--all" in args:
            # NP Matrix multiplication
            await mpc.transfer(mpc.pid)
            s = time.time()
            res = np_x @ np_y.T
            print(f"{await mpc.output(res[0, 0])} = {expected[0, 0]}")
            e = time.time()
            print(f"NP Matrix multiplication took {e - s}s at CP{mpc.pid}.")

        raw_x = [[(i % 5) + 1.1 for i in range(n)] for _ in range(32)]
        np_raw_x = np.array(raw_x)
        x = [[secnum(e) for e in row] for row in raw_x]
        np_x = secnum.array(np_raw_x)

        # Raw L2
        dot = np_raw_x * np_raw_x @ np.ones_like(raw_x).T
        distance = (np_raw_x + np_raw_x) @ (-np_raw_x).T + dot + dot.T
        raw_l2 = np.sqrt(distance)
        
        if "--l2" in args or "--all" in args:
            # L2
            await mpc.transfer(mpc.pid)
            s = time.time()
            dot = [[mpc.in_prod(a, b) for b in x] for a in x]
            res = [[_fsqrt(e) for e in row] for row in tqdm.tqdm(dot, "Sqrt")]
            print(f"{await mpc.output(res[0][0])} = {raw_l2[0, 0]}")
            e = time.time()
            print(f"L2 took {e - s}s at CP{mpc.pid}.")

        raw_ones = np.ones_like(raw_x).T
        np_ones = secnum.array(raw_ones)

        if "--l2-np" in args or "--all" in args:
            # NP L2
            await mpc.transfer(mpc.pid)
            s = time.time()
            dot = np_x * np_x @ np_ones
            distance = (np_x + np_x) @ (-np_x).T + dot + dot.T
            res = [[_fsqrt(e) for e in row] for row in tqdm.tqdm(distance, "NP sqrt")]
            print(f"{await mpc.output(res[0][0])} = {raw_l2[0, 0]}")
            e = time.time()
            print(f"NP L2 took {e - s}s at CP{mpc.pid}.")

args = sys.argv[1:]
mpc.run(main(args))
