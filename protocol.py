import os

import param

from mpc import MPCEnv
from custom_types import Zp, Vector, Matrix, TypeOps
from utils import get_cache_path, get_output_path, get_temp_path


def logireg_protocol(mpc: MPCEnv, pid: int, test_run: bool = True) -> bool:
    # TODO: Implemend threading. (See original implementation of this protocol)
    ntop: int = 100

    n0: int = param.NUM_INDS
    m0: int = param.NUM_SNPS
    k: int = param.NUM_DIM_TO_REMOVE

    # Shared variables
    ind: int = 0

    fp_one: Zp = mpc.double_to_fp(1, param.NBIT_K, param.NBIT_F, fid=0)

    pheno = Vector([Zp(0, param.BASE_P) for _ in range(n0)])
    cov = Matrix(n0, param.NUM_COVS)

    if not test_run:
        if (not os.path.exists(get_cache_path(pid, 'input_geno')) or
            not os.path.exists(get_cache_path(pid, 'input_pheno_cov'))):
            print('Initial data sharing results not found:')
            print(f'\t{get_cache_path(pid, "input_geno")}')
            print(f'\t{get_cache_path(pid, "input_pheno_cov")}')
            return False

    print('Initial data sharing results found')

    if not test_run:
        with open(get_cache_path(pid, 'input_geno'), 'rb') as f:
            pheno = mpc.read_vector(f, n0, fid=0)
            cov = mpc.read_matrix(f, n0, param.NUM_COVS, fid=0)
    else:
        with open(get_temp_path(pid, 'input_pheno_cov')) as f:
            pheno = list()
            cov = list()
            for _ in range(n0):
                pheno.append(Zp(int(f.readline()), base=param.BASE_P))
            for _ in range(n0):
                cov_row = list()
                for _ in range(param.NUM_COVS):
                    cov_row.append(Zp(int(f.readline()), base=param.BASE_P))
                cov.append(Vector(cov_row))
            pheno = Vector(pheno)
            cov = Matrix().from_value(Vector(cov))

    print('Phenotypes and covariates loaded')

    gkeep1 = Vector([Zp(0, base=param.BASE_P) for _ in range(m0)])
    
    print('Using locus missing rate filter from a previous run')
    
    if pid == 2:
        with open(get_output_path(pid, 'gkeep1')) as f:
            for i in range(m0):
                gkeep1[i] = Zp(int(f.readline()), base=param.BASE_P)
        mpc.send_elem(gkeep1, 0)
        mpc.send_elem(gkeep1, 1)
    else:
        gkeep1: Vector = mpc.receive_vector(2, msg_len=TypeOps.get_vec_len(m0), fid=0)

    m1: int = int(sum(gkeep1, Zp(0, base=param.BASE_P)))

    ikeep = Vector([Zp(0, base=param.BASE_P) for _ in range(n0)])

    print('Using individual missing rate/het rate filters from a previous run')
    
    if pid == 2:
        with open(get_output_path(pid, 'ikeep')) as f:
            lines = f.readlines()
            assert len(lines) == n0, f'{len(lines)} != {n0}'
            ikeep = Vector([Zp(int(e), base=param.BASE_P) for e in lines])
        mpc.send_elem(ikeep, 0)
        mpc.send_elem(ikeep, 1)
    else:
        ikeep: Vector = mpc.receive_vector(2, msg_len=TypeOps.get_vec_len(n0), fid=0)

    n1: int = int(sum(ikeep, Zp(0, base=param.BASE_P)))
    print(f'n1: {n1}, m1: {m1}')

    print('Filtering phenotypes and covariates')
    pheno = mpc.filter(pheno, ikeep)
    cov = mpc.filter_rows(cov, ikeep)

    gkeep2 = Vector([Zp(0, base=param.BASE_P) for _ in range(m1)])
    
    print('Using MAF/HWE filters from a previous run')
    
    if pid == 2:
        with open(get_output_path(pid, 'gkeep2')) as f:
            for i in range(m1):
                gkeep2[i] = Zp(int(f.readline()), base=param.BASE_P)
        mpc.send_elem(gkeep2, 0)
        mpc.send_elem(gkeep2, 1)
    else:
        gkeep2: Vector = mpc.receive_vector(2, msg_len=TypeOps.get_vec_len(m1), fid=0)

    m2: int = int(sum(gkeep2, Zp(0, base=param.BASE_P)))
    print(f'n1: {n1}, m2: {m2}')

    print('Using CA statistics from a previous run')
    
    gkeep3 = Vector([Zp(0, base=param.BASE_P) for _ in range(m2)])

    if pid == 2:
        cavec = list()
        with open(get_output_path(pid, 'assoc')) as f:
            for i in range(m2):
                val = float(f.readline())
                cavec.append((i, val * val))

        cavec.sort(key=TypeOps.switch_pair, reverse=True)

        print(f'Selected top {ntop} candidates')
        print(f'Top 5 CA stats: {cavec[:5]}')

        for i in range(ntop):
            gkeep3[cavec[i][0]] = Zp(1, base=param.BASE_P)

        mpc.send_elem(gkeep3, 0)
        mpc.send_elem(gkeep3, 1)
    else:
        gkeep3: Vector = mpc.receive_vector(2, msg_len=TypeOps.get_vec_len(m2), fid=0)

    V = Matrix(k, n1)

    print('Using eigenvectors from a previous run')
    if not test_run:
        with open(get_cache_path(pid, 'eigen')):
            V = mpc.read_matrix(k, n1)
    else:
        with open(get_temp_path(pid, 'eigen')) as f:
            V = list()
            for _ in range(k):
                V_row = list()
                for _ in range(n1):
                    V_row.append(Zp(int(f.readline()), base=param.BASE_P))
                V.append(Vector(V_row))
            V = Matrix().from_value(Vector(V))

    # Concatenate covariate matrix and jointly orthogonalize
    cov.transpose(inplace=True)
    V.set_dims(k + param.NUM_COVS, n1)
    if pid > 0:
        for i in range(param.NUM_COVS):
            V[k + i] = cov[i] * fp_one

    print('Finding orthonormal basis for ', cov.get_dims(), param.NUM_COVS, n1)
    V = mpc.orthonormal_basis(V)

    V_mean = Vector([Zp(0, base=param.BASE_P) for _ in range(V.num_rows())])
    fp_denom: Zp = mpc.double_to_fp(1 / V.num_cols(), param.NBIT_K, param.NBIT_F, fid=0)
    
    for i in range(len(V_mean)):
        V_mean[i] = sum(V[i], Zp(0, base=param.BASE_P)) * fp_denom
    mpc.trunc_vec(V_mean)
    
    for i in range(len(V_mean)):
        V[i] += -V_mean[i]

    V_var = mpc.inner_prod(V)
    mpc.trunc_vec(V_var)
    V_var *= fp_denom
    mpc.trunc_vec(V_var)

    print('Calculating fp_sqrt')
    _, V_stdinv = mpc.fp_sqrt(V_var)
    
    mpc.mult_vec

    for i in range(len(V_mean)):
        V[i] = mpc.mult_vec(V[i], V_stdinv[i])
    mpc.trunc(V, k=param.NBIT_K + param.NBIT_F, m=param.NBIT_F)

    gkeep: list = [False for _ in range(m0)]
    for j in range(m0):
        gkeep[j] = (gkeep1[j].value == 1)

    ind: int = 0
    for j in range(m0):
        if gkeep[j]:
            gkeep[j] = (gkeep2[ind].value == 1)
            ind += 1

    ind: int = 0
    for j in range(m0):
        if gkeep[j]:
            gkeep[j] = (gkeep3[ind].value == 1)
            ind += 1

    if not os.path.exists(get_cache_path(pid, 'logi_input')):
        raise NotImplementedError(
            'At this point, logi_input is expected in cache.\n'
            'TODO: Haris. Make it cache agnostic. (See original implementation)')
    
    print('logi_input cache found')
    if not test_run:
        with open(get_cache_path(pid, 'logi_input'), 'br') as f:
            X, X_mask = mpc.beaver_read_from_file(f, ntop, n1)
    else:
        with open(get_temp_path(pid, 'input_pheno_cov')) as f:
            X = list()
            X_mask = list()
            for _ in range(ntop):
                x_row = list()
                for _ in range(n1):
                    x_row.append(Zp(int(f.readline()), base=param.BASE_P))
                X.append(Vector(x_row))
            for _ in range(ntop):
                x_mask_row = list()
                for _ in range(n1):
                    x_mask_row.append(Zp(int(f.readline()), base=param.BASE_P))
                X_mask.append(Vector(x_mask_row))
            X = Matrix().from_value(Vector(X))
            X_mask = Matrix().from_value(Vector(X_mask))

    # TODO: Haris. Implement data shuffling. (See original implementation)

    V, V_mask = mpc.beaver_partition(V)

    pheno, pheno_mask = mpc.beaver_partition(pheno)

    _, _, bx = mpc.parallel_logistic_regression(X, X_mask, V, V_mask, pheno, pheno_mask, 3)

    bx: Vector = mpc.reveal_sym(bx)
    if pid == 2:
        bx_double = mpc.fp_to_double_vec(bx, param.NBIT_K, param.NBIT_F)
        output_path: str = get_output_path(pid, 'logi_coeff')
        with open(output_path, 'w') as f:
            for num in bx_double.value:
                f.write(f'{str(num)}\n')
        print(f'Result written to {output_path}')

    return True
