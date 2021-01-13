import os
from copy import deepcopy

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
        if (not os.path.exists(get_cache_path(mpc.pid, 'input_geno')) or
            not os.path.exists(get_cache_path(mpc.pid, 'input_pheno_cov'))):
            print('Initial data sharing results not found:')
            print(f'\t{get_cache_path(mpc.pid, "input_geno")}')
            print(f'\t{get_cache_path(mpc.pid, "input_pheno_cov")}')
            return False

    print('Initial data sharing results found')

    if not test_run:
        with open(get_cache_path(mpc.pid, 'input_geno'), 'rb') as f:
            pheno = mpc.read_vector(f, n0, fid=0)
            cov = mpc.read_matrix(f, n0, param.NUM_COVS, fid=0)
    else:
        with open(get_temp_path(mpc.pid, 'input_pheno_cov')) as f:
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
    
    if mpc.pid == 2:
        with open(get_output_path(mpc.pid, 'gkeep1')) as f:
            for i in range(m0):
                gkeep1[i] = Zp(int(f.readline()), base=param.BASE_P)
        mpc.send_elem(gkeep1, 0)
        mpc.send_elem(gkeep1, 1)
    else:
        gkeep1: Vector = mpc.receive_vector(2, msg_len=TypeOps.get_vec_len(m0), fid=0)

    m1: int = int(sum(gkeep1, Zp(0, base=param.BASE_P)))

    ikeep = Vector([Zp(0, base=param.BASE_P) for _ in range(n0)])

    print('Using individual missing rate/het rate filters from a previous run')
    
    if mpc.pid == 2:
        with open(get_output_path(mpc.pid, 'ikeep')) as f:
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
    
    if mpc.pid == 2:
        with open(get_output_path(mpc.pid, 'gkeep2')) as f:
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

    if mpc.pid == 2:
        cavec = list()
        with open(get_output_path(mpc.pid, 'assoc')) as f:
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
        with open(get_cache_path(mpc.pid, 'eigen')):
            V = mpc.read_matrix(k, n1)
    else:
        with open(get_temp_path(mpc.pid, 'eigen')) as f:
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
    if mpc.pid > 0:
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

    V_var = mpc.inner_prod(V, fid=0)
    mpc.trunc_vec(V_var)
    V_var *= fp_denom
    mpc.trunc_vec(V_var)

    print('Calculating fp_sqrt')
    _, V_stdinv = mpc.fp_sqrt(V_var)
    
    for i in range(len(V_mean)):
        V[i] = mpc.mult_vec(V[i], V_stdinv[i], fid=0)
    mpc.trunc(V, k=param.NBIT_K + param.NBIT_F, m=param.NBIT_F, fid=0)

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

    if not os.path.exists(get_cache_path(mpc.pid, 'logi_input')):
        raise NotImplementedError(
            'At this point, logi_input is expected in cache.\n'
            'TODO: Haris. Make it cache agnostic. (See original implementation)')
    
    print('logi_input cache found')
    if not test_run:
        with open(get_cache_path(mpc.pid, 'logi_input'), 'br') as f:
            X, X_mask = mpc.beaver_read_from_file(f, ntop, n1)
    else:
        with open(get_temp_path(mpc.pid, 'logi_input')) as f:
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

    V, V_mask = mpc.beaver_partition(V, fid=0)

    pheno, pheno_mask = mpc.beaver_partition(pheno, fid=0)

    _, _, bx = mpc.parallel_logistic_regression(X, X_mask, V, V_mask, pheno, pheno_mask, 3)

    bx: Vector = mpc.reveal_sym(bx)
    if mpc.pid == 2:
        bx_double = mpc.fp_to_double_vec(bx, param.NBIT_K, param.NBIT_F)
        output_path: str = get_output_path(mpc.pid, 'logi_coeff')
        with open(output_path, 'w') as f:
            for num in bx_double.value:
                f.write(f'{str(num)}\n')
        print(f'Result written to {output_path}')

    return True


def gwas_protocol(mpc: MPCEnv) -> bool:
    n0: int = param.NUM_INDS
    m0: int = param.NUM_SNPS
    k: int = param.NUM_DIM_TO_REMOVE
    kp: int = k + param.NUM_OVERSAMPLE

    # Read in SNP list
    snp_pos = Vector([0] * m0)

    with open(param.SNP_POS_FILE) as f:
        for i in range(m0):
            chrom, pos = f.readline().split()
            snp_pos[i] = int(chrom) * 1e9 + int(pos)

    # Useful constants
    two = Zp(2, base=param.BASE_P)
    twoinv = two.inv()

    fp_one: Zp = mpc.double_to_fp(1, param.NBIT_K, param.NBIT_F)

    pheno = Vector([Zp(0, base=param.BASE_P) for _ in range(n0)])
    cov = Matrix(n0, param.NUM_COVS)

    print("Initial data sharing results found")

    with open(get_cache_path(mpc.pid, "input_pheno_cov")) as f:
        pheno: Vector = mpc.read_vector(f, n0)
        cov = mpc.read_matrix(f, n0, param.NUM_COVS)

    print("Phenotypes and covariates loaded")

    gkeep1 = Vector([Zp(0, base=param.BASE_P) for _ in range(m0)])

    if param.SKIP_QC:
        for i in range(m0):
            gkeep1[i] = 1
        print("Locus missing rate filter skipped")
    else:
        history: bool = False
        if mpc.pid == 2:
            history = os.path.exists(get_output_path(mpc.pid, "gkeep1"))
            mpc.send_bool(history, 0)
            mpc.send_bool(history, 1)
        else:
            # ask P2 if gkeep1 has been computed before
            history = mpc.receive_bool(2)

        if history:
            print("Using locus missing rate filter from a previous run")
        
            if mpc.pid == 2:
                with open(get_output_path(mpc.pid, "gkeep1")):
                    for i in range(m0):
                        gkeep1[i] = Zp(int(f.readline()), base=param.BASE_P)

                mpc.send_elem(gkeep1, 0)
                mpc.send_elem(gkeep1, 1)
            else:
                gkeep1: Vector = mpc.receive_vector(2, msg_len=TypeOps.get_vec_len(m0), fid=0)
        else:
            gmiss: Vector = Vector([Zp(0, base=param.BASE_P) for _ in range(m0)])
            
            if os.path.exists(get_cache_path(mpc.pid, "gmiss")):
                print("Locus missing rate cache found")

                with open(get_cache_path(mpc.pid, "gmiss")) as f:
                    gmiss = mpc.read_vector(f, m0)
            else:
                print("Taking a pass to calculate locus missing rates:")

                if mpc.pid > 0:
                    with open(get_cache_path(mpc.pid, "input_geno")) as f:
                        mpc.import_seed(10, int(f.readline()))

                        bsize: int = n0 // 10

                        for i in range(n0):
                            # Load stored Beaver partition
                            mpc.switch_seed(10)
                            tmp_mat = Matrix(3, m0, randomise=True)  # g_mask
                            miss_mask: Vector = mpc.rand_vector(m0, fid=0)
                            mpc.restore_seed(10)

                            if mpc.pid == 2:
                                mpc.skip_data(f, 3, m0)
                                miss: Vector = mpc.read_vector(f, m0)

                            # Recover secret shares from Beaver partition
                            if mpc.pid == 1:
                                miss = miss_mask
                            else:
                                miss += miss_mask

                            # Add to running sum
                            gmiss += miss


                with open(get_cache_path(mpc.pid, "gmiss")) as f:
                    mpc.write_to_file(gmiss, f)

                print("Wrote results to cache")

            print("Locus missing rate filter ... ")

            gmiss_ub: Zp = Zp(n0 * param.GMISS_UB, base=param.BASE_P)
            gkeep1: Vector = mpc.less_than_public(gmiss, gmiss_ub)
            gkeep1: Vector = mpc.reveal_sym(gkeep1)

            if mpc.pid == 2:
                mpc.send_elem(gkeep1, 0)
            elif mpc.pid == 0:
                gkeep1: Vector = mpc.receive_vector(2, TypeOps.get_vec_len(m0))

            if mpc.pid == 2:
                with open(get_output_path(mpc.pid, "gkeep1"), 'w') as f:
                    for i in range(gkeep1.length()):
                        f.write(f'{gkeep1[i]}\n')

    m1: int = sum(gkeep1, Zp(0, base=param.BASE_P)).value
    print("n0:", n0, ",", "m1:", m1)

    print("Filtering SNP position vector")
    snp_pos: Vector = mpc.filter(snp_pos, gkeep1)

    ikeep = Vector([Zp(0, base=param.BASE_P) for _ in range(n0)])

    if param.SKIP_QC:
        for i in range(n0):
            ikeep[i] = 1
        print("Individual missing rate/het rate filters skipped")
    else:
        history: bool = False
        if mpc.pid == 2:
            history = os.path.exists(get_output_path(mpc.pid, "ikeep"))
            mpc.send_bool(history, 0)
            mpc.send_bool(history, 1)
        else:
            # ask P2 if ikeep has been computed before
            history = mpc.receive_bool(2)

        if history:
            print("Using individual missing rate/het rate filters from a previous run")
        
            if mpc.pid == 2:
                with open(get_output_path(mpc.pid, "ikeep")) as f:
                    for i in range(n0):
                        ikeep[i] = Zp(int(f.readline()), base=param.BASE_P)

                mpc.send_elem(ikeep, 0)
                mpc.send_elem(ikeep, 1)
            else:
                mpc.receive_vector(ikeep, 2, TypeOps.get_vec_len(n0))
        else:
            imiss = Vector([Zp(0, param.BASE_P) for _ in range(n0)])
            ihet = Vector([Zp(0, param.BASE_P) for _ in range(n0)])

            if os.path.exists(get_cache_path(mpc.pid, "imiss_ihet")):
                print("Individual missing rate and het rate cache found")

                with open(get_cache_path(mpc.pid, "imiss_ihet")) as f:
                    imiss: Vector = mpc.read_vector(f, n0)
                    ihet: Vector = mpc.read_vector(f, n0)
            else:
                print("Taking a pass to calculate individual missing rates and het rates:")

                if mpc.pid > 0:
                    with open(get_cache_path(mpc.pid, "input_geno")) as f:
                        mpc.import_seed(10, int(f.readline()))
                        bsize: int = n0 // 10

                        for i in range(n0):
                            # Load stored Beaver partition
                            mpc.switch_seed(10)
                            g_mask = Matrix(3, m0, randomise=True)
                            miss_mask: Vector = mpc.random_vector(m0)
                            mpc.restore_seed(10)

                            if mpc.pid == 2:
                                g: Matrix = mpc.read_matrix(f, 3, m0)
                                miss: Vector = mpc.read_vector(f, m0)

                            # Recover secret shares from Beaver partition
                            if mpc.pid == 1:
                                g = g_mask
                                miss = miss_mask
                            else:
                                g += g_mask
                                miss += miss_mask

                            # Add to running sum
                            for j in range(m0):
                                if gkeep1[j] == 1:
                                    imiss[i] += miss[j]
                                    ihet[i] += g[1][j]

                with open(get_cache_path(mpc.pid, "imiss_ihet")) as f:
                    mpc.write_to_file(imiss, fs)
                    mpc.write_to_file(ihet, fs)

                print("Wrote results to cache")

            # Individual missingness filter
            print("Individual missing rate filter ... ")
            imiss_ub: Zp = Zp(m1 * param.IMISS_UB, base=param.BASE_P)
            ikeep: Vector = mpc.less_than_public(imiss, imiss_ub)

            # Individual heterozygosity filter
            print("Individual heterozygosity rate filter ... ")
            ihet_ub_frac: Zp = mpc.double_to_fp(param.HET_UB, param.NBIT_K, param.NBIT_F)
            ihet_lb_frac: Zp = mpc.double_to_fp(param.HET_LB, param.NBIT_K, param.NBIT_F)

            # Number of observed SNPs per individual
            m1_obs = Vector([Zp(0, base=param.BASE_P) for _ in range(n0)])
            if mpc.pid > 0:
                for i in range(n0):
                    m1_obs[i] = -imiss[i]
                    if mpc.pid == 1:
                        m1_obs[i] += m1

            ihet_ub = Vector([Zp(0, base=param.BASE_P) for _ in range(n0)])
            ihet_lb = Vector([Zp(0, base=param.BASE_P) for _ in range(n0)])

            if mpc.pid > 0:
                for i in range(n0):
                    ihet_ub[i] = m1_obs[i] * ihet_ub_frac
                    ihet_lb[i] = m1_obs[i] * ihet_lb_frac
                    ihet[i] *= fp_one

            het_filt: Vector = mpc.less_than(ihet, ihet_ub)
            tmp_vec: Vector = mpc.not_less_than(ihet, ihet_lb)
            het_filt: Vector = mpc.mult_vec(het_filt, tmp_vec)

            ikeep: Vector = mpc.mult_vec(ikeep, het_filt)

            # Reveal samples to be filtered
            mpc.reveal_sym(ikeep)

            if mpc.pid == 2:
                mpc.send_elem(ikeep, 0)
            elif mpc.pid == 0:
                mpc.receive_vector(ikeep, 2, TypeOps.get_vec_len(n0))

            if mpc.pid == 2:
                with open(get_output_path(mpc.pid, "ikeep"), 'w') as f:
                    for i in range(ikeep.length()):
                        f.write(f'{ikeep[i]}\n')

    n1: int = int(sum(ikeep, Zp(0, base=param.BASE_P)))

    print("Filtering phenotypes and covariates")
    pheno: Vector = mpc.filter(pheno, gkeep1)
    cov: Matrix = mpc.filter_rows(cov, ikeep)

    ctrl: Vector = mpc.flip_bit(pheno)

    ctrl, ctrl_mask = mpc.beaver_partition(ctrl, fid=0)

    dosage_sum: Vector = Vector([Zp(0, param.BASE_P) for _ in range(m1)])
    gmiss: Vector = Vector([Zp(0, param.BASE_P) for _ in range(m1)])
    gmiss_ctrl: Vector = Vector([Zp(0, param.BASE_P) for _ in range(m1)])
    dosage_sum_ctrl: Vector = Vector([Zp(0, param.BASE_P) for _ in range(m1)])
    g_count_ctrl = Matrix(3, m1)
    n1_ctrl = Zp(0, param.BASE_P)

    if (os.path.exists(get_cache_path(mpc.pid, "geno_stats"))):
        print("Genotype statistics cache found")

        with open(get_cache_path(mpc.pid, "geno_stats")) as f:
            gmiss: Vector = mpc.read_vector(f, m1, fid=0)
            gmiss_ctrl: Vector = mpc.read_vector(f, m1, fid=0)
            dosage_sum: Vector = mpc.read_vector(f, m1, fid=0)
            dosage_sum_ctrl: Vector = mpc.read_vector(f, m1, fid=0)
            g_count_ctrl: Matrix = mpc.read_matrix(f, 3, m1, fid=0)
            n1_ctrl = Zp(int(f.readline()), param.BASE_P)
    else:
        print("Taking a pass to calculate genotype statistics:")

        with open(get_cache_path(mpc.pid, "input_geno")) as f:
            if mpc.pid > 0:
                mpc.import_seed(10, int(f.readline()))
            else:
                for p in range(1, 3):
                    mpc.import_seed(10 + p, int(f.readline()))

            report_bsize: int = n1 // 10
            bsize: int = param.PITER_BATCH_SIZE

            # Containers for batching the computation
            g = list()
            g_mask = list()
            Vec<ZZ_p> ctrl_vec, ctrl_mask_vec;
            g.set_length(3);
            g_mask.set_length(3);
            dosage = Matrix(bsize, m1)
            dosage_mask = Matrix(bsize, m1)
            miss = Matrix(bsize, m1)
            miss_mask = Matrix(bsize, m1)

            for k in range(3):
                g.append(Matrix(bsize, m1))
                g_mask.append(Matrix(bsize, m1))

            ctrl_vec = Vector([Zp(0, param.BASE_P) for _ in range(bsize)])
            ctrl_mask_vec = Vector([Zp(0, param.BASE_P) for _ in range(bsize)])

            ind: int = -1

            for i in range(n1):
                ind += 1

                Mat<ZZ_p> g0, g0_mask
                Vec<ZZ_p> miss0, miss0_mask;

                while (ikeep[ind] != 1):
                    if mpc.pid > 0:
                        mpc.skip_data(f, 3, m0)
                        mpc.skip_data(f, m0)

                        mpc.switch_seed(10)
                        g0_mask = Matrix(3, m0, randomise=True)
                        miss0_mask: Vector = mpc.random_vector(m0)
                        mpc.restore_seed(10)
                    else:
                        for p in range(1, 3):
                            mpc.switch_seed(10 + p)
                            g0_mask = Matrix(3, m0, randomise=True)
                            miss0_mask: Vector = mpc.random_vector(m0)
                            mpc.restore_seed(10 + p)

                    ind += 1

                if mpc.pid > 0:
                    g0: Matrix = mpc.read_matrix(f, 3, m0)
                    miss0: Vector = mpc.read_vector(f, m0)

                    mpc.switch_seed(10)
                    g0_mask = Matrix(3, m0, randomise=True)
                    miss0_mask: Vector = mpc.random_vector(m0)
                    mpc.restore_seed(10)
                else:
                    g0 = Matrix(3, m0)
                    g0_mask = Matrix(3, m0)
                    miss0 = Vector([Zp(0, param.BASE_P) for _ in range(m0)])
                    miss0_mask = Vector([Zp(0, param.BASE_P) for _ in range(m0)])

                    for p in range(1, 3):
                        mpc.switch_seed(10 + p)
                        tmp_mat = Matrix(3, m0, randomise=True)
                        tmp_vec = mpc.random_vector(m0)
                        mpc.restore_seed(10 + p)

                        g0_mask += tmp_mat
                        miss0_mask += tmp_vec
            

            # Filter out loci that failed missing rate filter
            ind2: int = 0
            for j in range(m0):
                if gkeep1[j] == 1:
                    for k in range(3):
                        g[k][i % bsize][ind2] = deepcopy(g0[k][j])
                        g_mask[k][i % bsize][ind2] = deepcopy(g0_mask[k][j])
                    miss[i % bsize][ind2] = deepcopy(miss0[j])
                    miss_mask[i % bsize][ind2] = deepcopy(miss0_mask[j])
                    ind2 += 1

            dosage[i % bsize] = g[1][i % bsize] + g[2][i % bsize] * 2
            dosage_mask[i % bsize] = g_mask[1][i % bsize] + g_mask[2][i % bsize] * 2

            ctrl_vec[i % bsize] = deepcopy(ctrl[i])
            ctrl_mask_vec[i % bsize] = deepcopy(ctrl_mask[i])

            # Update running sums
            if mpc.pid > 0:
                n1_ctrl += ctrl_mask[i]
                gmiss += miss_mask[i % bsize]
                dosage_sum += dosage_mask[i % bsize]

                if mpc.pid == 1:
                    n1_ctrl += ctrl[i]
                    gmiss += miss[i % bsize]
                    dosage_sum += dosage[i % bsize]

            if (i % bsize == bsize - 1 or i == n1 - 1):
                if i % bsize < bsize - 1:
                    new_bsize: int = (i % bsize) + 1

                    for k in range(3):
                        g[k].set_dims(new_bsize, m1)
                        g_mask[k].set_dims(new_bsize, m1)

                    dosage.set_dims(new_bsize, m1)
                    dosage_mask.set_dims(new_bsize, m1)
                    miss.set_dims(new_bsize, m1)
                    miss_mask.set_dims(new_bsize, m1)
                    ctrl_vec.set_length(new_bsize)
                    ctrl_mask_vec.set_length(new_bsize)

                gmiss_ctrl: Vector = mpc.beaver_mult_vec(
                    ctrl_vec, ctrl_mask_vec, miss, miss_mask)
                dosage_sum_ctrl: Vector = mpc.beaver_mult_vec(
                    ctrl_vec, ctrl_mask_vec, dosage, dosage_mask)
                for k in range(3):
                    g_count_ctrl[k] = mpc.beaver_mult_vec(ctrl_vec, ctrl_mask_vec, g[k], g_mask[k])

        gmiss_ctrl: Vector = mpc.beaver_reconstruct(gmiss_ctrl)
        dosage_sum_ctrl: Vector = mpc.beaver_reconstruct(dosage_sum_ctrl)
        g_count_ctrl: Vector = mpc.beaver_reconstruct(g_count_ctrl)

        # Write to cache
        with open(get_cache_path(mpc.pid, "geno_stats")) as f:
            mpc.write_to_file(gmiss, f)
            mpc.write_to_file(gmiss_ctrl, f)
            mpc.write_to_file(dosage_sum, f)
            mpc.write_to_file(dosage_sum_ctrl, f)
            mpc.write_to_file(g_count_ctrl, f)
            mpc.write_to_file(n1_ctrl, f)

            print("Wrote results to cache")

    # SNP MAF filter
    print("Locus minor allele frequency (MAF) filter ... ")
    maf_lb: Zp = mpc.double_to_fp(param.MAF_LB, param.NBIT_K, param.NBIT_F, fid=0)
    maf_ub: Zp = mpc.double_to_fp(param.MAF_UB, param.NBIT_K, param.NBIT_F, fid=0)

    dosage_tot = Vector()
    dosage_tot_ctrl = Vector()
    if mpc.pid > 0:
        dosage_tot = -gmiss
        dosage_tot_ctrl = -gmiss_ctrl
        dosage_tot += Zp(n1, param.BASE_P)
        dosage_tot_ctrl += n1_ctrl
        dosage_tot *= 2
        dosage_tot_ctrl *= 2
    else:
        dosage_tot.set_length(m1)
        dosage_tot_ctrl.set_length(m1)

    print("Calculating MAFs ... ")
    maf = Vector()
    maf_ctrl = Vector()
    if (os.path.exists(get_cache_path(mpc.pid, "maf"))):
        print("maf cache found")
        with open(get_cache_path(mpc.pid, "maf")) as f:
            maf: Vector = mpc.read_vector(f, dosage_tot.length())
            maf_ctrl: Vector = mpc.read_vector(f, dosage_tot_ctrl.length())
    else:
        maf: Vector = mpc.fp_div(dosage_sum, dosage_tot)
        maf_ctrl: Vector = mpc.fp_div(dosage_sum_ctrl, dosage_tot_ctrl)

        with open(get_cache_path(mpc.pid, "maf"), 'wb') as f:
            mpc.write_to_file(maf, fs)
            mpc.write_to_file(maf_ctrl, fs)
    print("done. ")

    Maf = Vector()  # MAJOR allele freq
    maf_ctrl = Vector()
    if mpc.pid > 0:
        Maf = -maf
        Maf_ctrl = -maf_ctrl
        Maf = mpc.add_public(Maf, fp_one)
        Maf_ctrl = mpc.add_public(Maf_ctrl, fp_one)
    else:
        Maf.set_length(m1)
        Maf_ctrl.set_length(m1)

    # Variance based on Bernoulli distribution over each allele
    g_var_bern: Vector = mpc.mult_vec(maf, Maf)
    mpc.trunc_vec(g_var_bern)

    gkeep2 = Vector([Zp(0, param.BASE_P) for _ in range(m1)])

    if param.SKIP_QC:
        for i in range(m1):
            gkeep2[i] = 1
        print("SNP MAF/HWE filters skipped")
    else:
        history: bool = False
        if mpc.pid == 2:
            history = os.path.exists(get_output_path(mpc.pid, "gkeep2"))
            mpc.send_bool(history, 0)
            mpc.send_bool(history, 1)
        else:
            # ask P2 if gkeep2 has been computed before
            history = mpc.receive_bool(2)

        if history:
            print("Using MAF/HWE filters from a previous run")
        
            if mpc.pid == 2:
                with open(get_output_path(mpc.pid, "gkeep2")) as f:
                    for i in range(m1):
                        gkeep2[i] = Zp(int(f.readline()), param.BASE_P)

                mpc.send_elem(gkeep2, 0)
                mpc.send_elem(gkeep2, 1)
            else:
                mpc.receive_vector(gkeep2, 2, TypeOps.get_vec_len(m1))
        else:
            gkeep2: Vector = mpc.less_than_public(maf, maf_ub)
            tmp_vecmpc.not_less_than_public(maf, maf_lb)
            gkeep2: Vector = mpc.mult_vec(gkeep2, tmp_vec)

            print("Locus Hardy-Weinberg equilibrium (HWE) filter ... ")
            hwe_ub: Zp = mpc.double_to_fp(param.HWE_UB, param.NBIT_K, param.NBIT_F)  # p < 1e-7
            
            # Calculate expected genotype distribution in control group
            g_exp_ctrl = Matrix(3, m1)

            g_exp_ctrl[0] = mpc.mult_vec(Maf_ctrl, Maf_ctrl)
            g_exp_ctrl[1] = mpc.mult_vec(Maf_ctrl, maf_ctrl)
            if mpc.pid > 0:
                g_exp_ctrl[1] *= 2
            g_exp_ctrl[2] = mpc.mult_vec(maf_ctrl, maf_ctrl)

            for i in range(3):
                g_exp_ctrl[i] = mpc.mult_vec(g_exp_ctrl[i], dosage_tot_ctrl)

        g_exp_ctrl *= twoinv  # dosage_tot_ctrl is twice the # individuals we actually want

        mpc.trunc(g_exp_ctrl, k=param.NBIT_K + param.NBIT_F, m=param.NBIT_F, fid=0)

        print("\tCalculated expected genotype counts, ")

        hwe_chisq = Vector([Zp(0, param.BASE_P) for _ in range(m1)])

        if (os.path.exists(get_cache_path(mpc.pid, "hwe"))):
            print("HWE cache found")
            with open(get_cache_path(mpc.pid, "hwe")) as f:
                hwe_chisq: Vector = mpc.read_vector(f, m1)
        else:
            for i in range(3):
                diff: Vector = [Zp(0, param.BASE_P) for _ in range(m1)]
                if mpc.pid > 0:
                    diff = g_count_ctrl[i] * fp_one - g_exp_ctrl[i]

                diff = mpc.mult_vec(diff, diff)
                mpc.trunc_vec(diff)

                tmp_vec = mpc.fp_div(diff, g_exp_ctrl[i], fid=0)
                hwe_chisq += tmp_vec

            with open(get_cache_path(mpc.pid, "hwe"), 'wb') as f:
                mpc.write_to_file(hwe_chisq, f)

        hwe_filt: Vector = mpc.less_than_public(hwe_chisq, hwe_ub)
        gkeep2 = mpc.mult_vec(gkeep2, hwe_filt)

        # Reveal which SNPs to discard 
        mpc.reveal_sym(gkeep2)
            
        if mpc.pid == 2:
            mpc.send_elem(gkeep2, 0)
        elif mpc.pid == 0:
            gkeep2 = mpc.receive_vector(2, TypeOps.get_vec_len(m1))

        if mpc.pid == 2:
            with open(get_output_path(mpc.pid, "gkeep2"), 'w') as f:
                for i in range(len(gkeep2)):
                    f.write(f'{gkeep2[i]}\n')

    m2: int = int(sum(gkeep2, Zp(0, param.BASE_P)))
    print("n1: ", n1, ", ", "m2: ", m2)

    print("Filtering genotype statistics")
    g_var_bern = mpc.filter(g_var_bern, gkeep2)
    maf = mpc.filter(maf, gkeep2)
    snp_pos = mpc.filter(snp_pos, gkeep2)

    g_std_bern_inv = Vector()
    if (os.path.exists(get_cache_path(mpc.pid, "stdinv_bern"))):
        print("Genotype standard deviation cache found")

        with open(get_cache_path(mpc.pid, "stdinv_bern")) as f:
            g_std_bern_inv = mpc.read_vector(f, len(g_var_bern))

    else:
        print("Calculating genotype standard deviations (inverse)")

        tmp_vec = mpc.fp_sqrt(g_std_bern_inv, g_var_bern)

        with open(get_cache_path(mpc.pid, "stdinv_bern"), 'wb') as f:
            mpc.write_to_file(g_std_bern_inv, f);

    g_mean = Vector([Zp(0, param.BASE_P) for _ in range(m2)])
    if mpc.pid > 0:
        g_mean = maf * 2

    print("Starting population stratification analysis")

    selected: list = [0] * m2  # 1 selected, 0 unselected, -1 TBD
    to_process: list = [False] * m2

    for i in range(m2):
        selected[i] = -1

    dist_thres: int = param.LD_DIST_THRES
    
    prev: int = -1
    for i in range(m2):
        selected[i] = 0
        if prev < 0 or snp_pos[i] - prev > dist_thres:
            selected[i] = 1
            prev = snp_pos[i]

    # At this point "selected" contains the SNP filter for PCA, shared across all parties
    m3: int = 0
    for i in range(len(selected)):
        if selected[i] == 1:
            m3 += 1

    print("SNP selection complete: ", m3, " / ", m2, " selected")

    # Cache the reduced G for PCA
    if (os.path.exists(get_cache_path(mpc.pid, "pca_input"))):
        print("pca_input cache found")
    else:
        gkeep3: list = [False] * m0
        for j in range(m0):
            gkeep3[j] = (gkeep1[j] == 1)

        ind = 0
        for j in range(m0):
            if gkeep3[j]:
                gkeep3[j] = (gkeep2[ind] == 1)
                ind += 1

        ind = 0
        for j in range(m0):
            if gkeep3[j]:
                gkeep3[j] = (selected[ind] == 1)
                ind += 1

        with open(get_cache_path(mpc.pid, "input_geno")) as f_geno:
            if mpc.pid > 0:
                mpc.import_seed(10, f_geno)
            else:
                for p in range(1, 3):
                    mpc.import_seed(10 + p, f_geno)

            bsize: int = n1 // 10

            print("Caching input data for PCA:")

            with open(get_cache_path(mpc.pid, "pca_input")) as f_pca:
                ind = -1
                for i in range(n1):
                    ind += 1

                    g0 = Matrix()
                    g0_mask = Matrix()
                    miss0 = Vector()
                    miss0_mask = Vector()

                    while ikeep[ind] != 1:
                        if mpc.pid > 0:
                            mpc.skip_data(f_geno, 3, m0)
                            mpc.skip_data(f_geno, m0)

                            mpc.switch_seed(10)
                            g0_mask = Matrix(3, m0, randomise=True)
                            miss0_mask = mpc.random_vector(m0)
                            mpc.restore_seed(10)
                        else:
                            for p in range(1, 3):
                                mpc.switch_seed(10 + p)
                                g0_mask = Matrix(3, m0, randomise=True)
                                miss0_mask = mpc.random_vector(m0)
                                mpc.restore_seed(10 + p)
                        ind += 1

                    if mpc.pid > 0:
                        g0 = mpc.read_matrix(f_geno, 3, m0)
                        miss0 = mpc.read_matrix(f_geno, m0)

                        mpc.switch_seed(10)
                        g0_mask = Matrix(3, m0, randomise=True)
                        miss0_mask = mpc.random_vector(m0)
                        mpc.restore_seed(10)
                    else:
                        g0 = Matrix(3, m0)
                        g0_mask = Matrix(3, m0)
                        miss0 = Vector([Zp(0, param.BASE_P) for _ in range(m0)])
                        miss0_mask = Vector([Zp(0, param.BASE_P) for _ in range(m0)])

                        for p in range(3):
                            mpc.switch_seed(10 + p)
                            tmp_mat = Matrix(3, m0, randomise=True)
                            tmp_vec = mpc.random_vector(m0)
                            mpc.restore_seed(10 + p)

                            g0_mask += tmp_mat
                            miss0_mask += tmp_vec
                    
                    # Filter out loci that failed missing rate filter
                    g = Matrix(3, m3)
                    g_mask = Matrix(3, m3)
                    miss = Vector([Zp(0, param.BASE_P) for _ in range(m3)])
                    miss_mask = Vector([Zp(0, param.BASE_P) for _ in range(m3)])
                    ind2: int = 0
                    for j in range(m0):
                        if gkeep3[j]:
                            for k in range(3):
                                g[k][ind2] = g0[k][j]
                                g_mask[k][ind2] = g0_mask[k][j]
                            miss[ind2] = miss0[j]
                            miss_mask[ind2] = miss0_mask[j]
                            ind2 += 1

                    dosage: Vector = g[1] + g[2] * 2
                    dosage_mask: Vector = g_mask[1] + g_mask[2] * 2

                    mpc.beaver_write_to_file(dosage, dosage_mask, f_pca)
                    mpc.beaver_write_to_file(miss, miss_mask, f_pca)

    g_mean_pca: Vector = mpc.filter(g_mean, selected)
    g_stdinv_pca: Vector = mpc.Filter(g_std_bern_inv, selected)

    g_mean_pca_mask: Vector = mpc.beaver_partition(g_mean_pca)
    g_stdinv_pca_mask: Vector = mpc.beaver_partition(g_stdinv_pca)

    # Pass 2: Random sketch
    Y_cur = Matrix(kp, m3)

    if (os.path.exists(get_cache_path(mpc.pid, "sketch"))):
        print("sketch cache found")
        with open(get_cache_path(mpc.pid, "sketch")) as f:
            kp = int(f.readline())
            Y_cur: Matrix = mpc.read_matrix(ifs, kp, m3)
    else:
        Y_cur_adj = Matrix(kp, m3)
        bucket_count: list = [0] * kp

        with open(get_cache_path(mpc.pid, "pca_input")) as f:
            for cur in range(n1):
                # Count sketch (use global PRG)
                mpc.switch_seed(-1)
                bucket_index: int = rand_int(0, kp - 1)
                rand_sign: int = rand_int(0, 1) * 2 - 1
                mpc.restore_seed(-1)

                g, g_mask = mpc.beaver_read_from_file(f, m3)
                miss, miss_mask = mpc.beaver_read_from_file(f, m3)

                # Flip miss bits so it points to places where g_mean should be subtracted
                mpc.beaver_flip_bit(miss, miss_mask)

                # Update running sum
                if mpc.pid > 0:
                    Y_cur[bucket_index] += g_mask * rand_sign
                    if mpc.pid == 1:
                        Y_cur[bucket_index] += g * rand_sign

                # Update adjustment factor
                miss *= rand_sign
                miss_mask *= rand_sign
                Y_cur_adj[bucket_index] = mpc.beaver_mult_elem(miss, miss_mask, g_mean_pca, g_mean_pca_mask)

                bucket_count[bucket_index] += 1

        # Subtract the adjustment factor
        Y_cur_adj: Matrix = mpc.beaver_reconstruct(Y_cur_adj)
        if mpc.pid > 0:
            Y_cur = Y_cur * fp_one - Y_cur_adj

        # Get rid of empty buckets and normalize nonempty ones
        empty_slot: int = 0
        for i in range(kp):
            if bucket_count[i] > 0:
                fp_count_inv: Zp = mpc.double_to_fp(
                    1 / bucket_count[i], param.NBIT_K, param.NBIT_F)
                Y_cur[empty_slot] = Y_cur[i] * fp_count_inv
                empty_slot += 1
        kp: int = empty_slot
        Y_cur.set_dims(kp, m3)
        mpc.trunc(Y_cur)

        with open(get_cache_path(mpc.pid, "sketch"), 'wb') as f:
            f.write(bytes([kp]))
            if mpc.pid > 0:
                mpc.write_to_file(Y_cur, f)

    Y_cur_mask: Matrix = mpc.beaver_partition(Y_cur)

    print(f"Initial sketch obtained, starting power iteration (num iter = {param.NUM_POWER_ITER})")

    gQ = Matrix()

    if (os.path.exists(get_cache_path(mpc.pid, "piter"))):
        print("piter cache found")
        with open(get_cache_path(mpc.pid, "piter")) as f:
            gQ = mpc.read_from_file(f, n1, kp)
    else:
        # Divide by standard deviation
        Y = Matrix(kp, m3)

        for i in range(kp):
            Y[i] = mpc.beaver_mult_elem(Y_cur[i], Y_cur_mask[i], g_stdinv_pca, g_stdinv_pca_mask)

        Y = mpc.beaver_reconstruct(Y)
        mpc.trunc(Y)

        # Calculate orthonormal bases of Y
        Q: Matrix = mpc.orthonormal_basis(Y)

        gQ_adj = Matrix()
        Q_mask = Matrix()
        Q_scaled = Matrix()
        Q_scaled_mask = Matrix()
        Q_scaled_gmean = Matrix()
        Q_scaled_gmean_mask = Matrix()

        # Power iteration
        for pit in range(param.NUM_POWER_ITER + 1):
            # This section is ran before each iteration AND once after all iterations
            Q, Q_mask = mpc.beaver_partition(Q)

            # Normalize Q by standard deviations
            Q_scaled = Matrix(kp, m3)
            for i in range(kp):
                Q_scaled[i] = mpc.beaver_mult_vec(Q[i], Q_mask[i], g_stdinv_pca, g_stdinv_pca_mask)

        Q_scaled = mpc.beaver_reconstruct(Q_scaled)
        mpc.trunc(Q_scaled)

        Q_scaled, Q_scaled_mask = mpc.beaver_partition(Q_scaled)

        # Pre-multiply with g_mean to simplify calculation of centering matrix
        Q_scaled_gmean = Matrix(kp, m3)
        for i in range(kp):
            Q_scaled_gmean[i] = mpc.beaver_mult_vec(Q_scaled[i], Q_scaled_mask[i], g_mean_pca, g_mean_pca_mask)
        Q_scaled_gmean = mpc.beaver_reconstruct(Q_scaled_gmean)
        mpc.trunc(Q_scaled_gmean)

        Q_scaled.transpose(inplace=True)  # m3-by-kp
        # transpose(, Q_scaled_mask); // m3-by-kp, unlike mpc.Transpose, P0 also transposes
        Q_scaled_mask.transpose(inplace=True)
        Q_scaled_gmean.transpose(inplace=True)  # m3-by-kp
        Q_scaled_gmean, Q_scaled_gmean_mask =  mpc.beaver_partition(Q_scaled_gmean)

        bsize: int = param.PITER_BATCH_SIZE

        g = Matrix(bsize, m3)
        g_mask = Matrix(bsize, m3)
        miss = Matrix(bsize, m3)
        miss_mask = Matrix(bsize, m3)
        
        # Pass 1
        gQ = Matrix(n1, kp)
        gQ_adj = Matrix(n1, kp)

        with open(get_cache_path(mpc.pid, "pca_input")) as f:
            for cur in range(n1):
                g[cur % bsize] = mpc.beaver_read_from_file(g_mask[cur % bsize], ifs, m3)
                miss[cur % bsize] = mpc.beaver_read_from_file(miss_mask[cur % bsize], ifs, m3)
                mpc.beaver_flip_bit(miss[cur % bsize], miss_mask[cur % bsize])

                if cur % bsize == bsize - 1:
                    tmp_mat = Matrix(bsize, kp)
                    tmp_mat = mpc.beaver_mult(g, g_mask, Q_scaled, Q_scaled_mask)
                    for i in range(bsize):
                        gQ[cur-(bsize-1)+i] = tmp_mat[i]

                tmp_mat = Matrix(bsize, kp)
                tmp_mat = mpc.beaver_mult(miss, miss_mask, Q_scaled_gmean, Q_scaled_gmean_mask)
                for i in range(bsize):
                    gQ_adj[cur-(bsize-1)+i] = tmp_mat[i]

        remainder: int = n1 % bsize
        if remainder > 0:
            g.set_dims(remainder, m3)
            g_mask.set_dims(remainder, m3)
            miss.set_dims(remainder, m3)
            miss_mask.set_dims(remainder, m3)

            tmp_mat = mpc.beaver_mult(g, g_mask, Q_scaled, Q_scaled_mask)
            for i in range(remainder):
                gQ[n1-remainder+i] = tmp_mat[i]

            tmp_mat = Matrix(remainder, kp)
            tmp_mat = mpc.beaver_mult(miss, miss_mask, Q_scaled_gmean, Q_scaled_gmean_mask)
            for i in range(remainder):
                gQ_adj[n1-remainder+i] = tmp_mat[i]

        gQ = mpc.beaver_reconstruct(gQ)
        gQ_adj = mpc.beaver_reconstruct(gQ_adj)
        if mpc.pid > 0:
            gQ -= gQ_adj

        if pit == param.NUM_POWER_ITER:  # Quit if all iterations are performed
            break

        gQ.transpose(inplace=True)  # kp-by-n1
        Q = mpc.orthonormal_basis(gQ)
        Q.transpose(inplace=True)  # n1-by-kp

        Q, Q_mask = mpc.beaver_partition(Q)

        gQ = Matrix(kp, m3)
        gQ_adj = Matrix(kp, m3)

        g = Matrix(bsize, m3)
        g_mask = Matrix(bsize, m3)
        miss = Matrix(bsize, m3)
        miss_mask = Matrix(bsize, m3)

        Qsub = Matrix(bsize, kp)
        Qsub_mask = Matrix(bsize, kp)

        # Pass 2
        with open(get_cache_path(mpc.pid, "pca_input")) as f:
            for cur in range(n1):
                g[cur % bsize] = mpc.beaver_read_from_file(g_mask[cur % bsize], f, m3)
                miss[cur % bsize] = mpc.beaver_read_from_file(miss_mask[cur % bsize], f, m3)
                mpc.beaver_flip_bit(miss[cur % bsize], miss_mask[cur % bsize])

                Qsub[cur % bsize] = Q[cur]
                Qsub_mask[cur % bsize] = Q_mask[cur]

                if cur % bsize == bsize - 1:
                    Qsub.transpose(inplace=True)
                    Qsub_mask.transpose(inplace=True)

                    gQ = mpc.beaver_mult(Qsub, Qsub_mask, g, g_mask)
                    gQ_adj = mpc.beaver_mult(Qsub, Qsub_mask, miss, miss_mask)

                    Qsub.set_dims(bsize, kp)
                    Qsub_mask.set_dims(bsize, kp)

        remainder: int = n1 % bsize
        if remainder > 0:
            g.set_dims(remainder, m3)
            g_mask.set_dims(remainder, m3)
            miss.set_dims(remainder, m3)
            miss_mask.set_dims(remainder, m3)
            Qsub.set_dims(remainder, kp)
            Qsub_mask.set_dims(remainder, kp)
            
            Qsub.transpose(inplace=True)
            Qsub_mask.transpose(inplace=True)

            gQ = mpc.beaver_mult(Qsub, Qsub_mask, g, g_mask)
            gQ_adj = mpc.beaver_mult(Qsub, Qsub_mask, miss, miss_mask)

        gQ = mpc.beaver_reconstruct(gQ)
        gQ_adj = mpc.beaver_reconstruct(gQ_adj)

        gQ_adj, gQ_adj_mask = mpc.beaver_partition(gQ_adj)

        gQ_adj_gmean = Matrix(kp, m3)
        for i in range(kp):
            gQ_adj_gmean[i] = mpc.beaver_mult_elem(
                gQ_adj[i], gQ_adj_mask[i], g_mean_pca, g_mean_pca_mask)
        gQ_adj_gmean = mpc.beaver_reconstruct(gQ_adj_gmean)
        mpc.trunc(gQ_adj_gmean)

        if mpc.pid > 0:
            gQ -= gQ_adj_gmean
        gQ, gQ_mask = mpc.beaver_partition(gQ)

        gQ_scaled.set_dims(kp, m3)
        gQ_scaled.clear()
        
        for i in range(kp):
            gQ_scaled[i] = mpc.beaver_mult_elem(gQ[i], gQ_mask[i], g_stdinv_pca, g_stdinv_pca_mask);
        gQ_scaled = mpc.beaver_reconstruct(gQ_scaled)
        mpc.trunc(gQ_scaled)

        Q = mpc.orthonormal_basis(gQ_scaled)

        with open(get_cache_path(mpc.pid, "piter")) as f:
            if mpc.pid > 0:
                mpc.write_to_file(gQ, f)

    print("Power iteration complete")

    Z: Matrix = deepcopy(gQ)

    print("Data projected to subspace")
    V = Matrix(k, n1)

    # Eigendecomposition
    if (os.path.exists(get_cache_path(mpc.pid, "eigen"))):
        print("eigen cache found")
        with open(get_cache_path(mpc.pid, "eigen")) as f:
            V = mpc.read_from_file(f, k, n1)
    else:
        fp_m2_inv: Zp = mpc.double_to_fp(1 / m2, param.NBIT_K, param.NBIT_F)
        Z *= fp_m2_inv
        mpc.trunc(Z)

        Z.transpose(inplace=True) // kp-by-n1

        Z, Z_mask = mpc.beaver_partition(Z)

        # Form covariance matrix
        Z_gram = Matrix(kp, kp)
        for i in range(kp):
            Z_gram[i] = mpc.beaver_mult(Z, Z_mask, Z[i], Z_mask[i])
        Z_gram = mpc.beaver_reconstruct(Z_gram)
        mpc.trunc(Z_gram)

        print("Constructed reduced eigenvalue problem")

        U, L = mpc.EigenDecomp(Z_gram)

        # Select top eigenvectors and eigenvalues
        U.set_dims(k, kp)
        L.set_length(k)

        print("Selected K eigenvectors")
        # Recover singular vectors
        U, U_mask = mpc.beaver_partition(U)

        V = mpc.beaver_mult(U, U_mask, Z, Z_mask)
        V = mpc.beaver_reconstruct(V)
        mpc.trunc(V)

        with open(get_cache_path(mpc.pid, "eigen")) as f:
            if mpc.pid > 0:
                mpc.write_to_file(V, f)

    # Concatenate covariate matrix and jointly orthogonalize
    cov.transpose(inplace=True)
    V.set_dims(k + param.NUM_COVS, n1)
    if mpc.pid > 0:
        for i in range(param.NUM_COVS):
            V[k + i] = cov[i] * fp_one
    V = mpc.orthonormal_basis(V)

    V, V_mask = mpc.beaver_partition(V)

    print("Bases for top singular vectors and covariates calculated")
    # Pass 4: Calculate GWAS statistics */

    pheno, pheno_mask = mpc.beaver_partition(pheno)

    Vp: Vector = mpc.beaver_mult_vec(V, V_mask, pheno, pheno_mask)
    Vp = mpc.beaver_reconstruct(Vp)
    Vp, Vp_mask = mpc.beaver_partition(Vp)
    
    VVp: Vector = mpc.beaver_mult(Vp, Vp_mask, V, V_mask)
    VVp = mpc.beaver_reconstruct(VVp)
    mpc.trunc_vec(VVp)

    VVp, VVp_mask = mpc.beaver_partition(VVp)

    p_hat: Vector = pheno * fp_one - VVp
    p_hat_mask: Vector = pheno_mask * fp_one - VVp_mask

    print("Phenotypes corrected")

    V_sum = Vector([Zp(0, param.BASE_P) for _ in range(k + param.NUM_COVS)])
    V_sum_mask = Vector([Zp(0, param.BASE_P) for _ in range(k + param.NUM_COVS)])
    for i in range(k + param.NUM_COVS):
        for j in range(n1):
            V_sum[i] += V[i][j]
            V_sum_mask[i] += V_mask[i][j]

    u: Vector = mpc.beaver_mult(V_sum, V_sum_mask, V, V_mask)
    u = mpc.beaver_reconstruct(u)
    mpc.trunc_vec(u)
    if mpc.pid > 0:
        u *= -1
        u = mpc.add_public(u, fp_one)
    u, u_mask = mpc.beaver_partition(u)

    print("Allocating sx, sxx, sxp, B ... ")

    sx = Vector([Zp(0, base=param.BASE_P) for _ in range(m2)])
    sxx = Vector([Zp(0, base=param.BASE_P) for _ in range(m2)])
    sxp = Vector([Zp(0, base=param.BASE_P) for _ in range(m2)])
    B = Matrix(k + param.NUM_COVS, m2)

    print("done.")

    if (os.path.exists(get_cache_path(mpc.pid, "gwas_stats"))):
        print("GWAS statistics cache found")
        with open(get_cache_path(mpc.pid, "gwas_stats")) as f:
            sx = mpc.read_from_file(f, m2)
            sxx = mpc.read_from_file(f, m2)
            sxp = mpc.read_from_file(f, m2)
            B = mpc.read_from_file(f, k + param.NUM_COVS, m2)
    else:
        with open(get_cache_path(mpc.pid, "input_geno")) as f:
            if mpc.pid > 0:
                mpc.import_seed(10, f)
            else:
                for p in range(1, 3):
                    mpc.import_seed(10 + p, int(f.readline()))

            bsize: int = param.PITER_BATCH_SIZE

            print("Allocating batch variables ... ")

            dosage = Matrix(bsize, m2)
            dosage_mask = Matrix(bsize, m2)

            u_vec = Vector([Zp(0, param.BASE_P) for _ in range(bsize)])
            u_mask_vec = Vector([Zp(0, param.BASE_P) for _ in range(bsize)])
            p_hat_vec = Vector([Zp(0, param.BASE_P) for _ in range(bsize)])
            p_hat_mask_vec = Vector([Zp(0, param.BASE_P) for _ in range(bsize)])

            V.transpose(inplace=True)  # n1-by-(k + NUM_COVS)
            V_mask.transpose(inplace=True)

            V_sub = Matrix(bsize, k + param.NUM_COVS)
            V_mask_sub = Matrix(bsize, k + param.NUM_COVS)

            print("done.")

            gkeep3: list = [False] * m0
            for j in range(m0):
                gkeep3[j] = (gkeep1[j] == 1)

            ind = 0
            for j in range(m0):
                if gkeep3[j]:
                    gkeep3[j] = (gkeep2[ind] == 1)
                    ind += 1

            ind = -1
            print("GWAS pass:")
            for cur in range(n1):
                ind += 1

                while ikeep[ind] != 1:
                    if mpc.pid > 0:
                        mpc.skip_data(f, 3, m0)  # g
                        mpc.skip_data(f, m0)  # miss

                        mpc.switch_seed(10)
                        g0_mask = Matrix(3, m0, randomise=True)
                        miss0_mask = mpc.random_vector(m0)
                        mpc.restore_seed(10)
                    else:
                        for p in range(1, 3):
                            mpc.switch_seed(10 + p)
                            g0_mask = Matrix(3, m0, randomise=True)
                            miss0_mask = mpc.random_vector(m0)
                            mpc.restore_seed(10 + p)
                    ind += 1

                if mpc.pid > 0:
                    g0 = mpc.read_from_file(f, 3, m0)  # g
                    miss0 = mpc.read_from_file(f, m0)  # miss

                    mpc.switch_seed(10)
                    g0_mask = Matrix(3, m0, randomise=True)
                    miss0_mask = mpc.random_vector(m0)
                    mpc.restore_seed(10)
                else:
                    g0 = Matrix(3, m0)
                    g0_mask = Matrix(3, m0)
                    miss0 = Vector([Zp(0, param.BASE_P) for _ in range(m0)])
                    miss0_mask = Vector([Zp(0, param.BASE_P) for _ in range(m0)])

                    for p in range(1 ,3):
                        mpc.switch_seed(10 + p)
                        tmp_mat = Matrix(3, m0, randomise=True)
                        tmp_vec = mpc.random_vector(m0)
                        mpc.restore_seed(10 + p)

                        g0_mask += tmp_mat
                        miss0_mask += tmp_vec
                
                g = Matrix(3, m2)
                miss = Vector([Zp(0, param.BASE_P) for _ in range(m2)])
                g_mask = Matrix(3, m2)
                miss_mask = Vector([Zp(0, param.BASE_P) for _ in range(m2)])
                ind2: int = 0
                
                for j in range(m0):
                    if gkeep3[j]:
                        for k in range(3):
                            g[k][ind2] = g0[k][j]
                            g_mask[k][ind2] = g0_mask[k][j]

                        miss[ind2] = miss0[j]
                        miss_mask[ind2] = miss0_mask[j]
                        ind2 += 1

            dosage[cur % bsize] = g[1] + g[2] * 2
            dosage_mask[cur % bsize] = g_mask[1] + g_mask[2] * 2

            u_vec[cur % bsize] = u[cur]
            u_mask_vec[cur % bsize] = u_mask[cur]
            p_hat_vec[cur % bsize] = p_hat[cur]
            p_hat_mask_vec[cur % bsize] = p_hat_mask[cur]

            V_sub[cur % bsize] = V[cur]
            V_mask_sub[cur % bsize] = V_mask[cur]

            if cur % bsize == bsize - 1:
                sx = mpc.beaver_mult(u_vec, u_mask_vec, dosage, dosage_mask)
                sxp = mpc.beaver_mult(p_hat_vec, p_hat_mask_vec, dosage, dosage_mask)

                sxx_tmp: Matrix = mpc.beaver_mult_elem(dosage, dosage_mask, dosage, dosage_mask)
                for b in range(bsize):
                    sxx += sxx_tmp[b]

                V_sub.transpose(inplace=True)  # (k + NUM_COVS)-by-bsize
                V_mask_sub.transpose(inplace=True)
                B = mpc.beaver_mult(V_sub, V_mask_sub, dosage, dosage_mask)

                dosage = Matrix(bsize, m2)
                dosage_mask = Matrix(bsize, m2)
                V_sub = Matrix(bsize, k + param.NUM_COVS)
                V_mask_sub = Matrix(bsize, k + param.NUM_COVS)

        remainder: int = n1 % bsize
        if remainder > 0:
            dosage.set_dims(remainder, m2)
            dosage_mask.set_dims(remainder, m2)
            u_vec.set_length(remainder)
            u_mask_vec.set_length(remainder)
            p_hat_vec.set_length(remainder)
            p_hat_mask_vec.set_length(remainder)
            V_sub.set_dims(remainder, k + param.NUM_COVS)
            V_mask_sub.set_dims(remainder, k + param.NUM_COVS)

            sx = mpc.beaver_mult(u_vec, u_mask_vec, dosage, dosage_mask)
            sxp = mpc.beaver_mult(p_hat_vec, p_hat_mask_vec, dosage, dosage_mask)

            sxx_tmp: Matrix = mpc.beaver_mult_elem(dosage, dosage_mask, dosage, dosage_mask)
            for b in range(remainder):
                sxx += sxx_tmp[b]

            V_sub.transpose(inplace=True)  # (k + NUM_COVS)-by-remainder
            V_mask_sub.transpose(inplace=True)

            B = mpc.beaver_mult(V_sub, V_mask_sub, dosage, dosage_mask)

        sx = mpc.beaver_reconstruct(sx)
        sxp = mpc.beaver_reconstruct(sxp)
        sxx = mpc.beaver_reconstruct(sxx)
        B = mpc.beaver_reconstruct(B)
        sxx *= fp_one

        with open(get_cache_path(mpc.pid, "gwas_stats"), 'wb') as f:
            mpc.write_to_file(sx, f)
            mpc.write_to_file(sxx, f)
            mpc.write_to_file(sxp, f)
            mpc.write_to_file(B, f)

        print("Wrote results to cache")

    B.transpose(inplace=True)  # m2-by-(k + param.NUM_COVS)

    BB = mpc.inner_prod(B)  # m2
    mpc.trunc(BB)
    if mpc.pid > 0:
        sxx -= BB

    sp = Zp(0, param.BASE_P)
    if mpc.pid > 0:
        for i in range(n1):
            sp += p_hat_mask[i]
            if mpc.pid == 1:
                sp += p_hat[i]

    spp = Zp(0, param.BASE_P)
    spp = mpc.beaver_inner_prod(p_hat, p_hat_mask)
    spp = mpc.beaver_reconstruct(spp)

    fp_n1_inv: Zp = mpc.double_to_fp(1 / n1, param.NBIT_K, param.NBIT_F)
    sx *= fp_n1_inv
    sp *= fp_n1_inv

    mpc.trunc(sx)
    mpc.trunc(sp)
    mpc.trunc(spp)

    sx_mask: Vector = mpc.beaver_partition(sx)
    sp_mask: Zp = mpc.beaver_partition(sp)

    sp2 = Zp(0, param.BASE_P)
    spsx = Vector([Zp(0, param.BASE_P) for _ in range(m2)])
    sx2 = Vector([Zp(0, param.BASE_P) for _ in range(m2)])

    spsx = mpc.beaver_mult(sx, sx_mask, sp, sp_mask)
    sp2 = mpc.beaver_mult(sp, sp_mask, sp, sp_mask)
    sx2 = mpc.beaver_mult_elem(sx, sx_mask, sx, sx_mask)

    spsx = mpc.beaver_reconstruct(spsx)
    sp2 = mpc.beaver_reconstruct(sp2)
    sx2 = mpc.beaver_reconstruct(sx2)

    spsx *= n1
    sp2 *= n1
    sx2 *= n1

    mpc.trunc(spsx)
    mpc.trunc(sp2)
    mpc.trunc(sx2)

    numer = Vector([Zp(0, param.BASE_P) for _ in range(m2)])
    denom = Vector([Zp(0, param.BASE_P) for _ in range(m2 + 1)])
    if mpc.pid > 0:
        numer = sxp - spsx
        for i in range(m2):
            denom[i] = sxx[i] - sx2[i]
        denom[m2] = spp - sp2

    denom1_sqrt_inv = Vector()
    if (os.path.exists(get_cache_path(mpc.pid, "denom_inv"))):
        print("denom_inv cache found")
        with open(get_cache_path(mpc.pid, "denom_inv")) as f:
            denom1_sqrt_inv = mpc.read_from_file(f, len(denom))
    else:
        tmp_vec = mpc.FPSqrt(denom1_sqrt_inv, denom)

        with open(get_cache_path(mpc.pid, "denom_inv"), 'wb') as f:
            if mpc.pid > 0:
               mpc.write_to_file(denom1_sqrt_inv, f)

    denom2_sqrt_inv: Zp = denom1_sqrt_inv[m2]  # p term
    denom1_sqrt_inv.set_length(m2)  # truncate

    z: Vector = mpc.mult_elem(numer, denom1_sqrt_inv)
    mpc.trunc_vec(z)

    z *= denom2_sqrt_inv
    mpc.trunc_vec(z)

    print("Association statistics calculated")
    z = mpc.reveal_sym(z)
    if mpc.pid == 2:
        z_double = mpc.fp_to_double_vec(z, param.NBIT_K, param.NBIT_F)
        with open(get_output_path(mpc.pid, "assoc"), 'w') as f:
            for i in range(len(z_double)):
                f.write(f'{z_double[i]}\n')
        print("Result written to ", get_output_path(mpc.pid, "assoc"))

    return True
