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
    
    if mpc.pid == 2:
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
    
    if mpc.pid == 2:
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
    
    if mpc.pid == 2:
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

    if mpc.pid == 2:
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

    if not os.path.exists(get_cache_path(pid, 'logi_input')):
        raise NotImplementedError(
            'At this point, logi_input is expected in cache.\n'
            'TODO: Haris. Make it cache agnostic. (See original implementation)')
    
    print('logi_input cache found')
    if not test_run:
        with open(get_cache_path(pid, 'logi_input'), 'br') as f:
            X, X_mask = mpc.beaver_read_from_file(f, ntop, n1)
    else:
        with open(get_temp_path(pid, 'logi_input')) as f:
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
        output_path: str = get_output_path(pid, 'logi_coeff')
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

    with open(get_cache_path(pid, "input_pheno_cov")) as f:
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
            history = os.path.exists(get_output_path("gkeep1"))
            mpc.send_bool(history, 0)
            mpc.send_bool(history, 1)
        else:
            # ask P2 if gkeep1 has been computed before
            history = mpc.receive_bool(2)

        if history:
            print("Using locus missing rate filter from a previous run")
        
            if mpc.pid == 2:
                with open(get_output_path("gkeep1")):
                    for i in range(m0):
                        gkeep1[i] = Zp(int(f.readline()), base=param.BASE_P)

                mpc.send_elem(gkeep1, 0)
                mpc.send_elem(gkeep1, 1)
            else:
                gkeep1: Vector = mpc.receive_vector(2, msg_len=TypeOps.get_vec_len(m0), fid=0)
        else:
            gmiss: Vector = Vector([Zp(0, base=param.BASE_P) for _ in range(m0)])
            
            if os.path.exists(get_cache_path(pid, "gmiss")):
                print("Locus missing rate cache found")

                with open(get_cache_path(pid, "gmiss")) as f:
                    gmiss = mpc.read_vector(f, m0)
            else:
                print("Taking a pass to calculate locus missing rates:")

                if mpc.pid > 0:
                    with open(get_cache_path(pid, "input_geno")) as f:
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


                with open(get_cache_path(pid, "gmiss")) as f:
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
                with open(get_output_path("gkeep1"), 'w') as f:
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
            history = os.path.exists(get_output_path("ikeep"))
            mpc.send_bool(history, 0)
            mpc.send_bool(history, 1)
        else:
            # ask P2 if ikeep has been computed before
            history = mpc.receive_bool(2)

        if history:
            print("Using individual missing rate/het rate filters from a previous run")
        
            if mpc.pid == 2:
                with open(get_output_path("ikeep")) as f:
                    for i in range(n0):
                        ikeep[i] = Zp(int(f.readline()), base=param.BASE_P)

                mpc.send_elem(ikeep, 0)
                mpc.send_elem(ikeep, 1)
            else:
                mpc.receive_vector(ikeep, 2, TypeOps.get_vec_len(n0))
        else:
            imiss = Vector([Zp(0, param.BASE_P) for _ in range(n0)])
            ihet = Vector([Zp(0, param.BASE_P) for _ in range(n0)])

            if os.path.exists(get_cache_path(pid, "imiss_ihet")):
                print("Individual missing rate and het rate cache found")

                with open(get_cache_path(pid, "imiss_ihet")) as f:
                    imiss: Vector = mpc.read_vector(f, n0)
                    ihet: Vector = mpc.read_vector(f, n0)
            else:
                print("Taking a pass to calculate individual missing rates and het rates:")

                if mpc.pid > 0:
                    with open(get_cache_path(pid, "input_geno")) as f:
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

                with open(get_cache_path(pid, "imiss_ihet")) as f:
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
                with open(get_output_path("ikeep"), 'w') as f:
                    for i in range(ikeep.length()):
                        f.write(f'{ikeep[i]}\n')

    mpc.ProfilerPopState(true); // ind_miss/het

    uint n1 = conv<uint>(Sum(ikeep));

    print("n1: " << n1 << ", " << "m1: " << m1)

    print("Filtering phenotypes and covariates")
    mpc.Filter(pheno, ikeep, n1);
    mpc.FilterRows(cov, ikeep, n1);

    Vec<ZZ_p> ctrl;
    mpc.FlipBit(ctrl, pheno);

    Vec<ZZ_p> ctrl_mask;
    mpc.BeaverPartition(ctrl_mask, ctrl);

    Vec<ZZ_p> dosage_sum;
    Vec<ZZ_p> gmiss, gmiss_ctrl, dosage_sum_ctrl;
    Mat<ZZ_p> g_count_ctrl;
    ZZ_p n1_ctrl(0);

    Init(gmiss, m1);
    Init(gmiss_ctrl, m1);
    Init(dosage_sum, m1);
    Init(dosage_sum_ctrl, m1);
    Init(g_count_ctrl, 3, m1);

    mpc.ProfilerPushState("data_scan");

    if (os.path.exists(get_cache_path(pid, "geno_stats"))) {
        print("Genotype statistics cache found")

        with open(get_cache_path(pid, "geno_stats"));
        mpc.ReadFromFile(gmiss, ifs, m1);
        mpc.ReadFromFile(gmiss_ctrl, ifs, m1);
        mpc.ReadFromFile(dosage_sum, ifs, m1);
        mpc.ReadFromFile(dosage_sum_ctrl, ifs, m1);
        mpc.ReadFromFile(g_count_ctrl, ifs, 3, m1);
        mpc.ReadFromFile(n1_ctrl, ifs);
        ifs.close();
    } else {
        print("Taking a pass to calculate genotype statistics:")

        with open(get_cache_path(pid, "input_geno"));
        if (mpc.pid > 0) {
        mpc.import_seed(10, ifs);
        } else {
        for (int p = 1; p <= 2; p++) {
            mpc.import_seed(10 + p, ifs);
        }
        }

        long report_bsize = n1 / 10;

        long bsize = param.PITER_BATCH_SIZE;

        // Containers for batching the computation
        Vec< Mat<ZZ_p> > g, g_mask;
        Mat<ZZ_p> dosage, dosage_mask;
        Mat<ZZ_p> miss, miss_mask;
        Vec<ZZ_p> ctrl_vec, ctrl_mask_vec;
        g.SetLength(3);
        g_mask.SetLength(3);
        dosage.SetDims(bsize, m1);
        dosage_mask.SetDims(bsize, m1);
        miss.SetDims(bsize, m1);
        miss_mask.SetDims(bsize, m1);
        for (int k = 0; k < 3; k++) {
        g[k].SetDims(bsize, m1);
        g_mask[k].SetDims(bsize, m1);
        }
        ctrl_vec.SetLength(bsize);
        ctrl_mask_vec.SetLength(bsize);

        ind = -1;
        tic();
        for i in range(n1):
        ind++;

        mpc.ProfilerPushState("file_io/rng");

        Mat<ZZ_p> g0, g0_mask;
        Vec<ZZ_p> miss0, miss0_mask;

        while (ikeep[ind] != 1) {
            if (mpc.pid > 0) {
            mpc.SkipData(ifs, 3, m0); // g
            mpc.SkipData(ifs, m0); // miss

            mpc.switch_seed(10);
            Matrix(g0_mask, 3, m0);
            mpc.random_vector(miss0_mask, m0);
            mpc.restore_seed();
            } else {
            for (int p = 1; p <= 2; p++) {
                mpc.switch_seed(10 + p);
                Matrix(g0_mask, 3, m0);
                mpc.random_vector(miss0_mask, m0);
                mpc.restore_seed();
            }
            }
            ind++;
        }

        if (mpc.pid > 0) {
            mpc.ReadFromFile(g0, ifs, 3, m0); // g
            mpc.ReadFromFile(miss0, ifs, m0); // miss

            mpc.switch_seed(10);
            Matrix(g0_mask, 3, m0);
            mpc.random_vector(miss0_mask, m0);
            mpc.restore_seed();
        } else {
            Init(g0, 3, m0);
            Init(g0_mask, 3, m0);
            Init(miss0, m0);
            Init(miss0_mask, m0);

            for (int p = 1; p <= 2; p++) {
            mpc.switch_seed(10 + p);
            Matrix(tmp_mat, 3, m0);
            mpc.random_vector(tmp_vec, m0);
            mpc.restore_seed();

            g0_mask += tmp_mat;
            miss0_mask += tmp_vec;
            }
        }
        
        mpc.ProfilerPopState(false); // file_io/rng

        // Filter out loci that failed missing rate filter
        int ind2 = 0;
        for j in range(m0):
            if (gkeep1[j] == 1) {
            for (int k = 0; k < 3; k++) {
                g[k][i % bsize][ind2] = g0[k][j];
                g_mask[k][i % bsize][ind2] = g0_mask[k][j];
            }
            miss[i % bsize][ind2] = miss0[j];
            miss_mask[i % bsize][ind2] = miss0_mask[j];
            ind2++;
            }
        }

        dosage[i % bsize] = g[1][i % bsize] + 2 * g[2][i % bsize];
        dosage_mask[i % bsize] = g_mask[1][i % bsize] + 2 * g_mask[2][i % bsize];

        ctrl_vec[i % bsize] = ctrl[i];
        ctrl_mask_vec[i % bsize] = ctrl_mask[i];

        // Update running sums
        if (mpc.pid > 0) {
            n1_ctrl += ctrl_mask[i];
            gmiss += miss_mask[i % bsize];
            dosage_sum += dosage_mask[i % bsize];

            if (mpc.pid == 1) {
            n1_ctrl += ctrl[i];
            gmiss += miss[i % bsize];
            dosage_sum += dosage[i % bsize];
            }
        }

        if (i % bsize == bsize - 1 || i == n1 - 1) {
            if (i % bsize < bsize - 1) {
            int new_bsize = (i % bsize) + 1;
            for (int k = 0; k < 3; k++) {
                g[k].SetDims(new_bsize, m1);
                g_mask[k].SetDims(new_bsize, m1);
            }
            dosage.SetDims(new_bsize, m1);
            dosage_mask.SetDims(new_bsize, m1);
            miss.SetDims(new_bsize, m1);
            miss_mask.SetDims(new_bsize, m1);
            ctrl_vec.SetLength(new_bsize);
            ctrl_mask_vec.SetLength(new_bsize);
            }

            mpc.BeaverMult(gmiss_ctrl, ctrl_vec, ctrl_mask_vec, miss, miss_mask);
            mpc.BeaverMult(dosage_sum_ctrl, ctrl_vec, ctrl_mask_vec, dosage, dosage_mask);
            for (int k = 0; k < 3; k++) {
            mpc.BeaverMult(g_count_ctrl[k], ctrl_vec, ctrl_mask_vec, g[k], g_mask[k]);
            }
        }

        if ((i + 1) % report_bsize == 0 || i == n1 - 1) {
            print("\t" << i+1 << " / " << n1 << ", "; toc(); tic();
        }
        }

        ifs.close();

        mpc.BeaverReconstruct(gmiss_ctrl);
        mpc.BeaverReconstruct(dosage_sum_ctrl);
        mpc.BeaverReconstruct(g_count_ctrl);

        // Write to cache
        with open(get_cache_path(pid, "geno_stats")) as f:
        mpc.write_to_file(gmiss, fs);
        mpc.write_to_file(gmiss_ctrl, fs);
        mpc.write_to_file(dosage_sum, fs);
        mpc.write_to_file(dosage_sum_ctrl, fs);
        mpc.write_to_file(g_count_ctrl, fs);
        mpc.write_to_file(n1_ctrl, fs);
        fs.close();

        print("Wrote results to cache")
    }

    mpc.ProfilerPopState(true); // data_scan

    mpc.ProfilerPushState("maf/hwe");

    if (param.DEBUG) {
        print("gmiss")
        mpc.Print(gmiss, 5);
        print("gmiss_ctrl")
        mpc.Print(gmiss_ctrl, 5);
        print("dosage_sum")
        mpc.Print(dosage_sum, 5);
        print("dosage_sum_ctrl")
        mpc.Print(dosage_sum_ctrl, 5);
        print("g_count_ctrl")
        for i in range(3):
        mpc.Print(g_count_ctrl[i], 5);
        }
    }

    mpc.ProfilerPushState("maf");

    // SNP MAF filter
    print("Locus minor allele frequency (MAF) filter ... ")
    ZZ_p maf_lb:Zp = mpc.double_to_fp(param.MAF_LB, param.NBIT_K, param.NBIT_F);
    ZZ_p maf_ub:Zp = mpc.double_to_fp(param.MAF_UB, param.NBIT_K, param.NBIT_F);

    Vec<ZZ_p> dosage_tot, dosage_tot_ctrl;
    if (mpc.pid > 0) {
        dosage_tot = -gmiss;
        dosage_tot_ctrl = -gmiss_ctrl;
        mpc.AddPublic(dosage_tot, ZZ_p(n1));
        mpc.Add(dosage_tot_ctrl, n1_ctrl);
        dosage_tot *= 2;
        dosage_tot_ctrl *= 2;
    } else {
        dosage_tot.SetLength(m1);
        dosage_tot_ctrl.SetLength(m1);
    }

    print("Calculating MAFs ... ") tic();
    Vec<ZZ_p> maf, maf_ctrl;
    if (os.path.exists(get_cache_path(pid, "maf"))) {
        print("maf cache found")
        with open(get_cache_path(pid, "maf"));
        mpc.ReadFromFile(maf, ifs, dosage_tot.length());
        mpc.ReadFromFile(maf_ctrl, ifs, dosage_tot_ctrl.length());
        ifs.close();
    } else {
        mpc.ProfilerPushState("div");
        mpc.FPDiv(maf, dosage_sum, dosage_tot); 
        mpc.FPDiv(maf_ctrl, dosage_sum_ctrl, dosage_tot_ctrl); 
        mpc.ProfilerPopState(false); // div

        with open(get_cache_path(pid, "maf")) as f:
        mpc.write_to_file(maf, fs);
        mpc.write_to_file(maf_ctrl, fs);
        fs.close();
    }
    print("done. "; toc();

    Vec<ZZ_p> Maf, Maf_ctrl; // MAJOR allele freq
    if (mpc.pid > 0) {
        Maf = -maf;
        Maf_ctrl = -maf_ctrl;
        mpc.AddPublic(Maf, fp_one);
        mpc.AddPublic(Maf_ctrl, fp_one);
    } else {
        Maf.SetLength(m1);
        Maf_ctrl.SetLength(m1);
    }

    // Variance based on Bernoulli distribution over each allele
    Vec<ZZ_p> g_var_bern;
    mpc.MultElem(g_var_bern, maf, Maf);
    mpc.Trunc(g_var_bern);

    mpc.ProfilerPopState(true); // maf

    if (param.DEBUG) {
        print("maf")
        mpc.PrintFP(maf, 5);
        print("maf_ctrl")
        mpc.PrintFP(maf_ctrl, 5);
    }

    Vec<ZZ_p> gkeep2;
    Init(gkeep2, m1);

    if (param.SKIP_QC) {
        for i in range(m1):
        gkeep2[i] = 1;
        }
        print("SNP MAF/HWE filters skipped")
    } else {
        bool history;
        if (mpc.pid == 2) {
        history = os.path.exists(get_output_path("gkeep2"));
        mpc.send_bool(history, 0);
        mpc.send_bool(history, 1);
        } else {
        // ask P2 if gkeep2 has been computed before
        history = mpc.receive_bool(2);
        }

        if (history) {
        print("Using MAF/HWE filters from a previous run")
    
        if (mpc.pid == 2) {
            with open(get_output_path("gkeep2"));
            for i in range(m1):
            ifs >> gkeep2[i];
            }
            ifs.close();

            mpc.send_elem(gkeep2, 0);
            mpc.send_elem(gkeep2, 1);
        } else {
            mpc.receive_vector(gkeep2, 2, m1);
        }
        } else {
        
        mpc.ProfilerPushState("maf_filt");

        mpc.less_than_public(gkeep2, maf, maf_ub);
        mpc.NotLessThanPublic(tmp_vec, maf, maf_lb);
        mpc.MultElem(gkeep2, gkeep2, tmp_vec);

        mpc.ProfilerPopState(true); // maf_filt

        mpc.ProfilerPushState("hwe_filt");

        print("Locus Hardy-Weinberg equilibrium (HWE) filter ... ") tic();
        ZZ_p hwe_ub:Zp = mpc.double_to_fp(param.HWE_UB, param.NBIT_K, param.NBIT_F); // p < 1e-7
        
        // Calculate expected genotype distribution in control group
        Mat<ZZ_p> g_exp_ctrl;
        Init(g_exp_ctrl, 3, m1);

        mpc.MultElem(g_exp_ctrl[0], Maf_ctrl, Maf_ctrl); // AA
        mpc.MultElem(g_exp_ctrl[1], Maf_ctrl, maf_ctrl); // Aa
        if (mpc.pid > 0) {
            g_exp_ctrl[1] *= 2;
        }
        mpc.MultElem(g_exp_ctrl[2], maf_ctrl, maf_ctrl); // aa

        for i in range(3):
            mpc.MultElem(g_exp_ctrl[i], g_exp_ctrl[i], dosage_tot_ctrl);
        }
        g_exp_ctrl *= twoinv; // dosage_tot_ctrl is twice the # individuals we actually want

        mpc.Trunc(g_exp_ctrl);

        print("\tCalculated expected genotype counts, "; toc(); tic();

        Vec<ZZ_p> hwe_chisq; 
        Init(hwe_chisq, m1);

        if (os.path.exists(get_cache_path(pid, "hwe"))) {
            print("HWE cache found")
            with open(get_cache_path(pid, "hwe"));
            mpc.ReadFromFile(hwe_chisq, ifs, m1);
            ifs.close();
        } else {
            for i in range(3):
            Vec<ZZ_p> diff;
            if (mpc.pid > 0) {
                diff = fp_one * g_count_ctrl[i] - g_exp_ctrl[i];
            } else {
                diff.SetLength(m1);
            }

            mpc.MultElem(diff, diff, diff); // square
            mpc.Trunc(diff);

            mpc.ProfilerPushState("div");
            mpc.FPDiv(tmp_vec, diff, g_exp_ctrl[i]);
            mpc.ProfilerPopState(false); // div
            hwe_chisq += tmp_vec;

            print("\tChi-square test (" << i+1 << "/3), "; toc(); tic();
            }

            with open(get_cache_path(pid, "hwe")) as f:
            mpc.write_to_file(hwe_chisq, fs);
            fs.close();
        }

        if (param.DEBUG) {
            print("hwe")
            mpc.PrintFP(hwe_chisq, 5);
        }
        
        Vec<ZZ_p> hwe_filt;
        mpc.less_than_public(hwe_filt, hwe_chisq, hwe_ub);
        mpc.MultElem(gkeep2, gkeep2, hwe_filt);
        hwe_filt.kill();

        // Reveal which SNPs to discard 
        mpc.reveal_sym(gkeep2);
            
        if (mpc.pid == 2) {
            mpc.send_elem(gkeep2, 0);
        elif (mpc.pid == 0) {
            mpc.receive_vector(gkeep2, 2, m1);
        }

        if (mpc.pid == 2) {
            owith open(get_output_path("gkeep2"));
            for i in range(gkeep2.length()):
            ofs << gkeep2[i])
            }
            ofs.close();
        }

        mpc.ProfilerPopState(true); // hwe_filt
        }
    }

    uint m2 = conv<uint>(Sum(gkeep2));
    print("n1: " << n1 << ", " << "m2: " << m2)

    print("Filtering genotype statistics")
    mpc.Filter(g_var_bern, gkeep2, m2);
    mpc.Filter(maf, gkeep2, m2);
    FilterVec(snp_pos, gkeep2);

    gmiss.kill();
    gmiss_ctrl.kill();
    dosage_sum.kill();
    dosage_sum_ctrl.kill();
    g_count_ctrl.kill();

    mpc.ProfilerPopState(false); // maf/hwe
    mpc.ProfilerPopState(true); // qc
    mpc.ProfilerPushState("std_param");

    Vec<ZZ_p> g_std_bern_inv;
    if (os.path.exists(get_cache_path(pid, "stdinv_bern"))) {
        print("Genotype standard deviation cache found")

        with open(get_cache_path(pid, "stdinv_bern"));
        mpc.ReadFromFile(g_std_bern_inv, ifs, g_var_bern.length());
        ifs.close();

    } else {
        print("Calculating genotype standard deviations (inverse)")

        mpc.ProfilerPushState("sqrt");
        mpc.FPSqrt(tmp_vec, g_std_bern_inv, g_var_bern);
        mpc.ProfilerPopState(false); // sqrt

        with open(get_cache_path(pid, "stdinv_bern")) as f:
        mpc.write_to_file(g_std_bern_inv, fs);
        fs.close();
    }

    if (param.DEBUG) {
        print("g_std_bern_inv")
        mpc.PrintFP(g_std_bern_inv, 5);
    }

    Vec<ZZ_p> g_mean;
    if (mpc.pid > 0) {
        g_mean = 2 * maf;
    } else {
        g_mean.SetLength(m2);
    }

    mpc.ProfilerPopState(true); // std_param

    print("Starting population stratification analysis")

    mpc.ProfilerPushState("pop_strat");
    mpc.ProfilerPushState("select_snp");

    Vec<int8_t> selected; // 1 selected, 0 unselected, -1 TBD
    Vec<bool> to_process;
    selected.SetLength(m2);
    to_process.SetLength(m2);

    for i in range(m2):
        selected[i] = -1;
    }

    ZZ dist_thres(param.LD_DIST_THRES);
    
    ZZ prev(-1);
    for i in range(m2):
        selected[i] = 0;
        if (prev < 0 || snp_pos[i] - prev > dist_thres) {
        selected[i] = 1;
        prev = snp_pos[i];
        }
    }

    // At this point "selected" contains the SNP filter for PCA, shared across all parties
    uint32_t m3 = 0;
    for i in range(selected.length()):
        if (selected[i] == 1) {
        m3++;
        }
    }

    print("SNP selection complete: " << m3 << " / " << m2 << " selected")
    mpc.ProfilerPopState(false); // select_snp
    mpc.ProfilerPushState("reduce_file");

    // Cache the reduced G for PCA
    if (os.path.exists(get_cache_path(pid, "pca_input"))) {
        print("pca_input cache found")
    } else {
        Vec<bool> gkeep3;
        gkeep3.SetLength(m0);
        for j in range(m0):
        gkeep3[j] = gkeep1[j] == 1;
        }

        ind = 0;
        for j in range(m0):
        if (gkeep3[j]) {
            gkeep3[j] = gkeep2[ind] == 1;
            ind++;
        }
        }

        ind = 0;
        for j in range(m0):
        if (gkeep3[j]) {
            gkeep3[j] = selected[ind] == 1;
            ind++;
        }
        }

        with open(get_cache_path(pid, "input_geno"));
        if (mpc.pid > 0) {
        mpc.import_seed(10, ifs);
        } else {
        for (int p = 1; p <= 2; p++) {
            mpc.import_seed(10 + p, ifs);
        }
        }

        long bsize = n1 / 10;

        print("Caching input data for PCA:")

        with open(get_cache_path(pid, "pca_input")) as f:

        ind = -1;
        tic();
        for i in range(n1):
        ind++;

        mpc.ProfilerPushState("file_io/rng");

        Mat<ZZ_p> g0, g0_mask;
        Vec<ZZ_p> miss0, miss0_mask;

        while (ikeep[ind] != 1) {
            if (mpc.pid > 0) {
            mpc.SkipData(ifs, 3, m0); // g
            mpc.SkipData(ifs, m0); // miss

            mpc.switch_seed(10);
            Matrix(g0_mask, 3, m0);
            mpc.random_vector(miss0_mask, m0);
            mpc.restore_seed();
            } else {
            for (int p = 1; p <= 2; p++) {
                mpc.switch_seed(10 + p);
                Matrix(g0_mask, 3, m0);
                mpc.random_vector(miss0_mask, m0);
                mpc.restore_seed();
            }
            }
            ind++;
        }

        if (mpc.pid > 0) {
            mpc.ReadFromFile(g0, ifs, 3, m0); // g
            mpc.ReadFromFile(miss0, ifs, m0); // miss

            mpc.switch_seed(10);
            Matrix(g0_mask, 3, m0);
            mpc.random_vector(miss0_mask, m0);
            mpc.restore_seed();
        } else {
            Init(g0, 3, m0);
            Init(g0_mask, 3, m0);
            Init(miss0, m0);
            Init(miss0_mask, m0);

            for (int p = 1; p <= 2; p++) {
            mpc.switch_seed(10 + p);
            Matrix(tmp_mat, 3, m0);
            mpc.random_vector(tmp_vec, m0);
            mpc.restore_seed();

            g0_mask += tmp_mat;
            miss0_mask += tmp_vec;
            }
        }
        
        mpc.ProfilerPopState(false); // file_io/rng

        // Filter out loci that failed missing rate filter
        Mat<ZZ_p> g, g_mask;
        Vec<ZZ_p> miss, miss_mask;
        g.SetDims(3, m3);
        g_mask.SetDims(3, m3);
        miss.SetLength(m3);
        miss_mask.SetLength(m3);
        int ind2 = 0;
        for j in range(m0):
            if (gkeep3[j]) {
            for (int k = 0; k < 3; k++) {
                g[k][ind2] = g0[k][j];
                g_mask[k][ind2] = g0_mask[k][j];
            }
            miss[ind2] = miss0[j];
            miss_mask[ind2] = miss0_mask[j];
            ind2++;
            }
        }

        Vec<ZZ_p> dosage, dosage_mask;
        dosage = g[1] + 2 * g[2];
        dosage_mask = g_mask[1] + 2 * g_mask[2];

        mpc.BeaverWriteToFile(dosage, dosage_mask, fs);
        mpc.BeaverWriteToFile(miss, miss_mask, fs);

        if ((i + 1) % bsize == 0 || i == n1 - 1) {
            print("\t" << i+1 << " / " << n1 << ", "; toc(); tic();
        }
        }

        ifs.close();
        fs.close();
    }

    mpc.ProfilerPopState(false); // reduce_file

    Vec<ZZ_p> g_mean_pca = g_mean;
    mpc.Filter(g_mean_pca, selected, m3);

    Vec<ZZ_p> g_stdinv_pca = g_std_bern_inv;
    mpc.Filter(g_stdinv_pca, selected, m3);

    Vec<ZZ_p> g_mean_pca_mask, g_stdinv_pca_mask;
    mpc.BeaverPartition(g_mean_pca_mask, g_mean_pca);
    mpc.BeaverPartition(g_stdinv_pca_mask, g_stdinv_pca);

    /* Pass 2: Random sketch */
    Mat<ZZ_p> Y_cur;
    Init(Y_cur, kp, m3);

    if (os.path.exists(get_cache_path(pid, "sketch"))) {

        print("sketch cache found")
        with open(get_cache_path(pid, "sketch"), ios::in | ios::binary);
        ifs >> kp;
        mpc.ReadFromFile(Y_cur, ifs, kp, m3);
        ifs.close();

    } else {

        mpc.ProfilerPushState("rand_proj");
        
        Mat<ZZ_p> Y_cur_adj;
        Init(Y_cur_adj, kp, m3);

        Vec<int> bucket_count;
        bucket_count.SetLength(kp);
        for i in range(kp):
        bucket_count[i] = 0;
        }

        with open(get_cache_path(pid, "pca_input"), ios::in | ios::binary);
        for (int cur = 0; cur < n1; cur++) {
        // Count sketch (use global PRG)
        mpc.switch_seed(-1);
        long bucket_index = RandomBnd(kp);
        long rand_sign = RandomBnd(2) * 2 - 1;
        mpc.restore_seed();

        Vec<ZZ_p> g, g_mask, miss, miss_mask;
        mpc.BeaverReadFromFile(g, g_mask, ifs, m3);
        mpc.BeaverReadFromFile(miss, miss_mask, ifs, m3);

        // Flip miss bits so it points to places where g_mean should be subtracted
        mpc.BeaverFlipBit(miss, miss_mask);

        // Update running sum
        if (mpc.pid > 0) {
            Y_cur[bucket_index] += rand_sign * g_mask;
            if (mpc.pid == 1) {
            Y_cur[bucket_index] += rand_sign * g;
            }
        }

        // Update adjustment factor
        miss *= rand_sign;
        miss_mask *= rand_sign;
        mpc.BeaverMultElem(Y_cur_adj[bucket_index], miss, miss_mask, g_mean_pca, g_mean_pca_mask);

        bucket_count[bucket_index]++;
        }
        ifs.close();

        // Subtract the adjustment factor
        mpc.BeaverReconstruct(Y_cur_adj);
        if (mpc.pid > 0) {
        Y_cur = fp_one * Y_cur - Y_cur_adj;
        }
        Y_cur_adj.kill();

        if (param.DEBUG) {
        print("Y_cur")
        mpc.PrintFP(Y_cur[0], 5);
        print("g_mean_pca")
        mpc.PrintBeaverFP(g_mean_pca, g_mean_pca_mask, 10);
        print("g_stdinv_pca")
        mpc.PrintBeaverFP(g_stdinv_pca, g_stdinv_pca_mask, 10);
        }

        // Get rid of empty buckets and normalize nonempty ones
        int empty_slot = 0;
        for i in range(kp):
        if (bucket_count[i] > 0) {
            ZZ_p fp_count_inv:Zp = mpc.double_to_fp(1 / ((double) bucket_count[i]), param.NBIT_K, param.NBIT_F);
            Y_cur[empty_slot] = Y_cur[i] * fp_count_inv;
            empty_slot++;
        }
        }
        kp = empty_slot;
        Y_cur.SetDims(kp, m3);
        mpc.Trunc(Y_cur);

        mpc.ProfilerPopState(true); // rand_proj

        with open(get_cache_path(pid, "sketch")) as f:
        fs << kp;
        if (mpc.pid > 0) {
        mpc.write_to_file(Y_cur, fs);
        }
        fs.close();
    }

    mpc.ProfilerPushState("power_iter");

    Mat<ZZ_p> Y_cur_mask;
    mpc.BeaverPartition(Y_cur_mask, Y_cur);

    print("Initial sketch obtained, starting power iteration (num iter = " << param.NUM_POWER_ITER << ")")
    tic();

    Mat<ZZ_p> gQ;

    if (os.path.exists(get_cache_path(pid, "piter"))) {

        print("piter cache found")
        with open(get_cache_path(pid, "piter"), ios::in | ios::binary);
        mpc.ReadFromFile(gQ, ifs, n1, kp);
        ifs.close();
        
    } else {

        // Divide by standard deviation
        Mat<ZZ_p> Y;
        Init(Y, kp, m3);

        for i in range(kp):
        mpc.BeaverMultElem(Y[i], Y_cur[i], Y_cur_mask[i], g_stdinv_pca, g_stdinv_pca_mask);
        }
        Y_cur.kill();
        Y_cur_mask.kill();

        mpc.BeaverReconstruct(Y);
        mpc.Trunc(Y);

        /* Calculate orthonormal bases of Y */
        Mat<ZZ_p> Q;
        mpc.ProfilerPushState("qr_m");
        mpc.OrthonormalBasis(Q, Y);
        mpc.ProfilerPopState(false); // qr_m
        Y.kill();

        Mat<ZZ_p> gQ_adj;
        Mat<ZZ_p> Q_mask;
        Mat<ZZ_p> Q_scaled, Q_scaled_mask;
        Mat<ZZ_p> Q_scaled_gmean, Q_scaled_gmean_mask;

        /* Power iteration */
        for (int pit = 0; pit <= param.NUM_POWER_ITER; pit++) {
        /* This section is ran before each iteration AND once after all iterations */
        mpc.BeaverPartition(Q_mask, Q);

        // Normalize Q by standard deviations
        Init(Q_scaled, kp, m3);
        for i in range(kp):
            mpc.BeaverMultElem(Q_scaled[i], Q[i], Q_mask[i], g_stdinv_pca, g_stdinv_pca_mask);
        }
        mpc.BeaverReconstruct(Q_scaled);
        mpc.Trunc(Q_scaled);

        mpc.BeaverPartition(Q_scaled_mask, Q_scaled);

        // Pre-multiply with g_mean to simplify calculation of centering matrix
        Init(Q_scaled_gmean, kp, m3);
        for i in range(kp):
            mpc.BeaverMultElem(Q_scaled_gmean[i], Q_scaled[i], Q_scaled_mask[i],
                            g_mean_pca, g_mean_pca_mask);
        }
        mpc.BeaverReconstruct(Q_scaled_gmean);
        mpc.Trunc(Q_scaled_gmean);

        mpc.Transpose(Q_scaled); // m3-by-kp
        transpose(Q_scaled_mask, Q_scaled_mask); // m3-by-kp, unlike mpc.Transpose, P0 also transposes
        mpc.Transpose(Q_scaled_gmean); // m3-by-kp
        mpc.BeaverPartition(Q_scaled_gmean_mask, Q_scaled_gmean);

        Mat<ZZ_p> g, g_mask, miss, miss_mask;

        long bsize = param.PITER_BATCH_SIZE;

        Init(g, bsize, m3);
        Init(g_mask, bsize, m3);
        Init(miss, bsize, m3);
        Init(miss_mask, bsize, m3);
        
        /* Pass 1 */
        Init(gQ, n1, kp);
        Init(gQ_adj, n1, kp);

        mpc.ProfilerPushState("data_scan0");

        mpc.ProfilerPushState("file_io");
        with open(get_cache_path(pid, "pca_input"), ios::in | ios::binary);
        for (int cur = 0; cur < n1; cur++) {
            mpc.BeaverReadFromFile(g[cur % bsize], g_mask[cur % bsize], ifs, m3);
            mpc.BeaverReadFromFile(miss[cur % bsize], miss_mask[cur % bsize], ifs, m3);
            mpc.BeaverFlipBit(miss[cur % bsize], miss_mask[cur % bsize]);

            if (cur % bsize == bsize - 1) {
            mpc.ProfilerPopState(false); // file_io

            Init(tmp_mat, bsize, kp);
            mpc.BeaverMult(tmp_mat, g, g_mask, Q_scaled, Q_scaled_mask);
            for i in range(bsize):
                gQ[cur-(bsize-1)+i] = tmp_mat[i];
            }

            Init(tmp_mat, bsize, kp);
            mpc.BeaverMult(tmp_mat, miss, miss_mask, Q_scaled_gmean, Q_scaled_gmean_mask);
            for i in range(bsize):
                gQ_adj[cur-(bsize-1)+i] = tmp_mat[i];
            }

            mpc.ProfilerPushState("file_io");
            }
        }
        ifs.close();
        mpc.ProfilerPopState(false); // file_io

        long remainder = n1 % bsize;
        if (remainder > 0) {
            g.SetDims(remainder, m3);
            g_mask.SetDims(remainder, m3);
            miss.SetDims(remainder, m3);
            miss_mask.SetDims(remainder, m3);

            Init(tmp_mat, remainder, kp);
            mpc.BeaverMult(tmp_mat, g, g_mask, Q_scaled, Q_scaled_mask);
            for i in range(remainder):
            gQ[n1-remainder+i] = tmp_mat[i];
            }

            Init(tmp_mat, remainder, kp);
            mpc.BeaverMult(tmp_mat, miss, miss_mask, Q_scaled_gmean, Q_scaled_gmean_mask);
            for i in range(remainder):
            gQ_adj[n1-remainder+i] = tmp_mat[i];
            }

        }

        mpc.BeaverReconstruct(gQ);
        mpc.BeaverReconstruct(gQ_adj);
        if (mpc.pid > 0) {
            gQ -= gQ_adj;
        }

        mpc.ProfilerPopState(false); // data_scan1

        if (pit == param.NUM_POWER_ITER) { // Quit if all iterations are performed
            break;
        }

        mpc.Transpose(gQ); // kp-by-n1
        mpc.ProfilerPushState("qr_n");
        mpc.OrthonormalBasis(Q, gQ);
        mpc.ProfilerPopState(false); // qr_n
        mpc.Transpose(Q); // n1-by-kp

        mpc.BeaverPartition(Q_mask, Q);

        Init(gQ, kp, m3);
        Init(gQ_adj, kp, m3);

        Init(g, bsize, m3);
        Init(g_mask, bsize, m3);
        Init(miss, bsize, m3);
        Init(miss_mask, bsize, m3);

        Mat<ZZ_p> Qsub, Qsub_mask;
        Init(Qsub, bsize, kp);
        Init(Qsub_mask, bsize, kp);

        mpc.ProfilerPushState("data_scan2");

        // Pass 2
        mpc.ProfilerPushState("file_io");
        with open(get_cache_path(pid, "pca_input"), ios::in | ios::binary);
        for (int cur = 0; cur < n1; cur++) {
            mpc.BeaverReadFromFile(g[cur % bsize], g_mask[cur % bsize], ifs, m3);
            mpc.BeaverReadFromFile(miss[cur % bsize], miss_mask[cur % bsize], ifs, m3);
            mpc.BeaverFlipBit(miss[cur % bsize], miss_mask[cur % bsize]);

            Qsub[cur % bsize] = Q[cur];
            Qsub_mask[cur % bsize] = Q_mask[cur];

            if (cur % bsize == bsize - 1) {
            mpc.ProfilerPopState(false); // file_io

            mpc.Transpose(Qsub);
            transpose(Qsub_mask, Qsub_mask);

            mpc.BeaverMult(gQ, Qsub, Qsub_mask, g, g_mask);
            mpc.BeaverMult(gQ_adj, Qsub, Qsub_mask, miss, miss_mask);

            Qsub.SetDims(bsize, kp);
            Qsub_mask.SetDims(bsize, kp);

            mpc.ProfilerPushState("file_io");
            }
        }
        ifs.close();
        mpc.ProfilerPopState(false); // file_io

        remainder = n1 % bsize;
        if (remainder > 0) {
            g.SetDims(remainder, m3);
            g_mask.SetDims(remainder, m3);
            miss.SetDims(remainder, m3);
            miss_mask.SetDims(remainder, m3);
            Qsub.SetDims(remainder, kp);
            Qsub_mask.SetDims(remainder, kp);
            
            mpc.Transpose(Qsub);
            transpose(Qsub_mask, Qsub_mask);

            mpc.BeaverMult(gQ, Qsub, Qsub_mask, g, g_mask);
            mpc.BeaverMult(gQ_adj, Qsub, Qsub_mask, miss, miss_mask);
        }

        Qsub.kill();
        Qsub_mask.kill();
        g.kill();
        g_mask.kill();
        miss.kill();
        miss_mask.kill();

        mpc.BeaverReconstruct(gQ);
        mpc.BeaverReconstruct(gQ_adj);

        mpc.ProfilerPopState(false); // data_scan2

        Mat<ZZ_p> gQ_adj_mask;
        mpc.BeaverPartition(gQ_adj_mask, gQ_adj);

        Mat<ZZ_p> gQ_adj_gmean;
        Init(gQ_adj_gmean, kp, m3);
        for i in range(kp):
            mpc.BeaverMultElem(gQ_adj_gmean[i], gQ_adj[i], gQ_adj_mask[i],
                            g_mean_pca, g_mean_pca_mask);
        }
        mpc.BeaverReconstruct(gQ_adj_gmean);
        mpc.Trunc(gQ_adj_gmean);

        if (mpc.pid > 0) {
            gQ -= gQ_adj_gmean;
        }
        gQ_adj_gmean.kill();

        Mat<ZZ_p> gQ_mask;
        mpc.BeaverPartition(gQ_mask, gQ);

        Mat<ZZ_p> gQ_scaled;
        gQ_scaled.SetDims(kp, m3);
        clear(gQ_scaled);
        for i in range(kp):
            mpc.BeaverMultElem(gQ_scaled[i], gQ[i], gQ_mask[i], g_stdinv_pca, g_stdinv_pca_mask);
        }
        mpc.BeaverReconstruct(gQ_scaled);
        mpc.Trunc(gQ_scaled);

        mpc.ProfilerPushState("qr_m");
        mpc.OrthonormalBasis(Q, gQ_scaled);
        mpc.ProfilerPopState(false); // qr_m

        print("Iter " << pit + 1 << " complete, "; toc();
        tic();
        }

        with open(get_cache_path(pid, "piter")) as f:
        if (mpc.pid > 0) {
        mpc.write_to_file(gQ, fs);
        }
        fs.close();

    }

    mpc.ProfilerPopState(true); // power_iter
    print("Power iteration complete")

    Mat<ZZ_p> Z = gQ;
    gQ.kill();

    print("Data projected to subspace")
    if (param.DEBUG) {
        print("Z")
        mpc.PrintFP(Z[0], 5);
    }

    Mat<ZZ_p> V;
    Init(V, k, n1);

    /* Eigendecomposition */
    if (os.path.exists(get_cache_path(pid, "eigen"))) {

        print("eigen cache found")
        with open(get_cache_path(pid, "eigen"));
        mpc.ReadFromFile(V, ifs, k, n1);
        ifs.close();

    } else {

        ZZ_p fp_m2_inv:Zp = mpc.double_to_fp(1 / ((double) m2), param.NBIT_K, param.NBIT_F);
        Z *= fp_m2_inv;
        mpc.Trunc(Z);

        mpc.Transpose(Z); // kp-by-n1

        Mat<ZZ_p> Z_mask;
        mpc.BeaverPartition(Z_mask, Z);

        /* Form covariance matrix */
        Mat<ZZ_p> Z_gram;
        Init(Z_gram, kp, kp);
        for i in range(kp):
        mpc.BeaverMult(Z_gram[i], Z, Z_mask, Z[i], Z_mask[i]);
        }
        mpc.BeaverReconstruct(Z_gram);
        mpc.Trunc(Z_gram);

        print("Constructed reduced eigenvalue problem")

        if (param.DEBUG) {
        print("Z_gram")
        mpc.PrintFP(Z_gram[0], 5);
        }

        mpc.ProfilerPushState("eigen_solve");

        Mat<ZZ_p> U;
        Vec<ZZ_p> L;
        mpc.EigenDecomp(U, L, Z_gram);
        Z_gram.kill();

        // Select top eigenvectors and eigenvalues
        U.SetDims(k, kp);
        L.SetLength(k);

        print("Selected K eigenvectors")
        mpc.ProfilerPopState(false); // eigen_solve

        if (param.DEBUG) {
        mpc.PrintFP(U[0], 5);
        }

        // Recover singular vectors
        Mat<ZZ_p> U_mask;
        mpc.BeaverPartition(U_mask, U);

        mpc.BeaverMultMat(V, U, U_mask, Z, Z_mask);
        U.kill();
        U_mask.kill();
        Z_mask.kill();
        mpc.BeaverReconstruct(V);
        mpc.Trunc(V);

        with open(get_cache_path(pid, "eigen")) as f:
        if (mpc.pid > 0) {
        mpc.write_to_file(V, fs);
        }
        fs.close();

    }

    Z.kill();

    mpc.ProfilerPopState(true); // pop_strat

    mpc.ProfilerPushState("assoc_test");
    mpc.ProfilerPushState("covar");

    // Concatenate covariate matrix and jointly orthogonalize
    mpc.Transpose(cov);
    V.SetDims(k + param.NUM_COVS, n1);
    if (mpc.pid > 0) {
        for i in range(param.NUM_COVS):
        V[k + i] = cov[i] * fp_one;
        }
    }
    cov.kill();
    mpc.OrthonormalBasis(V, V);

    Mat<ZZ_p> V_mask;
    mpc.BeaverPartition(V_mask, V);

    print("Bases for top singular vectors and covariates calculated")
    mpc.ProfilerPopState(false); // covar

    if (param.DEBUG) {
        mpc.PrintBeaverFP(V[0], V_mask[0], 5);
    }

    /* Pass 4: Calculate GWAS statistics */

    Vec<ZZ_p> pheno_mask;
    mpc.BeaverPartition(pheno_mask, pheno);

    Vec<ZZ_p> Vp;
    Init(Vp, k + param.NUM_COVS);
    mpc.BeaverMult(Vp, V, V_mask, pheno, pheno_mask);
    mpc.BeaverReconstruct(Vp);

    Vec<ZZ_p> Vp_mask;
    mpc.BeaverPartition(Vp_mask, Vp);
    
    Vec<ZZ_p> VVp;
    Init(VVp, n1);
    mpc.BeaverMult(VVp, Vp, Vp_mask, V, V_mask);
    mpc.BeaverReconstruct(VVp);
    mpc.Trunc(VVp);

    Vec<ZZ_p> VVp_mask;
    mpc.BeaverPartition(VVp_mask, VVp);

    Vec<ZZ_p> p_hat, p_hat_mask;
    p_hat = fp_one * pheno - VVp;
    p_hat_mask = fp_one * pheno_mask - VVp_mask;

    Vp.kill();
    Vp_mask.kill();
    VVp.kill();
    VVp_mask.kill();

    print("Phenotypes corrected")

    Vec<ZZ_p> V_sum, V_sum_mask;
    Init(V_sum, k + param.NUM_COVS);
    Init(V_sum_mask, k + param.NUM_COVS);
    for i in range(k + param.NUM_COVS):
        for j in range(n1):
        V_sum[i] += V[i][j];
        V_sum_mask[i] += V_mask[i][j];
        }
    }

    Vec<ZZ_p> u;
    Init(u, n1);
    mpc.BeaverMult(u, V_sum, V_sum_mask, V, V_mask);
    mpc.BeaverReconstruct(u);
    mpc.Trunc(u);
    if (mpc.pid > 0) {
        u *= -1;
        mpc.AddPublic(u, fp_one);
    }

    Vec<ZZ_p> u_mask;
    mpc.BeaverPartition(u_mask, u);

    if (param.DEBUG) {
        print("u")
        mpc.PrintBeaverFP(u, u_mask, 10);
    }

    print("Allocating sx, sxx, sxp, B ... ";

    Vec<ZZ_p> sx, sxx, sxp;
    Mat<ZZ_p> B;
    Init(sx, m2);
    Init(sxx, m2);
    Init(sxp, m2);
    Init(B, k + param.NUM_COVS, m2);

    print("done.";

    mpc.ProfilerPushState("data_scan");

    if (os.path.exists(get_cache_path(pid, "gwas_stats"))) {
        print("GWAS statistics cache found")
        with open(get_cache_path(pid, "gwas_stats"));
        mpc.ReadFromFile(sx, ifs, m2);
        mpc.ReadFromFile(sxx, ifs, m2);
        mpc.ReadFromFile(sxp, ifs, m2);
        mpc.ReadFromFile(B, ifs, k + param.NUM_COVS, m2);
        ifs.close();

    } else {

        with open(get_cache_path(pid, "input_geno"));
        if (mpc.pid > 0) {
        mpc.import_seed(10, ifs);
        } else {
        for (int p = 1; p <= 2; p++) {
            mpc.import_seed(10 + p, ifs);
        }
        }

        long bsize = param.PITER_BATCH_SIZE;

        print("Allocating batch variables ... ";

        Mat<ZZ_p> dosage, dosage_mask;
        Init(dosage, bsize, m2);
        Init(dosage_mask, bsize, m2);

        Vec<ZZ_p> u_vec, u_mask_vec, p_hat_vec, p_hat_mask_vec;
        Init(u_vec, bsize);
        Init(u_mask_vec, bsize);
        Init(p_hat_vec, bsize);
        Init(p_hat_mask_vec, bsize);

        mpc.Transpose(V); // n1-by-(k + NUM_COVS)
        transpose(V_mask, V_mask);

        Mat<ZZ_p> V_sub, V_mask_sub;
        Init(V_sub, bsize, k + param.NUM_COVS);
        Init(V_mask_sub, bsize, k + param.NUM_COVS);

        print("done.")

        Vec<bool> gkeep3;
        gkeep3.SetLength(m0);
        for j in range(m0):
        gkeep3[j] = gkeep1[j] == 1;
        }

        ind = 0;
        for j in range(m0):
        if (gkeep3[j]) {
            gkeep3[j] = gkeep2[ind] == 1;
            ind++;
        }
        }

        ind = -1;
        tic();
        mpc.ProfilerPushState("file_io/rng");
        print("GWAS pass:")
        for (int cur = 0; cur < n1; cur++) {
        ind++;

        Mat<ZZ_p> g0, g0_mask;
        Vec<ZZ_p> miss0, miss0_mask;

        while (ikeep[ind] != 1) {
            if (mpc.pid > 0) {
            mpc.SkipData(ifs, 3, m0); // g
            mpc.SkipData(ifs, m0); // miss

            mpc.switch_seed(10);
            Matrix(g0_mask, 3, m0);
            mpc.random_vector(miss0_mask, m0);
            mpc.restore_seed();
            } else {
            for (int p = 1; p <= 2; p++) {
                mpc.switch_seed(10 + p);
                Matrix(g0_mask, 3, m0);
                mpc.random_vector(miss0_mask, m0);
                mpc.restore_seed();
            }
            }
            ind++;
        }

        if (mpc.pid > 0) {
            mpc.ReadFromFile(g0, ifs, 3, m0); // g
            mpc.ReadFromFile(miss0, ifs, m0); // miss

            mpc.switch_seed(10);
            Matrix(g0_mask, 3, m0);
            mpc.random_vector(miss0_mask, m0);
            mpc.restore_seed();
        } else {
            Init(g0, 3, m0);
            Init(g0_mask, 3, m0);
            Init(miss0, m0);
            Init(miss0_mask, m0);

            for (int p = 1; p <= 2; p++) {
            mpc.switch_seed(10 + p);
            Matrix(tmp_mat, 3, m0);
            mpc.random_vector(tmp_vec, m0);
            mpc.restore_seed();

            g0_mask += tmp_mat;
            miss0_mask += tmp_vec;
            }
        }
        
        Mat<ZZ_p> g, g_mask;
        Vec<ZZ_p> miss, miss_mask;
        g.SetDims(3, m2);
        miss.SetLength(m2);
        g_mask.SetDims(3, m2);
        miss_mask.SetLength(m2);
        int ind2 = 0;
        for j in range(m0):
            if (gkeep3[j]) {
            for (int k = 0; k < 3; k++) {
                g[k][ind2] = g0[k][j];
                g_mask[k][ind2] = g0_mask[k][j];
            }
            miss[ind2] = miss0[j];
            miss_mask[ind2] = miss0_mask[j];
            ind2++;
            }
        }

        dosage[cur % bsize] = g[1] + 2 * g[2];
        dosage_mask[cur % bsize] = g_mask[1] + 2 * g_mask[2];

        u_vec[cur % bsize] = u[cur];
        u_mask_vec[cur % bsize] = u_mask[cur];
        p_hat_vec[cur % bsize] = p_hat[cur];
        p_hat_mask_vec[cur % bsize] = p_hat_mask[cur];

        V_sub[cur % bsize] = V[cur];
        V_mask_sub[cur % bsize] = V_mask[cur];

        if (cur % bsize == bsize - 1) {
            mpc.ProfilerPopState(false); // file_io/rng

            mpc.BeaverMult(sx, u_vec, u_mask_vec, dosage, dosage_mask);
            mpc.BeaverMult(sxp, p_hat_vec, p_hat_mask_vec, dosage, dosage_mask);

            Mat<ZZ_p> sxx_tmp;
            Init(sxx_tmp, bsize, m2);
            mpc.BeaverMultElem(sxx_tmp, dosage, dosage_mask, dosage, dosage_mask);
            for (int b = 0; b < bsize; b++) {
            sxx += sxx_tmp[b];
            }
            sxx_tmp.kill();

            mpc.Transpose(V_sub); // (k + NUM_COVS)-by-bsize
            transpose(V_mask_sub, V_mask_sub);

            mpc.BeaverMult(B, V_sub, V_mask_sub, dosage, dosage_mask);

            print("\t" << cur+1 << " / " << n1 << ", "; toc(); tic();

            Init(dosage, bsize, m2);
            Init(dosage_mask, bsize, m2);
            Init(V_sub, bsize, k + param.NUM_COVS);
            Init(V_mask_sub, bsize, k + param.NUM_COVS);

            mpc.ProfilerPushState("file_io/rng");
        }
        }
        ifs.close();
        mpc.ProfilerPopState(false); // file_io/rng

        long remainder = n1 % bsize;
        if (remainder > 0) {
        dosage.SetDims(remainder, m2);
        dosage_mask.SetDims(remainder, m2);
        u_vec.SetLength(remainder);
        u_mask_vec.SetLength(remainder);
        p_hat_vec.SetLength(remainder);
        p_hat_mask_vec.SetLength(remainder);
        V_sub.SetDims(remainder, k + param.NUM_COVS);
        V_mask_sub.SetDims(remainder, k + param.NUM_COVS);

        mpc.BeaverMult(sx, u_vec, u_mask_vec, dosage, dosage_mask);
        mpc.BeaverMult(sxp, p_hat_vec, p_hat_mask_vec, dosage, dosage_mask);

        Mat<ZZ_p> sxx_tmp;
        Init(sxx_tmp, remainder, m2);
        mpc.BeaverMultElem(sxx_tmp, dosage, dosage_mask, dosage, dosage_mask);
        for (int b = 0; b < remainder; b++) {
            sxx += sxx_tmp[b];
        }
        sxx_tmp.kill();

        mpc.Transpose(V_sub); // (k + NUM_COVS)-by-remainder
        transpose(V_mask_sub, V_mask_sub);

        mpc.BeaverMult(B, V_sub, V_mask_sub, dosage, dosage_mask);

        print("\t" << n1 << " / " << n1 << ", "; toc(); tic();
        }

        mpc.BeaverReconstruct(sx);
        mpc.BeaverReconstruct(sxp);
        mpc.BeaverReconstruct(sxx);
        mpc.BeaverReconstruct(B);
        sxx *= fp_one;

        with open(get_cache_path(pid, "gwas_stats")) as f:
        mpc.write_to_file(sx, fs);
        mpc.write_to_file(sxx, fs);
        mpc.write_to_file(sxp, fs);
        mpc.write_to_file(B, fs);
        fs.close();

        print("Wrote results to cache")
    }

    mpc.ProfilerPopState(true); // data_scan

    if (param.DEBUG) {
        print("sx")
        mpc.PrintFP(sx, 3);
        print("sxp")
        mpc.PrintFP(sxp, 3);
        print("sxx")
        mpc.PrintFP(sxx, 3);
        print("B")
        mpc.PrintFP(B, 3, 3);
    }

    mpc.Transpose(B); // m2-by-(k + param.NUM_COVS)

    Vec<ZZ_p> BB;
    mpc.InnerProd(BB, B); // m2
    mpc.Trunc(BB);
    if (mpc.pid > 0) {
        sxx -= BB;
    }

    ZZ_p sp(0);
    if (mpc.pid > 0) {
        for i in range(n1):
        sp += p_hat_mask[i];
        if (mpc.pid == 1) {
            sp += p_hat[i];
        }
        }
    }

    ZZ_p spp(0);
    mpc.BeaverInnerProd(spp, p_hat, p_hat_mask);
    mpc.BeaverReconstruct(spp);

    ZZ_p fp_n1_inv:Zp = mpc.double_to_fp(1 / ((double) n1), param.NBIT_K, param.NBIT_F);
    sx *= fp_n1_inv;
    sp *= fp_n1_inv;

    mpc.Trunc(sx);
    mpc.Trunc(sp);
    mpc.Trunc(spp);

    Vec<ZZ_p> sx_mask;
    mpc.BeaverPartition(sx_mask, sx);

    ZZ_p sp_mask;
    mpc.BeaverPartition(sp_mask, sp);

    Vec<ZZ_p> spsx, sx2;
    ZZ_p sp2(0);
    Init(spsx, m2);
    Init(sx2, m2);

    mpc.BeaverMult(spsx, sx, sx_mask, sp, sp_mask);
    mpc.BeaverMult(sp2, sp, sp_mask, sp, sp_mask);
    mpc.BeaverMultElem(sx2, sx, sx_mask, sx, sx_mask);

    mpc.BeaverReconstruct(spsx);
    mpc.BeaverReconstruct(sp2);
    mpc.BeaverReconstruct(sx2);

    spsx *= n1;
    sp2 *= n1;
    sx2 *= n1;

    mpc.Trunc(spsx);
    mpc.Trunc(sp2);
    mpc.Trunc(sx2);

    Vec<ZZ_p> numer, denom;
    Init(numer, m2);
    Init(denom, m2 + 1);
    if (mpc.pid > 0) {
        numer = sxp - spsx;
        for i in range(m2):
        denom[i] = sxx[i] - sx2[i];
        }
        denom[m2] = spp - sp2;
    }

    Vec<ZZ_p> denom1_sqrt_inv;
    if (os.path.exists(get_cache_path(pid, "denom_inv"))) {
        print("denom_inv cache found")
        with open(get_cache_path(pid, "denom_inv"));
        mpc.ReadFromFile(denom1_sqrt_inv, ifs, denom.length());
        ifs.close();
    } else {
        mpc.ProfilerPushState("sqrt");
        mpc.FPSqrt(tmp_vec, denom1_sqrt_inv, denom);
        mpc.ProfilerPopState(false); // sqrt

        with open(get_cache_path(pid, "denom_inv")) as f:
        if (mpc.pid > 0) {
        mpc.write_to_file(denom1_sqrt_inv, fs);
        }
        fs.close();
    }

    denom.kill();
    tmp_vec.kill();

    ZZ_p denom2_sqrt_inv = denom1_sqrt_inv[m2]; // p term
    denom1_sqrt_inv.SetLength(m2); // truncate

    Vec<ZZ_p> z;
    mpc.MultElem(z, numer, denom1_sqrt_inv);
    mpc.Trunc(z);

    mpc.MultMat(z, z, denom2_sqrt_inv);
    mpc.Trunc(z);

    mpc.ProfilerPopState(false); // assoc_test

    print("Association statistics calculated")
    mpc.reveal_sym(z);
    if (mpc.pid == 2) {
        Vec<double> z_double;
        FPToDouble(z_double, z, param.NBIT_K, param.NBIT_F);
        owith open(get_output_path("assoc"), ios::out);
        for i in range(z_double.length()):
        ofs << z_double[i])
        }
        ofs.close();
        print("Result written to " << get_output_path("assoc"))
    }

    mpc.ProfilerPopState(true); // main

    return true;
    }