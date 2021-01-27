def parallel_logistic_regression(
        self: 'MPCEnv', xr: Matrix, xm: Matrix, vr: Matrix,
        vm: Matrix, yr: Vector, ym: Vector, max_iter: int) -> tuple:
        n: int = vr.shape[1]
        p: int = vr.shape[0]
        c: int = xr.shape[0]
        assert vm.shape[0] == p
        assert vm.shape[1] == n
        assert xm.shape[0] == c
        assert xm.shape[1] == n
        assert xr.shape[1] == n
        assert len(yr) == n
        assert len(ym) == n

        b0 = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])
        bv = Matrix(c, p)
        bx = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])

        yneg_r = -yr
        yneg_m = -ym
        if self.pid > 0:
            for i in range(n):
                yneg_r[i] += 1

        yneg = deepcopy(yneg_m)
        if self.pid == 1:
            for i in range(n):
                yneg[i] += yneg_r[i]

        fp_memory: Zp = self.double_to_fp(0.5, param.NBIT_K, param.NBIT_F, fid=0)
        fp_one: Zp = self.double_to_fp(1, param.NBIT_K, param.NBIT_F, fid=0)
        eta: float = 0.3

        step0 = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])
        stepv = Matrix(c, p)
        stepx = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])

        nbatch: int = 10
        batch_size: int = (n + nbatch - 1) // nbatch

        for it in range(max_iter):
            print(f'Logistic regression iteration {it} initialized')
            batch_index: int = it % nbatch
            start_ind: int = batch_size * batch_index
            end_ind: int = start_ind + batch_size
            if end_ind > n:
                end_ind = n
            cur_bsize: int = end_ind - start_ind

            xr_batch = Matrix(c, cur_bsize)
            xm_batch = Matrix(c, cur_bsize)
            vr_batch = Matrix(p, cur_bsize)
            vm_batch = Matrix(p, cur_bsize)
            yn_batch = Vector([Zp(0, base=param.BASE_P) for _ in range(cur_bsize)])
            ynr_batch = Vector([Zp(0, base=param.BASE_P) for _ in range(cur_bsize)])
            ynm_batch = Vector([Zp(0, base=param.BASE_P) for _ in range(cur_bsize)])

            for j in range(c):
                for i in range(cur_bsize):
                    xr_batch[j][i].value = xr[j][start_ind + i].value
                    xm_batch[j][i].value = xm[j][start_ind + i].value

            for j in range(p):
                for i in range(cur_bsize):
                    vr_batch[j][i].value = vr[j][start_ind + i].value
                    vm_batch[j][i].value = vm[j][start_ind + i].value

            for i in range(cur_bsize):
                yn_batch[i].value = yneg[start_ind + i].value
                ynr_batch[i].value = yneg_r[start_ind + i].value
                ynm_batch[i].value = yneg_m[start_ind + i].value

            fp_bsize_inv: Zp = self.double_to_fp(eta * (1 / cur_bsize), param.NBIT_K, param.NBIT_F, fid=0)

            bvr, bvm = self.beaver_partition(bv, fid=0)
            bxr, bxm = self.beaver_partition(bx, fid=0)

            h: Matrix = self.beaver_mult(bvr, bvm, vr_batch, vm_batch, False, fid=0)
            for j in range(c):
                xrvec = xr_batch[j] * fp_one
                xmvec = xm_batch[j] * fp_one
                h[j] += self.beaver_mult_vec(xrvec, xmvec, bxr[j], bxm[j], fid=0)
            h: Matrix = self.beaver_reconstruct(h, fid=0)
            self.trunc(h, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            for j in range(c):
                h[j] += b0[j]

            hvec = Matrix().from_value(h).flatten()
            _, s_grad_vec = self.neg_log_sigmoid(hvec, fid=0)

            s_grad = Matrix().from_value(Vector([s_grad_vec], deep_copy=True))
            s_grad.reshape(c, cur_bsize)

            d0 = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])
            dv = Matrix(c, p)
            dx = Vector([Zp(0, base=param.BASE_P) for _ in range(c)])

            for j in range(c):
                s_grad[j] += yn_batch * fp_one
                d0[j] = sum(s_grad[j], Zp(0, base=param.BASE_P))

            s_grad_r, s_grad_m = self.beaver_partition(s_grad, fid=0)

            for j in range(c):
                dx[j] = self.beaver_inner_prod_pair(
                    xr_batch[j], xm_batch[j], s_grad_r[j], s_grad_m[j], fid=0)
            dx = self.beaver_reconstruct(dx, fid=0)

            vr_batch.transpose(inplace=True)
            vm_batch.transpose(inplace=True)
            dv: Matrix = self.beaver_mult(s_grad_r, s_grad_m, vr_batch, vm_batch, False, fid=0)
            dv: Matrix = self.beaver_reconstruct(dv, fid=0)
            self.trunc(dv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            step0: Vector = step0 * fp_memory - d0 * fp_bsize_inv
            stepv: Matrix = stepv * fp_memory - dv * fp_bsize_inv
            stepx: Vector = stepx * fp_memory - dx * fp_bsize_inv
            self.trunc_vec(step0, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            self.trunc(stepv, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)
            self.trunc_vec(stepx, param.NBIT_K + param.NBIT_F, param.NBIT_F, fid=0)

            b0: Vector = b0 + step0
            bv: Matrix = Matrix().from_value(bv + stepv)
            bx: Vector = bx + stepx
    
        return b0, bv, bx

    def neg_log_sigmoid(self: 'MPCEnv', a: Vector, fid: int) -> tuple:
        n: int = len(a)
        depth: int = 6
        step: float = 4
        cur: Vector = deepcopy(a)
        a_ind = Vector([Zp(0, base=self.primes[fid]) for _ in range(len(a))])

        for i in range(depth):
            cur_sign: Vector = self.is_positive(cur)
            index_step = Zp(1 << (depth - 1 - i), base=self.primes[fid])

            for j in range(n):
                a_ind[j] += cur_sign[j] * index_step

            cur_sign *= 2
            if self.pid == 1:
                for j in range(n):
                    cur_sign[j] -= 1

            step_fp: Zp = self.double_to_fp(
                step, param.NBIT_K, param.NBIT_F, fid=fid)

            for j in range(n):
                cur[j] -= step_fp * cur_sign[j]

            step //= 2

        if self.pid == 1:
            for j in range(n):
                a_ind[j] += 1

        params: Matrix = self.table_lookup(a_ind, 2, fid=0)

        b: Vector = self.mult_vec(params[1], a, fid=fid)
        self.trunc_vec(b)

        if self.pid > 0:
            for j in range(n):
                b[j] += params[0][j]

        b_grad = deepcopy(params[1])

        return b, b_grad


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
