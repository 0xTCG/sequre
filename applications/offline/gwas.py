import numpy as np

from sklearn.decomposition import PCA
from scipy.linalg import orth


def read_vector(f, length):
    return [int(e.strip()) for e in next(f).split()[:length]]


def read_matrix(f, rows, cols):
    return [read_vector(f, cols) for _ in range(rows)]


def read_matrix_interleaved(f, rows, cols, multiplicity):
    deinterleaved_matrices = [[] for _ in range(multiplicity)]
    
    for _ in range(rows):
        for i in range(multiplicity):
            deinterleaved_matrices[i].append(read_vector(f, cols))

    return deinterleaved_matrices


def load_dosage(f_geno, f_miss, rows, cols):
    g = np.array(read_matrix_interleaved(f_geno, rows, cols, 3))
    miss = np.array(read_matrix(f_miss, rows, cols))
    dosage = g[1] + g[2] * 2

    return miss, dosage


def householder(x):
    u = x.copy()
    u[0] += np.linalg.norm(x) * np.sign(x[0])
    return u / np.linalg.norm(u)


def orthonormal_basis(A):
    v_cache = []
    m, n = A.shape
    Q = np.pad(np.identity(m), ((0,0),(0,n - m)), 'constant')

    for i in range(len(A)):
        v = np.nan_to_num(np.expand_dims(householder(A[0]), axis=0))
        B = A - A @ v.T @ v * 2
        A = B[1:, 1:]
        v_cache.append(v)

    for i in range(len(Q) - 1, -1, -1):
        Qsub = Q[:, i:]
        Q[:, i:] = Qsub - Qsub @ v_cache[i].T @ v_cache[i] * 2
    
    return Q


def offline_gwas(pheno_path, cov_path, geno_path, miss_path, num_inds, num_snps, num_covs, top_components):
    f_pheno = open(pheno_path)
    f_cov   = open(cov_path)
    f_geno  = open(geno_path)
    f_miss  = open(miss_path)
    
    pheno = np.array(read_vector(f_pheno, num_inds))
    cov = np.array(read_matrix(f_cov, num_inds, num_covs))
    miss, dosage = load_dosage(f_geno, f_miss, num_inds, num_snps)

    # TODO: Implement QC
    
    maf = np.nan_to_num(dosage.sum(axis=0) / ((num_inds - miss.sum(axis=0)) * 2))
    g_std_bern = np.sqrt(maf * (1 - maf))
    standardized_dosage = np.nan_to_num((dosage - (1 - miss) * maf) / g_std_bern)

    pca_components = PCA(top_components).fit_transform(standardized_dosage)
    V = orthonormal_basis(np.hstack((pca_components, cov)).T)

    p_hat = np.expand_dims(pheno, axis=0) - np.expand_dims(pheno, axis=0) @ V.T @ V
    sp = p_hat.flatten().sum()
    sx = ((1 - np.expand_dims(V.T.sum(axis=0), axis=0) @ V) @ dosage)[0]
    spp = np.dot(p_hat.flatten(), p_hat.flatten())
    sxp = (p_hat @ dosage).flatten()
    V_dosage = V @ dosage
    sxx = (dosage * dosage).sum(axis=0) - (V_dosage * V_dosage).sum(axis=0)
    norm_sp = sp / len(pheno)
    numer = sxp - sx * norm_sp
    denom = (sxx - sx * sx / len(pheno)) * (spp - sp * norm_sp)
    
    return np.nan_to_num(numer / np.sqrt(denom))


kwargs = {
    'pheno_path': 'data/gwas/input/lung_reduced/pheno.txt',
    'cov_path': 'data/gwas/input/lung_reduced/cov.txt',
    'geno_path': 'data/gwas/input/lung_reduced/geno.txt',
    'miss_path': 'data/gwas/input/lung_reduced/miss.txt',
    'num_inds': 1000,
    'num_snps': 30000,
    'num_covs': 10,
    'top_components': 5
}

with open('log.txt', 'w') as f:
    assoc = offline_gwas(**kwargs)
    f.write('\n'.join([str(e) for e in assoc]))
