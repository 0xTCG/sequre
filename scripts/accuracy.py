import numpy as np
import os


target_dir = "results"
exps = ["king_CP1_ind_32_snps_278528", "pca_projection_CP1_ind_524288_snps_524288", "gwas_assoc_CP1_ind_128_snps_524288"]
techs = ["mpc", "mhe"]

for exp in exps:
    raw_path = os.path.join(target_dir, f"raw_{exp}.txt")
    raw_res = np.nan_to_num(np.loadtxt(raw_path))
    for tech in techs:
        tech_path = os.path.join(target_dir, f"{tech}_{exp}.txt")
        tech_res = np.nan_to_num(np.loadtxt(tech_path))
        mean_diff = np.mean(np.abs(tech_res - raw_res))

        print(f"Mean diff in {exp}_{tech}: {mean_diff}")
