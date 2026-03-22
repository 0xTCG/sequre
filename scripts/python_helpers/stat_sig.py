import numpy as np
import statsmodels.api as sm


def sign(x):
    return -1 if x < 0 else 1 if x > 0 else 0


def get_logreg_stats(X, y):
    X2 = sm.add_constant(X)
    est = sm.Logit(y, X2)
    est2 = est.fit(disp=0)
    return est2.params, est2.pvalues


def get_linreg_stats(X, y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2.params, est2.pvalues


def count_discrepancies(data_instance, mdni_stats, sequre_stats):
    mdni_coeff, mdni_pval = mdni_stats
    sequre_coeff, sequre_pval = sequre_stats

    discr = 0
    for i in range(len(mdni_coeff)):
        if mdni_pval[i] <= 0.05 and (sequre_pval[i] > 0.05 or sign(mdni_coeff[i]) != sign(sequre_coeff[i])):
            print(f"Discrepancy found at var {i} and instance {data_instance}! Reference MICE p_val: {mdni_pval[i]}, Sequre p_val: {sequre_pval[i]}, Reference MICE coeff: {mdni_coeff[i]}, Sequre coeff: {sequre_coeff[i]}")
            discr += 1
        else:
            print(f"No discrepancy found at var {i} and instance {data_instance}:\n\t\tReference MICE p_val: {mdni_pval[i]}\n\t\tSequre p_val: {sequre_pval[i]}\n\t\tReference MICE coeff: {mdni_coeff[i]}\n\t\tSequre coeff: {sequre_coeff[i]}")
    
    return discr


# imputed_instances = 10
# number_of_variables = 10
# complete_data = np.loadtxt("data/mi/input_matrix.txt")
# y = complete_data @ np.ones((number_of_variables, 1)) + 1.0
# y += np.random.normal(size=y.shape)  # Add some noise
# discr = 0

# for i in range(1, imputed_instances + 1):
#     X_mdni = np.loadtxt(f"data/mi/output_matrix_{i}.txt")
#     X_sequre = np.loadtxt(f"data/mi/sequre_output_{i}.txt")

#     mdni_stats = get_stats(X_mdni, y)
#     sequre_stats = get_stats(X_sequre, y)

#     discr += count_discrepancies(i, mdni_stats, sequre_stats)

# print("Total number of discrepancies:", discr)


scenario = "real"  # or scenario_1 or etc.
space = "balanced_mimic" # 500
imputed_instances = 5
labels = np.loadtxt(f"data/mi/{scenario}/{space}/{'labels' if scenario == 'real' else 'non_tampered_labels'}.txt")
# X_mice = np.loadtxt(f"data/mi/{scenario}/{space}/py_mice_data.txt")
X_mice = np.loadtxt(f"data/mi/{scenario}/py_mice_data.txt")
mice_stats = get_logreg_stats(X_mice, labels) if "mimic" in space else get_linreg_stats(X_mice, labels)

discr = 0
for i in range(1, imputed_instances + 1):
    X_sequre = np.loadtxt(f"data/mi/dump/sequre_imputed_dataset_{scenario}_{i}.txt")
    sequre_stats = get_logreg_stats(X_sequre, labels) if "mimic" in space else get_linreg_stats(X_sequre, labels)
    discr += count_discrepancies(i, mice_stats, sequre_stats)

print("Total number of discrepancies:", discr, "Imputed instances:", imputed_instances)
