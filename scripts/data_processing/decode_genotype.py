from tqdm import tqdm


folder_path = 'data/gwas/input/lung/'
geno_path = 'geno_raw.txt'


with open(f'{folder_path}/{geno_path}') as geno_file, open(f'{folder_path}/geno.txt', 'w') as geno_out, open(f'{folder_path}/miss.txt', 'w') as miss_out:
    for geno_line in tqdm(geno_file):
        elements = geno_line.split()
        
        geno_submat = [['0' for _ in range(len(elements))] for _ in range(3)]
        miss_row = ['0' for _ in range(len(elements))]

        for i, elem in enumerate(elements):
            elem_int = int(elem.strip())

            if elem_int in {0, 1, 2}: geno_submat[elem_int][i] = '1'
            else: miss_row[i] = '1'
        
        for row in geno_submat:
            geno_out.write(' '.join(row) + '\n')
        miss_out.write(' '.join(miss_row) + '\n')
