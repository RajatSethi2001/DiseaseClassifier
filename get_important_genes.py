import h5py
import mygene
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from utils import clean_dose_unit, get_zscores, get_minmax, get_zscore_minmax

def main():
    gctx_file = "Data/annotated_GSE92742_Broad_LINCS_Level5_COMPZ_n473647x12328.gctx"
    sample_condition_csv = "Conditions/Healthy/SRR16316900.csv"
    gctx_fp = h5py.File(gctx_file, "r")
    data_limit = 10000
    num_genes = 1500

    print("Loading GCTX Data")
    data_idx = random.sample(range(0, len(gctx_fp["0/META/COL/id"])), data_limit)
    data_idx.sort()
    data_idx = np.array(data_idx)

    ids = [s.decode('utf-8') for s in gctx_fp["0/META/COL/id"][data_idx]]
    pert_dose_units = [clean_dose_unit(s) for s in gctx_fp["0/META/COL/pert_dose_unit"][data_idx]]
    pert_time_units = [s.decode('utf-8') for s in gctx_fp["0/META/COL/pert_time_unit"][data_idx]]
    pert_types = [s.decode('utf-8') for s in gctx_fp["0/META/COL/pert_type"][data_idx]]
    gene_symbols = np.array([s.decode("utf-8") for s in gctx_fp["/0/META/ROW/pr_gene_symbol"]])

    print("Matching CTL and TRT groups")
    data_map = {}
    for idx in range(data_limit):
        id_parts = ids[idx].split(":")
    
        if len(id_parts) < 2:
            continue

        condition = id_parts[0]
        pert_type = pert_types[idx]
        
        if condition not in data_map:
            data_map[condition] = {"ctl_idx": [], "trt_idx": []}
        
        if pert_type == "ctl_untrt" or pert_type == "ctl_vehicle":
            data_map[condition]["ctl_idx"].append(idx)

        elif pert_type == "trt_cp":
            pert_dose_unit = pert_dose_units[idx]
            pert_time_unit = pert_time_units[idx]

            if pert_dose_unit == "uM" and pert_time_unit == "h":
                data_map[condition]["trt_idx"].append(idx)

    print("Calculting Delta Exprs")
    delta_exprs = []
    for id, gene_data in data_map.items():
        for ctl_idx in gene_data["ctl_idx"]:
            ctl_expr = get_zscore_minmax(gctx_fp["0/DATA/0/matrix"][ctl_idx, :])
            for trt_idx in gene_data["trt_idx"]:
                trt_expr = get_zscore_minmax(gctx_fp["0/DATA/0/matrix"][trt_idx, :])
                delta_exprs.append(trt_expr - ctl_expr)
    
    delta_exprs = np.array(delta_exprs)

    print("Running PCA")
    pca = PCA()
    pca.fit(delta_exprs)
    loadings = np.abs(pca.components_)
    gene_scores = loadings[:20].sum(axis=0)
    top_gene_indices = np.argsort(gene_scores)[::-1]
    top_gene_symbols = gene_symbols[top_gene_indices]

    print("Converting gene symbols to ENSEMBL")
    mg = mygene.MyGeneInfo()
    results = mg.querymany(top_gene_symbols, scopes='symbol', fields='ensembl.gene', species='human')
    condition_df = pd.read_csv(sample_condition_csv, index_col=0)
    condition_ensembl_ids = {ensembl_id.split(".")[0] for ensembl_id in condition_df.index}

    important_genes = set()
    results_idx = 0
    while len(important_genes) < num_genes and results_idx < len(results):
        result = results[results_idx]
        if 'ensembl' in result:
            if isinstance(result['ensembl'], list):
                ensembl_id = result['ensembl'][0]['gene']
            else:
                ensembl_id = result['ensembl']['gene']
            
            if ensembl_id in condition_ensembl_ids:
                important_genes.add((str(top_gene_symbols[results_idx]), ensembl_id))
        results_idx += 1
    
    with open("Data/important_genes.csv", "w") as f:
        for gene_symbol, ensembl_id in important_genes:
            f.write(f"{gene_symbol},{ensembl_id}\n")

if __name__=="__main__":
    main()