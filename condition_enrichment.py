import gseapy as gp
import mygene
import pandas as pd
import torch
from train_condition_model import ConditionModelLinear

condition_of_interest = "Breast_Cancer"
savefile = "linear_model_no_bias.pth"

checkpoint = torch.load(savefile, weights_only=True)
genes = checkpoint["genes"]
conditions = checkpoint["conditions"]
model = ConditionModelLinear(len(genes), len(conditions))
model = torch.compile(model)
model.load_state_dict(checkpoint["model_state_dict"])

weights = model.get_network_weights()
condition_index = conditions.index(condition_of_interest)
condition_weights = weights[condition_index].tolist()
condition_weight_pairs = list(zip(genes, condition_weights))
condition_weight_pairs.sort(key = lambda x: abs(x[1]), reverse=True)

ensembl_ids = [x[0].split(".")[0] for x in condition_weight_pairs]
mg = mygene.MyGeneInfo()
query_result = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
id_to_symbol = {item['query']: item.get('symbol', None) for item in query_result}
gene_symbols = list(filter(None, id_to_symbol.values()))
expr_values = [x[1] for x in condition_weight_pairs]

ranked_genes = dict(zip(gene_symbols, expr_values))
rnk = pd.Series(ranked_genes).sort_values(ascending=False)

gsea_res = gp.prerank(rnk=rnk,
                     gene_sets=['KEGG_2021_Human', 'Reactome_2022'],
                     outdir=f'{condition_of_interest}_Enrichment',
                     permutation_num=100)

print(gsea_res.res2d.head())
