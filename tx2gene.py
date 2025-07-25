import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path of quant.sf file')
parser.add_argument('-o', '--outfile', help="Path of output csv file")
args = parser.parse_args()
path = args.path
outfile = args.outfile

tx2gene = pd.read_csv("tx2gene.csv", header=None, names=["transcript", "gene"])
tx2gene = tx2gene.set_index("transcript")

df = pd.read_csv(path, sep="\t", index_col=0)
df.index = df.index.map(lambda x: x.split("|")[0])
df = df.join(tx2gene, how="inner")

gene_tpm = df.groupby("gene")["TPM"].sum()
gene_tpm.to_csv(outfile)