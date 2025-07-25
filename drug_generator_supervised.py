import mygene
import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import QED
from torch.utils.data import Dataset, DataLoader, Subset
from train_gctx import GenePertModel, get_minmax, get_zscores

def process_gene_csv(path, desired_genes, ensembl_to_gene_sym):
    df = pd.read_csv(path, index_col=0)
    df = np.log1p(df)
    df = df.apply(get_zscores)
    df = df.transpose()

    df.columns = ensembl_to_gene_sym

    desired_genes_set = set(desired_genes)
    current_genes_set = set(df.columns)
    missing_genes = [g for g in desired_genes_set if g not in current_genes_set]
    missing_df = pd.DataFrame(
        data=0.0,
        index=df.index,
        columns=missing_genes
    )
    df = pd.concat([df, missing_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[desired_genes]
    gene_expr = df.to_numpy().flatten()
    gene_expr = get_minmax(gene_expr)
    return gene_expr

def selfies_encoding_to_smiles(selfies_encoding, selfies_alphabet):
    selfies_argmax = np.argmax(selfies_encoding, axis=1)
    selfies = ""
    for token_idx in selfies_argmax: 
        token_value = selfies_alphabet[token_idx]
        if token_value == "[STOP]":
            break
        elif token_value == "[SKIP]":
            continue
        else:
            selfies += token_value
    
    smiles = sf.decoder(selfies)
    return smiles

def validate_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol)  # This will raise an exception if the molecule is invalid
        qed_score = QED.qed(mol)
        return qed_score
    except:
        return 0.0

class ConditionDataset(Dataset):
    def __init__(self, condition_dirs, genes, ensembl_to_gene_sym):
        data_list = []
        self.file_list = []
        for dir in condition_dirs:
            for file in os.listdir(dir):
                file_path = f"{dir}/{file}"
                data_list.append(process_gene_csv(file_path, genes, ensembl_to_gene_sym))
                self.file_list.append(file_path)
        
        self.condition_exprs = np.array(data_list)
    
    def __len__(self):
        return len(self.condition_exprs)

    def __getitem__(self, index):
        return torch.tensor(self.condition_exprs[index], dtype=torch.float32), self.file_list[index]

class DrugGenerator(nn.Module):
    def __init__(self, num_genes, selfies_alphabet_len, max_selfies_len=50, hidden_size=2048):
        super().__init__()
        self.input_layer = nn.Linear(num_genes, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, max_selfies_len * selfies_alphabet_len + 2)

        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, expr):
        input_layer = self.dropout(self.activation(self.input_layer(expr)))
        fc1 = self.dropout(self.activation(self.fc1(input_layer)))
        fc2 = self.dropout(self.activation(self.fc2(fc1)))
        fc3 = self.dropout(self.activation(self.fc3(fc2)))
        output_layer = self.sigmoid(self.output_layer(fc3))

        return output_layer

class CustomLoss(nn.Module):
    def __init__(self, gctx_model, healthy_exprs, selfies_alphabet, max_selfies_len=50):
        super().__init__()
        healthy_exprs = torch.tensor(healthy_exprs, dtype=torch.float32)
        self.register_buffer("healthy_exprs", healthy_exprs)
        self.selfies_alphabet = selfies_alphabet
        self.max_selfies_len = max_selfies_len
        self.gctx_model = gctx_model

    def forward(self, condition_exprs, output):
        selfies_len = len(self.selfies_alphabet) * self.max_selfies_len
        selfies_encodings = F.gumbel_softmax(output[:, :selfies_len].reshape(-1, self.max_selfies_len, len(self.selfies_alphabet)), tau=1.0, hard=True, dim=-1)
        dosage_concs = output[:, selfies_len].unsqueeze(1)
        dosage_times = output[:, selfies_len + 1].unsqueeze(1)
        
        new_exprs = self.gctx_model(condition_exprs, selfies_encodings, dosage_concs, dosage_times)
        repeated_new_exprs = new_exprs.repeat_interleave(len(self.healthy_exprs), dim=0)
        repeated_healthy_exprs = self.healthy_exprs.repeat((len(new_exprs), 1))

        cos_sim = F.cosine_similarity(repeated_new_exprs, repeated_healthy_exprs, dim=1)
        cos_loss = 1 - cos_sim.mean()  # Want similarity close to 1

        return cos_loss

def main():
    max_selfies_len = 50
    gctx_filepath = "gctx.pth"
    healthy_dir = "Healthy"
    condition_dirs = ["Breast_Cancer", "Hidradenitis_Supparativa", "Parkinson", "Depression", "Lupus", "COPD", "Acute_Pancreatitis"]
    train_test_split = 0.2

    gctx_checkpoint = torch.load(gctx_filepath, weights_only=False)
    genes = gctx_checkpoint["genes"]
    selfies_alphabet = gctx_checkpoint["selfies_alphabet"]
    gctx_model = GenePertModel(len(genes), len(selfies_alphabet))
    gctx_model.load_state_dict(gctx_checkpoint["model_state_dict"])

    healthy_files = os.listdir(healthy_dir)
    df = pd.read_csv(f"{healthy_dir}/{healthy_files[0]}", index_col=0)
    ensembl_ids = [id.split(".")[0] for id in df.index]

    print("Converting ENSEMBL IDs to Gene Symbols")
    mg = mygene.MyGeneInfo()
    query_result = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
    id_to_symbol = {item['query']: item.get('symbol', "UNKNOWN") for item in query_result}
    ensembl_to_gene_sym = list(id_to_symbol.values())
    print("Finished Converting ENSEMBL IDs to Gene Symbols")

    healthy_exprs = torch.tensor(np.array([process_gene_csv(f"{healthy_dir}/{healthy_file}", genes, ensembl_to_gene_sym) for healthy_file in os.listdir(healthy_dir)]))
    dataset = ConditionDataset(condition_dirs, genes, ensembl_to_gene_sym)
    train_size = int(len(dataset) * (1 - train_test_split))
    test_size = int(len(dataset) * train_test_split)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DrugGenerator(len(genes), len(selfies_alphabet), max_selfies_len)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    criterion = CustomLoss(gctx_model, healthy_exprs, selfies_alphabet)

    for epoch in range(1000):
        print(f"Epoch = {epoch}")
        model.train()
        for condition_exprs, filenames in train_loader:
            optimizer.zero_grad()
            outputs = model(condition_exprs)
            loss = criterion(condition_exprs, outputs)
            loss.backward()
            optimizer.step()
            print(f"Training Batch Loss = {loss.item()}")
        
        torch.save(model, "drug_generator.pth")
        print()
        model.eval()
        for condition_exprs, filenames in test_loader:
            outputs = model(condition_exprs)
            loss = criterion(condition_exprs, outputs)
            print(f"Testing Batch Loss = {loss.item()}")

            for batch_idx in range(batch_size):
                output = outputs[batch_idx].detach()
                selfies_len = len(selfies_alphabet) * max_selfies_len
                selfies_encoding = F.gumbel_softmax(output[:selfies_len].reshape(max_selfies_len, len(selfies_alphabet)), tau=1.0, hard=True, dim=-1)
                dosage_conc = (10 ** output[selfies_len]) - 1
                dosage_time = (10 ** output[selfies_len + 1]) - 1
                smiles = selfies_encoding_to_smiles(selfies_encoding, selfies_alphabet)
                qed_score = validate_molecule(smiles)

                if qed_score > 0.5:
                    condition_expr = condition_exprs[batch_idx]
                    filename = filenames[batch_idx]
                    print(f"File = {filename}")
                    print(f"SMILES = {smiles}")
                    print(f"Dosage Concentration = {dosage_conc}")
                    print(f"Dosage Time = {dosage_time}")
                    print(f"QED Score = {qed_score}")

                    loss = criterion(condition_expr.unsqueeze(0), output.unsqueeze(0))
                    print(f"Cosine Similarity = {1 - loss.item()}")

if __name__=="__main__":
    main()