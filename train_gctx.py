import h5py
import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import torch
import torch.nn as nn
import torch.optim as optim
from selfies_autoencoder import SelfiesEncoder, SelfiesDecoder
from torch.utils.data import Dataset, DataLoader, Subset
from utils import get_minmax, get_zscores, clean_dose_unit, smiles_to_embedding, set_seeds
"""
/0
/0/DATA
/0/DATA/0
/0/DATA/0/matrix
/0/META
/0/META/COL
/0/META/COL/cell_id
/0/META/COL/distil_id
/0/META/COL/id
/0/META/COL/pert_dose
/0/META/COL/pert_dose_unit
/0/META/COL/pert_id
/0/META/COL/pert_idose
/0/META/COL/pert_iname
/0/META/COL/pert_itime
/0/META/COL/pert_time
/0/META/COL/pert_time_unit
/0/META/COL/pert_type
/0/META/ROW
/0/META/ROW/id
/0/META/ROW/pr_gene_symbol
/0/META/ROW/pr_gene_title
/0/META/ROW/pr_is_bing
/0/META/ROW/pr_is_lm
"""

class GenePertDataset(Dataset):
    def __init__(self, gctx_file, compound_file, data_limit=100000, max_selfies_len=50):
        self.gctx_fp = h5py.File(gctx_file, "r")
        self.distil_ids = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/distil_id"][:data_limit]]
        self.pert_dose = [float(s.decode('utf-8').split("|")[0]) for s in self.gctx_fp["0/META/COL/pert_dose"][:data_limit]]
        self.pert_dose_units = [clean_dose_unit(s) for s in self.gctx_fp["0/META/COL/pert_dose_unit"][:data_limit]]
        self.pert_time = [float(s) for s in self.gctx_fp["0/META/COL/pert_time"][:data_limit]]
        self.pert_time_units = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_time_unit"][:data_limit]]
        self.pert_types = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_type"][:data_limit]]
        self.pert_id = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_id"][:data_limit]]
        self.gene_symbols = [s.decode("utf-8") for s in self.gctx_fp["/0/META/ROW/pr_gene_symbol"]]

        compound_df = pd.read_csv(compound_file, sep="\t")
        self.smiles_lookup = compound_df.set_index("pert_id")["canonical_smiles"].to_dict()
        self.max_selfies_len = max_selfies_len

        self.important_genes = pd.read_csv("Data/important_genes.csv", header=None)[0].to_list()
        self.gene_idx = np.array([self.gene_symbols.index(gene) for gene in self.important_genes])
        with open("Data/selfies_alphabet.txt", "r") as f:
            self.selfies_alphabet = f.read().splitlines()

        data_map = {}
        for idx in range(data_limit):
            distil_id = self.distil_ids[idx].split(":")[0]
            if distil_id not in data_map:
                data_map[distil_id] = {"ctl_idx": [], "trt_idx": []}
            
            pert_type = self.pert_types[idx]
            if pert_type == "ctl_untrt" or pert_type == "ctl_vehicle":
                data_map[distil_id]["ctl_idx"].append(idx)
            elif pert_type == "trt_cp":
                pert_dose_unit = self.pert_dose_units[idx]
                pert_time_unit = self.pert_time_units[idx]
                pert_id = self.pert_id[idx]
                smiles = self.smiles_lookup[pert_id]
                try:
                    selfies = sf.encoder(smiles)
                    selfies_tokens = list(sf.split_selfies(selfies))
                except:
                    continue

                if pert_dose_unit == "uM" and pert_time_unit == "h" and len(selfies_tokens) <= max_selfies_len:
                    data_map[distil_id]["trt_idx"].append(idx)
        
        self.encoder = SelfiesEncoder(len(self.selfies_alphabet), hidden_size=1500)
        self.decoder = SelfiesDecoder(len(self.selfies_alphabet), hidden_size=1500)
        ae_checkpoint = torch.load("Models/selfies_autoencoder.pth")
        self.encoder.load_state_dict(ae_checkpoint["encoder_model"])
        self.decoder.load_state_dict(ae_checkpoint["decoder_model"])
        self.encoder.eval()
        self.decoder.eval()

        self.gctx_data = []
        for distil_id, gene_data in data_map.items():
            if len(self.gctx_data) > 10000:
                break
            for ctl_idx in gene_data["ctl_idx"]:
                ctl_expr_total = np.array(self.gctx_fp["0/DATA/0/matrix"][ctl_idx, :])[self.gene_idx]
                ctl_expr = torch.tensor(get_minmax(get_zscores(ctl_expr_total)), dtype=torch.float32)
                for trt_idx in gene_data["trt_idx"]:
                    trt_expr_total = np.array(self.gctx_fp["0/DATA/0/matrix"][trt_idx, :])[self.gene_idx]
                    trt_expr = torch.tensor(get_minmax(get_zscores(trt_expr_total)), dtype=torch.float32)
                    dose = torch.tensor([np.log1p(self.pert_dose[trt_idx])], dtype=torch.float32)
                    time = torch.tensor([np.log1p(self.pert_time[trt_idx])], dtype=torch.float32)
                    smiles = self.smiles_lookup[self.pert_id[trt_idx]]
                    smiles_embedding = torch.tensor(smiles_to_embedding(smiles, self.selfies_alphabet, self.encoder), dtype=torch.float32)

                    self.gctx_data.append((ctl_expr, trt_expr, smiles_embedding, dose, time))
                print(f"Genes Processed = {len(self.gctx_data)}")

        del self.distil_ids
        del self.pert_dose
        del self.pert_dose_units
        del self.pert_time
        del self.pert_time_units
        del self.pert_types
        del self.pert_id
        self.gctx_fp.close()

    def __len__(self):
        return len(self.gctx_data)

    def __getitem__(self, idx):
        return self.gctx_data[idx]

    def get_selfies_alphabet(self):
        return self.selfies_alphabet

    def get_gene_symbols(self):
        return self.important_genes

class GenePertModel(nn.Module):
    def __init__(self, num_genes, embedding_len, hidden_size=2000):
        super().__init__()
        self.fc_input_len = num_genes + embedding_len + 2
        self.fc1 = nn.Linear(self.fc_input_len, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, num_genes)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, ctl_expr, smiles_embedding, dose, time):
        fc_input = torch.cat((ctl_expr, smiles_embedding, dose, time), dim=-1)
        fc1 = self.dropout(self.activation(self.bn1(self.fc1(fc_input))))
        fc2 = self.dropout(self.activation(self.bn2(self.fc2(fc1))))
        fc3 = self.dropout(self.activation(self.bn3(self.fc3(fc2))))
        output = self.output(fc3)
        return output

def main():
    set_seeds()

    train_test_split = 0.1
    dataset = GenePertDataset("Data/annotated_GSE92742_Broad_LINCS_Level5_COMPZ_n473647x12328.gctx", "Data/compoundinfo_beta.txt")
    model_savefile = "Models/gctx.pth"

    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.SmoothL1Loss()
    model = GenePertModel(len(dataset.get_gene_symbols()), 1500)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    if os.path.exists(model_savefile):
        checkpoint = torch.load(model_savefile, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    epochs = 100

    for epoch in range(epochs):
        print(f"Training Epoch {epoch}")
        model.train()
        train_loss = 0
        batch = 0
        for ctl_expr, trt_expr, smiles_embedding, dose, time in train_loader:
            optimizer.zero_grad()
            output = model(ctl_expr, smiles_embedding, dose, time)
            loss = criterion(output, trt_expr)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch += 1
            
            print(f"Train Batch {batch}: Loss = {loss.item()}")
        
        train_loss /= batch
        print(f"Training Loss = {train_loss}")
        print(f"Saving Model Data: {model_savefile}")
        torch.save({
            "model_state_dict": model.state_dict()
        }, model_savefile)

        print(f"Testing Epoch {epoch}")
        model.eval()
        test_loss = 0
        batch = 0
        for ctl_expr, trt_expr, smiles_embedding, dose, time in test_loader:
            output = model(ctl_expr, smiles_embedding, dose, time)
            input(ctl_expr)
            input(trt_expr)
            input(output)
            loss = criterion(output, trt_expr)
            test_loss += loss.item()

            batch += 1

            print(f"Test Batch {batch}: Loss = {loss.item()}")

        test_loss /= batch   
        print(f"Testing Batch Loss = {test_loss}")

if __name__=="__main__":
    main()