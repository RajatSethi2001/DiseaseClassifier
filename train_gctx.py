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
from utils import get_zscore_minmax, get_zscores, clean_dose_unit, smiles_to_embedding, set_seeds
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
def diagnose_training_issues(model, train_loader, test_loader, criterion, device='cpu'):
    """Diagnose what's happening during training"""
    
    print("=== TRAINING DIAGNOSIS ===")
    
    model.eval()
    
    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    all_controls = []
    all_diffs = []  # |target - control|
    
    with torch.no_grad():
        for batch_idx, (ctl_expr, trt_expr, smiles_embedding, dose, time) in enumerate(train_loader):
            if batch_idx >= 5:  # Only analyze first 5 batches
                break
                
            ctl_expr = ctl_expr.to(device)
            trt_expr = trt_expr.to(device)
            smiles_embedding = smiles_embedding.to(device)
            dose = dose.to(device)
            time = time.to(device)
            
            predictions = model(ctl_expr, smiles_embedding, dose, time)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(trt_expr.cpu())
            all_controls.append(ctl_expr.cpu())
            
            # Calculate control-treatment differences
            diff = torch.abs(trt_expr - ctl_expr).cpu()
            all_diffs.append(diff)
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    controls = torch.cat(all_controls, dim=0)
    diffs = torch.cat(all_diffs, dim=0)
    
    print(f"Analyzed {predictions.shape[0]} samples")
    
    # 1. Check if model is just predicting control values
    control_mse = torch.mean((predictions - controls) ** 2).item()
    target_mse = torch.mean((predictions - targets) ** 2).item()
    
    print(f"\n--- Prediction Analysis ---")
    print(f"MSE(prediction, control): {control_mse:.6f}")
    print(f"MSE(prediction, target): {target_mse:.6f}")
    
    if control_mse < 0.001:
        print("WARNING: Model is mostly predicting control values!")
        print("This suggests the model isn't learning compound effects.")
    
    # 2. Check prediction diversity
    pred_std = torch.std(predictions, dim=0).mean().item()
    target_std = torch.std(targets, dim=0).mean().item()
    control_std = torch.std(controls, dim=0).mean().item()
    
    print(f"\n--- Diversity Analysis ---")
    print(f"Prediction std (avg across genes): {pred_std:.6f}")
    print(f"Target std (avg across genes): {target_std:.6f}")
    print(f"Control std (avg across genes): {control_std:.6f}")
    
    if pred_std < 0.01:
        print("WARNING: Predictions have very low diversity!")
        print("Model might be predicting nearly constant values.")
    
    # 3. Check actual treatment effects in data
    avg_effect_size = torch.mean(diffs).item()
    max_effect_size = torch.max(diffs).item()
    
    print(f"\n--- Treatment Effect Analysis ---")
    print(f"Average |treatment - control|: {avg_effect_size:.6f}")
    print(f"Maximum |treatment - control|: {max_effect_size:.6f}")
    
    if avg_effect_size < 0.02:
        print("WARNING: Treatment effects are very small!")
        print("This might indicate weak compound effects or data issues.")
    
    # 4. Check for single control issue
    unique_controls = torch.unique(controls, dim=0).shape[0]
    total_samples = controls.shape[0]
    
    print(f"\n--- Control Diversity ---")
    print(f"Unique control profiles: {unique_controls}")
    print(f"Total samples: {total_samples}")
    print(f"Control reuse ratio: {total_samples / unique_controls:.1f}")
    
    if unique_controls < 20:
        print("WARNING: Very few unique control profiles!")
        print("This limits the model's ability to learn.")
    
    # 5. Sample some predictions vs targets
    print(f"\n--- Sample Predictions vs Targets (first 3 samples, first 5 genes) ---")
    for i in range(min(3, predictions.shape[0])):
        print(f"Sample {i}:")
        print(f"  Control:    {controls[i, :5].numpy()}")
        print(f"  Target:     {targets[i, :5].numpy()}")
        print(f"  Prediction: {predictions[i, :5].numpy()}")
        print(f"  |Tgt-Ctl|:  {diffs[i, :5].numpy()}")
        print()

def check_gradient_flow(model, train_loader, criterion, device='cpu'):
    """Check if gradients are flowing properly"""
    
    print("=== GRADIENT FLOW CHECK ===")
    
    model.train()
    
    # Take one batch
    ctl_expr, trt_expr, smiles_embedding, dose, time = next(iter(train_loader))
    ctl_expr = ctl_expr.to(device)
    trt_expr = trt_expr.to(device)
    smiles_embedding = smiles_embedding.to(device)
    dose = dose.to(device)
    time = time.to(device)
    
    # Forward pass
    predictions = model(ctl_expr, smiles_embedding, dose, time)
    loss = criterion(predictions, trt_expr)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    print("Gradient norms by layer:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
            
            if grad_norm < 1e-8:
                zero_grad_count += 1
                
            print(f"  {name}: {grad_norm:.8f}")
        else:
            print(f"  {name}: NO GRADIENT")
    
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"\nGradient Summary:")
    print(f"  Total gradient norm: {total_grad_norm:.8f}")
    print(f"  Parameters with gradients: {param_count}")
    print(f"  Parameters with near-zero gradients: {zero_grad_count}")
    
    if total_grad_norm < 1e-6:
        print("WARNING: Very small gradients detected!")
        print("This suggests vanishing gradients or that the model isn't learning.")
    
    if zero_grad_count > param_count * 0.8:
        print("WARNING: Most parameters have near-zero gradients!")

def simple_baseline_test(train_loader, test_loader, device='cpu'):
    """Test if a simple baseline can achieve similar performance"""
    
    print("=== SIMPLE BASELINE TEST ===")
    
    # Baseline 1: Always predict control
    baseline1_train_loss = 0
    baseline1_test_loss = 0
    train_samples = 0
    test_samples = 0
    
    criterion = nn.MSELoss()
    
    # Test on training data
    for ctl_expr, trt_expr, _, _, _ in train_loader:
        loss = criterion(ctl_expr.to(device), trt_expr.to(device))
        baseline1_train_loss += loss.item()
        train_samples += 1
    
    # Test on test data  
    for ctl_expr, trt_expr, _, _, _ in test_loader:
        loss = criterion(ctl_expr.to(device), trt_expr.to(device))
        baseline1_test_loss += loss.item()
        test_samples += 1
    
    baseline1_train_loss /= train_samples
    baseline1_test_loss /= test_samples
    
    print(f"Baseline 1 (predict control):")
    print(f"  Train loss: {baseline1_train_loss:.6f}")
    print(f"  Test loss: {baseline1_test_loss:.6f}")
    
    # If your model's loss is similar to this baseline, it's not learning
    print(f"\nIf your model's loss is close to {baseline1_train_loss:.6f},")
    print(f"then it's essentially just predicting the control values.")

class GenePertDataset(Dataset):
    def __init__(self, gctx_file, compound_file, data_limit=30000, max_selfies_len=50):
        self.gctx_fp = h5py.File(gctx_file, "r")
        
        data_idx = random.sample(range(0, len(self.gctx_fp["0/META/COL/id"])), data_limit)
        data_idx.sort()
        data_idx = np.array(data_idx)

        self.ids = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/id"][data_idx]]
        self.pert_dose = [float(s.decode('utf-8').split("|")[0]) for s in self.gctx_fp["0/META/COL/pert_dose"][data_idx]]
        self.pert_dose_units = [clean_dose_unit(s) for s in self.gctx_fp["0/META/COL/pert_dose_unit"][data_idx]]
        self.pert_time = [float(s) for s in self.gctx_fp["0/META/COL/pert_time"][data_idx]]
        self.pert_time_units = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_time_unit"][data_idx]]
        self.pert_types = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_type"][data_idx]]
        self.pert_id = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_id"][data_idx]]
        self.gene_symbols = [s.decode("utf-8") for s in self.gctx_fp["/0/META/ROW/pr_gene_symbol"]]

        print("Data Collected")

        compound_df = pd.read_csv(compound_file, sep="\t")
        self.smiles_lookup = compound_df.set_index("pert_id")["canonical_smiles"].to_dict()
        self.max_selfies_len = max_selfies_len

        self.important_genes = pd.read_csv("Data/important_genes.csv", header=None)[0].to_list()
        self.gene_idx = np.array([self.gene_symbols.index(gene) for gene in self.important_genes])
        with open("Data/selfies_alphabet.txt", "r") as f:
            self.selfies_alphabet = f.read().splitlines()

        data_map = {}
        for idx in range(data_limit):
            id_parts = self.ids[idx].split(":")
        
            if len(id_parts) < 2:
                continue

            condition = id_parts[0]
            pert_type = self.pert_types[idx]
            
            if condition not in data_map:
                data_map[condition] = {"ctl_idx": [], "trt_idx": []}
            
            if pert_type == "ctl_untrt" or pert_type == "ctl_vehicle":
                data_map[condition]["ctl_idx"].append(idx)

            elif pert_type == "trt_cp":
                pert_dose_unit = self.pert_dose_units[idx]
                pert_time_unit = self.pert_time_units[idx]
                pert_id = self.pert_id[idx]
                if pert_id not in self.smiles_lookup:
                    continue
                smiles = self.smiles_lookup[pert_id]
                try:
                    selfies = sf.encoder(smiles)
                    selfies_tokens = list(sf.split_selfies(selfies))
                except:
                    continue

                if pert_dose_unit == "uM" and pert_time_unit == "h" and len(selfies_tokens) <= self.max_selfies_len:
                    data_map[condition]["trt_idx"].append(idx)
        
        valid_conditions = {}
        for condition, data in data_map.items():
            if len(data["ctl_idx"]) > 0 and len(data["trt_idx"]) > 0:
                valid_conditions[condition] = data
                print(f"Condition {condition}: {len(data['ctl_idx'])} controls, {len(data['trt_idx'])} treatments")
            else:
                print(f"Condition {condition} has no valid mappings")
        
        enc_hidden_size = 1200
        enc_dropout_prob = 0.0
        enc_layers = 3
        enc_activation = nn.GELU

        dec_hidden_size = 1200
        dec_dropout_prob = 0.0
        dec_layers = 3
        dec_activation = nn.GELU

        self.encoder = SelfiesEncoder(len(self.selfies_alphabet),
                            max_selfies_len=max_selfies_len,
                            hidden_size=enc_hidden_size,
                            dropout_prob=enc_dropout_prob,
                            num_layers=enc_layers,
                            activation_fn=enc_activation)
    
        self.decoder = SelfiesDecoder(len(self.selfies_alphabet),
                                max_selfies_len=max_selfies_len,
                                embedding_size=dec_hidden_size,
                                hidden_size=dec_hidden_size,
                                dropout_prob=dec_dropout_prob,
                                num_layers=dec_layers,
                                activation_fn=dec_activation)
        ae_checkpoint = torch.load("Models/selfies_autoencoder.pth")
        self.encoder.load_state_dict(ae_checkpoint["encoder_model"])
        self.decoder.load_state_dict(ae_checkpoint["decoder_model"])
        self.encoder.eval()
        self.decoder.eval()

        self.gctx_data = []
        data_map_items = list(valid_conditions.items())
        random.shuffle(data_map_items)
        for condition_id, gene_data in data_map_items:
            if len(self.gctx_data) > 3000:
                break

            ctl_exprs = []
            for ctl_idx in gene_data["ctl_idx"]:
                ctl_expr_total = np.array(self.gctx_fp["0/DATA/0/matrix"][ctl_idx, :])
                ctl_expr = get_zscore_minmax(ctl_expr_total)[self.gene_idx]
                ctl_exprs.append(ctl_expr)
            
            if len(ctl_exprs) == 0:
                continue
                
            ctl_exprs = np.array(ctl_exprs)
            ctl_expr_median = np.median(ctl_exprs, axis=0)
            
            if np.isnan(ctl_expr_median).any():
                continue
                
            ctl_expr_median = torch.tensor(ctl_expr_median, dtype=torch.float32)

            for trt_idx in gene_data["trt_idx"]:
                try:
                    trt_expr_total = np.array(self.gctx_fp["0/DATA/0/matrix"][trt_idx, :])
                    trt_expr = torch.tensor(get_zscore_minmax(trt_expr_total)[self.gene_idx], dtype=torch.float32)
                    dose = torch.tensor([np.log1p(self.pert_dose[trt_idx])], dtype=torch.float32)
                    time = torch.tensor([np.log1p(self.pert_time[trt_idx])], dtype=torch.float32)
                    smiles = self.smiles_lookup[self.pert_id[trt_idx]]
                    smiles_embedding = torch.tensor(smiles_to_embedding(smiles, self.selfies_alphabet, self.encoder), dtype=torch.float32)

                    self.gctx_data.append((ctl_expr_median, trt_expr, smiles_embedding, dose, time))
                except Exception as e:
                    print(f"Error processing treatment {trt_idx}: {e}")
                    continue
            
            print(f"Conditions Processed: {len(self.gctx_data)}")
        
        del self.ids
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
    def __init__(self, num_genes, embedding_len, hidden_size=2000, dropout_prob=0.4, activation_fn=nn.GELU):
        super().__init__()
        self.input_size = num_genes + embedding_len + 2
        self.input_layer = nn.Linear(self.input_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.LayerNorm(hidden_size)
        
        self.output_layer = nn.Linear(hidden_size, num_genes)

        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, ctl_expr, smiles_embedding, dose, time):
        x = torch.cat((ctl_expr, smiles_embedding, dose, time), dim=-1)
        x = self.input_layer(x)
        
        residual = x
        x = self.dropout(self.activation(self.bn1(self.fc1(x))))
        x = x + residual

        residual = x
        x = self.dropout(self.activation(self.bn2(self.fc2(x))))
        x = x + residual

        residual = x
        x = self.dropout(self.activation(self.bn3(self.fc3(x))))
        x = x + residual

        output = self.sigmoid(self.output_layer(x))
        return output

def main():
    set_seeds(1111)

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

    criterion = nn.MSELoss()
    model = GenePertModel(len(dataset.get_gene_symbols()), 1200, 1200, dropout_prob=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2, betas=(0.9, 0.999))

    if os.path.exists(model_savefile):
        checkpoint = torch.load(model_savefile, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    epochs = 400

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
            
            # print(f"Train Batch {batch}: Loss = {loss.item()}")
        
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
            loss = criterion(output, trt_expr)
            test_loss += loss.item()

            batch += 1

            # print(f"Test Batch {batch}: Loss = {loss.item()}")

        test_loss /= batch   
        print(f"Testing Loss = {test_loss}")

        # diagnose_training_issues(model, train_loader, test_loader, criterion)
        # check_gradient_flow(model, train_loader, criterion)
        # simple_baseline_test(train_loader, test_loader)


if __name__=="__main__":
    main()