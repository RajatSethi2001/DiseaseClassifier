import numpy as np
import optuna
import os
import pandas as pd
import random
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
from utils import one_hot_encode, one_hot_decode, set_seeds

class SelfiesEncoder(nn.Module):
    def __init__(self, selfies_alphabet_len, max_selfies_len=50, hidden_size=1500, dropout_prob=0.4, num_layers=2, activation_fn=nn.ReLU):
        super().__init__()
        self.input_size = selfies_alphabet_len * max_selfies_len
        self.dropout_prob = dropout_prob
        self.input_layer = nn.Linear(self.input_size, hidden_size)

        self.fc_layers = []
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.fc_layers.append(nn.BatchNorm1d(hidden_size))
            self.fc_layers.append(activation_fn())
            self.fc_layers.append(nn.Dropout(dropout_prob))
        
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, selfies_one_hot):
        x = selfies_one_hot.reshape(-1, self.input_size)
        x = self.input_layer(x)
        for layer in self.fc_layers:
            x = layer(x)    
        output = self.sigmoid(self.output_layer(x))
        return output
    
class SelfiesDecoder(nn.Module):
    def __init__(self, selfies_alphabet_len, max_selfies_len=50, embedding_size=1500, hidden_size=1500, dropout_prob=0.4, num_layers=2, activation_fn=nn.ReLU):
        super().__init__()
        self.selfies_alphabet_len = selfies_alphabet_len
        self.max_selfies_len = max_selfies_len
        self.output_size = selfies_alphabet_len * max_selfies_len
        
        self.input_layer = nn.Linear(embedding_size, hidden_size)
        self.fc_layers = []
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.fc_layers.append(nn.BatchNorm1d(hidden_size))
            self.fc_layers.append(activation_fn())
            self.fc_layers.append(nn.Dropout(dropout_prob))
        
        self.output_layer = nn.Linear(hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, selfies_embedding):
        x = self.input_layer(selfies_embedding)
        for layer in self.fc_layers:
            x = layer(x)    
        output = self.sigmoid(self.output_layer(x))
        output = output.view(-1, self.max_selfies_len, self.selfies_alphabet_len)
        return output

class SelfiesDataset(Dataset):
    def __init__(self, smiles_list, selfies_alphabet, max_selfies_len=50):
        self.smiles_list = []
        self.selfies_one_hot_list = []
        for smiles in smiles_list:
            try:
                selfies = sf.encoder(smiles)
                selfies_tokens = list(sf.split_selfies(selfies))
                if len(selfies_tokens) > max_selfies_len:
                    continue
                selfies_tokens += ["[SKIP]" for _ in range(max_selfies_len - len(selfies_tokens))]
                selfies_one_hot = one_hot_encode(selfies_tokens, selfies_alphabet)
                self.selfies_one_hot_list.append(torch.tensor(selfies_one_hot, dtype=torch.float32))
                self.smiles_list.append(smiles)
            except:
                pass

    def __len__(self):
        return len(self.selfies_one_hot_list)

    def __getitem__(self, index):
        return self.selfies_one_hot_list[index], self.smiles_list[index]

def train_selfies_ae(trial: optuna.trial.Trial):
    train_test_split = 0.2
    max_selfies_len = 50
    compound_file = "Data/compoundinfo_beta.txt"
    
    batch_size = trial.suggest_int("batch_size", 1, 256)
    enc_hidden_size = trial.suggest_int("enc_hidden_size", 1, 2000)
    enc_dropout_prob = trial.suggest_float("enc_dropout", 0, 1)
    enc_layers = trial.suggest_int("enc_layers", 1, 4)
    enc_activation = trial.suggest_categorical("enc_activation", [nn.Identity, nn.ReLU, nn.GELU, nn.Tanh])
    enc_lr = trial.suggest_float("enc_lr", 1e-6, 1e-2, log=True)
    enc_weight_decay = trial.suggest_float("enc_weight_decay", 1e-8, 1e-2, log=True)

    dec_hidden_size = trial.suggest_int("dec_hidden_size", 1, 2000)
    dec_dropout_prob = trial.suggest_float("dec_dropout", 0, 1)
    dec_layers = trial.suggest_int("dec_layers", 1, 4)
    dec_activation = trial.suggest_categorical("dec_activation", [nn.Identity, nn.ReLU, nn.GELU, nn.Tanh])
    dec_lr = trial.suggest_float("dec_lr", 1e-6, 1e-2, log=True)
    dec_weight_decay = trial.suggest_float("dec_weight_decay", 1e-8, 1e-2, log=True)

    compound_df = pd.read_csv(compound_file, sep="\t")
    smiles_list = compound_df["canonical_smiles"].to_list()
    with open("Data/selfies_alphabet.txt", "r") as f:
        selfies_alphabet = f.read().splitlines()
    
    dataset = SelfiesDataset(smiles_list, selfies_alphabet, max_selfies_len=max_selfies_len)
    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.BCELoss()
    encoder = SelfiesEncoder(len(selfies_alphabet),
                            max_selfies_len=max_selfies_len,
                            hidden_size=enc_hidden_size,
                            dropout_prob=enc_dropout_prob,
                            num_layers=enc_layers,
                            activation_fn=enc_activation)
    
    decoder = SelfiesDecoder(len(selfies_alphabet),
                             max_selfies_len=max_selfies_len,
                             embedding_size=enc_hidden_size,
                             hidden_size=dec_hidden_size,
                             dropout_prob=dec_dropout_prob,
                             num_layers=dec_layers,
                             activation_fn=dec_activation)

    encoder_optim = optim.AdamW(encoder.parameters(), lr=enc_lr, weight_decay=enc_weight_decay)
    decoder_optim = optim.AdamW(decoder.parameters(), lr=dec_lr, weight_decay=dec_weight_decay)

    for epoch in range(10):
        print(f"Epoch {epoch}")
        encoder.train()
        decoder.train()
        batch_idx = 0
        for selfies_one_hot, smiles in train_loader:
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            selfies_encoding = encoder(selfies_one_hot)
            selfies_decoding = decoder(selfies_encoding)
            
            loss = criterion(selfies_decoding, selfies_one_hot)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            encoder_optim.step()
            decoder_optim.step()

            batch_idx += 1
        
        encoder.eval()
        decoder.eval()
        batch_idx = 0
        test_loss = 0.0
        for selfies_one_hot, smiles in test_loader:
            selfies_encoding = encoder(selfies_one_hot)
            selfies_decoding = decoder(selfies_encoding)

            loss = criterion(selfies_decoding, selfies_one_hot)
            test_loss += loss.item()

            selfies_probs_np = selfies_decoding.detach().cpu().numpy()
            batch_idx += 1
        
        correct_tokens = 0
        total_tokens = 0
        for batch_idx in range(batch_size):
            selfies_one_hot_og = selfies_one_hot[batch_idx].cpu().numpy()
            selfies_og_tokens = one_hot_decode(selfies_one_hot_og, selfies_alphabet)
            selfies_og_tokens_clean = [token for token in selfies_og_tokens if token != "[SKIP]"]
            selfies_og = "".join(selfies_og_tokens_clean)
            
            selfies_one_hot_ae = selfies_probs_np[batch_idx]
            selfies_ae_tokens = one_hot_decode(selfies_one_hot_ae, selfies_alphabet)
            selfies_ae_tokens_clean = [token for token in selfies_ae_tokens if token != "[SKIP]"]
            selfies_ae = "".join(selfies_ae_tokens_clean)

            for token_idx in range(len(selfies_og_tokens)):
                if selfies_og_tokens[token_idx] == selfies_ae_tokens[token_idx]:
                    correct_tokens += 1
                total_tokens += 1
            
            try:
                smiles_og = sf.decoder(selfies_og)
                smiles_ae = sf.decoder(selfies_ae)
                print(f"SMILES Before: {smiles_og}")
                print(f"SMILES After:  {smiles_ae}")
                print()
            except Exception as e:
                print(f"Decoding failed for: {selfies_ae}, Error: {e}")

        token_accuracy = correct_tokens / total_tokens
        print(f"SELFIES Accuracy = {token_accuracy}")
    
    return token_accuracy

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(train_selfies_ae, n_trials=50)
    print("Best trial:", study.best_trial.params)

if __name__=="__main__":
    main()