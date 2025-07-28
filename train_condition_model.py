import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
from utils import set_seeds, get_minmax, get_zscores, get_zscore_minmax

class ConditionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, genes, conditions):
        self.df = df
        self.genes = genes
        self.conditions = conditions
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        gene_data = torch.tensor(self.df.iloc[idx, :][self.genes].values, dtype=torch.float32)
        condition_data = torch.tensor(self.df.iloc[idx, :][self.conditions].values, dtype=torch.float32)
        return gene_data, condition_data

class ConditionModel(nn.Module):
    def __init__(self, num_genes, num_conditions, hidden_neurons=1024):
        super().__init__()
        self.input_layer = nn.Linear(num_genes, hidden_neurons)
        self.fc1 = nn.Linear(hidden_neurons, hidden_neurons)
        self.bn1 = nn.BatchNorm1d(hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.bn2 = nn.BatchNorm1d(hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, num_conditions)
    
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.dropout(self.activation(self.input_layer(x)))
        x = self.dropout(self.activation(self.bn1(self.fc1(x))))
        x = self.dropout(self.activation(self.bn2(self.fc2(x))))
        x = self.sigmoid(self.output_layer(x))
        return x

class ConditionModelLinear(nn.Module):
    def __init__(self, num_genes, num_conditions):
        super().__init__()
        self.dropout = nn.Dropout(0.4)
        self.network = nn.Linear(num_genes, num_conditions, bias=False)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.network(x))
        return x

    def get_network_weights(self):
        return self.network.weight.data

def main():
    set_seeds(11)
    condition_parent_dir = "Conditions"
    directory_list = ["Lupus", "Podoconiosis", "Primary_Sclerosing_Cholangitis", "Ulcerative_Colitis", "Shingles", "Sepsis", "Scleroderma", "MRSA_Bacteremia", "Crohns_Disease", "Acute_Pancreatitis", "Aneurysm", "Tuberculosis", "Acute_Myeloid_Leukemia", "Endocarditis", "Schistosomiasis", "Leprosy", "Amyotrophic_Lateral_Sclerosis", "Chronic_Myeloid_Leukemia", "Dengue", "Alzheimer", "Restless_Legs_Syndrome", "Coronary_Artery_Disease", "COPD", "Breast_Cancer", "Crimean_Congo_Hemorrhagic_Fever", "Hypertension,Drug_Abuse", "Hypertension", "COVID19", "Depression", "PTSD", "HIV", "HIV,Tuberculosis", "Malaria", "Hidradenitis_Supparativa", "SFTS", "Cystic_Fibrosis", "Chikungunya", "Rheumatoid_Arthritis", "Polycystic_Kidney_Disease", "Parkinson", "Myelofibrosis"]
    savefile = "Models/condition_model.pth"
    train_test_split = 0.2
    batch_size = 128
    genes = pd.read_csv("Data/important_genes.csv", header=None)[1].to_list()

    condition_df = pd.DataFrame()
    conditions = set()
    for directory in directory_list:
        path = f"{condition_parent_dir}/{directory}"
        print(f"Processing Directory: {path}")
        filenames = os.listdir(path)
        random.shuffle(filenames)

        for filename in filenames:
            csv_path = f"{path}/{filename}"
            df = pd.read_csv(csv_path, index_col=0)
            df.index = [index.split(".")[0] for index in df.index]
            df = np.log2(df + 1)
            df = df.apply(get_zscores, axis=0)
            df = df.apply(get_zscore_minmax, axis=0)
            df = df.loc[genes, :]
            df = df.transpose()
            dir_conditions = directory.split(",")
            for condition in dir_conditions:
                df[condition] = 1
                conditions.add(condition)
            condition_df = pd.concat([condition_df, df])

    condition_df = condition_df.fillna(0)
    conditions = sorted(list(conditions))

    print(condition_df)

    dataset = ConditionDataset(condition_df, genes, conditions)
    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ConditionModel(len(genes), len(conditions), hidden_neurons=32)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if os.path.exists(savefile):
        checkpoint = torch.load(savefile, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCELoss()
    for epoch in range(500):
        print(f"Epoch: {epoch}")
        train_metrics = {condition: {"tp": np.float32(0), "tn": np.float32(0), "fp": np.float32(0), "fn": np.float32(0)} for condition in conditions}
        train_loss = 0
        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            for batch_idx in range(len(outputs)):
                output_batch = outputs[batch_idx]
                label_batch = labels[batch_idx]
                for condition_idx in range(len(conditions)):
                    condition = conditions[condition_idx]
                    output_choice = round(output_batch[condition_idx].item())
                    label_choice = round(label_batch[condition_idx].item())

                    if output_choice == 1 and label_choice == 1:
                        train_metrics[condition]["tp"] += 1
                    
                    elif output_choice == 0 and label_choice == 0:
                        train_metrics[condition]["tn"] += 1
                    
                    elif output_choice == 1 and label_choice == 0:
                        train_metrics[condition]["fp"] += 1
                    
                    elif output_choice == 0 and label_choice == 1:
                        train_metrics[condition]["fn"] += 1
        
        for condition in conditions:
            tp = train_metrics[condition]["tp"]
            tn = train_metrics[condition]["tn"]
            fp = train_metrics[condition]["fp"]
            fn = train_metrics[condition]["fn"]
            with np.errstate(invalid='ignore', divide='ignore'):
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f_score = 2 * precision * recall / (precision + recall)
                print(f"{condition} Train Metrics: Precision = {round(precision, 3)}, Recall = {round(recall, 3)}, F-Score = {round(f_score, 3)}")
        
        train_loss /= len(train_dataset)
        print(f"Train Loss = {train_loss}")    
        print("Saving Model")
        print()
        torch.save({
            "model_state_dict": model.state_dict(),
            "genes": genes,
            "conditions": conditions
        }, savefile)

        model.eval()
        test_metrics = {condition: {"tp": np.float32(0), "tn": np.float32(0), "fp": np.float32(0), "fn": np.float32(0)} for condition in conditions}
        test_loss = 0
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            for batch_idx in range(len(outputs)):
                output_batch = outputs[batch_idx]
                label_batch = labels[batch_idx] 
                for condition_idx in range(len(conditions)):
                    condition = conditions[condition_idx]
                    output_choice = round(output_batch[condition_idx].item())
                    label_choice = round(label_batch[condition_idx].item())

                    if output_choice == 1 and label_choice == 1:
                        test_metrics[condition]["tp"] += 1
                    
                    elif output_choice == 0 and label_choice == 0:
                        test_metrics[condition]["tn"] += 1
                    
                    elif output_choice == 1 and label_choice == 0:
                        test_metrics[condition]["fp"] += 1
                    
                    elif output_choice == 0 and label_choice == 1:
                        test_metrics[condition]["fn"] += 1
        
        for condition in conditions:
            tp = test_metrics[condition]["tp"]
            tn = test_metrics[condition]["tn"]
            fp = test_metrics[condition]["fp"]
            fn = test_metrics[condition]["fn"]
            with np.errstate(invalid='ignore', divide='ignore'):
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f_score = 2 * precision * recall / (precision + recall)
                print(f"{condition} Test Metrics: Precision = {round(precision, 3)}, Recall = {round(recall, 3)}, F-Score = {round(f_score, 3)}")
        
        test_loss /= len(test_dataset)
        print(f"Test Loss = {test_loss}")
        print()

if __name__=="__main__":
    main()