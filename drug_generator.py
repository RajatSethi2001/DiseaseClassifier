import gymnasium as gym
import matplotlib.pyplot as plt
import mygene
import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from rdkit import Chem
from rdkit.Chem import QED
from scipy import spatial
from selfies_autoencoder import SelfiesEncoder, SelfiesDecoder
from stable_baselines3 import PPO, TD3, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from train_condition_model import ConditionModelLinear, ConditionModel
from train_health_model import HealthModel, HealthModelLinear
from train_gctx import GenePertModel
from utils import smiles_to_embedding, embedding_to_smiles, get_minmax, get_zscores, get_zscore_minmax
from scipy.stats import pearsonr

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def process_gene_csv(path, desired_genes):
    df = pd.read_csv(path, index_col=0)
    df.index = [index.split(".")[0] for index in df.index]
    df = np.log2(df + 1)
    df = df.apply(get_zscores, axis=0)
    df = df.apply(get_zscore_minmax, axis=0)
    df = df.loc[desired_genes, :]
    df = df.transpose()
    gene_expr = df.to_numpy().flatten()
    return gene_expr

def validate_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol)  # This will raise an exception if the molecule is invalid
        qed_score = QED.qed(mol)
        return qed_score
    except:
        return 0

class DrugGenEnv(gym.Env):
    def __init__(self, gctx_savefile, autoencoder_savefile, health_savefile, condition_dirs, max_selfies_len=50):
        super().__init__()
        with open("Data/selfies_alphabet.txt", "r") as f:
            self.selfies_alphabet = f.read().splitlines()
        self.genes = pd.read_csv("Data/important_genes.csv", header=None)[1].to_list()

        gctx_checkpoint = torch.load(gctx_savefile, weights_only=False)
        self.gctx_model = GenePertModel(len(self.genes), 1200, 1200, dropout_prob=0.5)
        self.gctx_model.load_state_dict(gctx_checkpoint["model_state_dict"])
        self.gctx_model.eval()

        ae_checkpoint = torch.load(autoencoder_savefile)
        dec_hidden_size = 1200
        dec_dropout_prob = 0.0
        dec_layers = 3
        dec_activation = nn.GELU
        self.decoder = SelfiesDecoder(len(self.selfies_alphabet),
                                max_selfies_len=max_selfies_len,
                                embedding_size=dec_hidden_size,
                                hidden_size=dec_hidden_size,
                                dropout_prob=dec_dropout_prob,
                                num_layers=dec_layers,
                                activation_fn=dec_activation)
        self.decoder.load_state_dict(ae_checkpoint["decoder_model"])
        self.decoder.eval()

        health_checkpoint = torch.load(health_savefile)
        # conditions = condition_checkpoint["conditions"]
        self.health_model = HealthModel(len(self.genes))
        self.health_model.load_state_dict(health_checkpoint["model_state_dict"])
        self.health_model.eval()

        self.max_selfies_len = max_selfies_len
        self.reward_list = []

        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Reward")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Reward Over Time")
        self.ax.legend()
        self.fig.show()
        self.fig.canvas.draw()

        self.condition_expr = {}
        for dir in condition_dirs:
            for file in os.listdir(dir):
                filename = f"{dir}/{file}"
                print(f"Processing {filename}")
                self.condition_expr[filename] = process_gene_csv(filename, self.genes)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.genes),), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1202,), dtype=np.float32)
        self.max_reward = 0

    def step(self, action: np.ndarray):
        selfies_embedding = action[:(len(action) - 2)]
        dosage_conc = action[len(action) - 2] * 5
        dosage_time = action[len(action) - 1] * 5

        smiles = embedding_to_smiles(selfies_embedding, self.selfies_alphabet, self.decoder)
        qed_score = validate_molecule(smiles)

        with torch.no_grad():
            current_obs_expr_tensor = torch.tensor(self.current_obs_expr, dtype=torch.float32).unsqueeze(0)
            selfies_embedding_tensor = torch.tensor(selfies_embedding, dtype=torch.float32).unsqueeze(0)
            dosage_conc_tensor = torch.tensor(dosage_conc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            dosage_time_tensor = torch.tensor(dosage_time, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            new_expr = self.gctx_model(current_obs_expr_tensor, selfies_embedding_tensor, dosage_conc_tensor, dosage_time_tensor)

        original_health = self.health_model(current_obs_expr_tensor)[0].item()
        new_health = self.health_model(new_expr)[0].item()

        healthiness = new_health - original_health
        unnorm_dosage_conc = (np.e ** dosage_conc) - 1
        unnorm_dosage_time = (np.e ** dosage_time) - 1
        if healthiness < 0 or qed_score < 0:
            reward = 0
        else:
            reward = healthiness * qed_score
        self.reward_list.append(reward)

        if reward > self.max_reward:
            print(f"File: {self.current_obs_file}")
            print(f"SMILES: {smiles}")
            print(f"Dosage Concentration: {unnorm_dosage_conc} uM")
            print(f"Dosage Time: {unnorm_dosage_time} h")
            print(f"Healthiness Improvement: {healthiness}")
            print(f"Drug QED Score: {qed_score}")
            print(f"Reward: {reward}")
            self.max_reward = reward

        return new_expr.detach().cpu().numpy(), reward, True, False, {}

    def reset(self, seed=None, options=None):
        self.current_obs_file = random.choice(list(self.condition_expr.keys()))
        self.current_obs_expr = self.condition_expr[self.current_obs_file]

        if len(self.reward_list) % 100 == 0 and len(self.reward_list) > 0:
            reward_list_smooth = moving_average(self.reward_list)
            self.line.set_data(range(len(reward_list_smooth)), reward_list_smooth)

            self.ax.relim()
            self.ax.autoscale_view()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            plt.pause(0.01)
        return self.current_obs_expr, {}

def main():    
    gctx_savefile = "Models/gctx.pth"
    ae_savefile = "Models/selfies_autoencoder.pth"
    condition_savefile = "Models/health_model.pth"

    condition_dirs = ["Conditions/Unhealthy"]
    policy_savefile = "Models/drug_generator"

    env = DrugGenEnv(gctx_savefile, ae_savefile, condition_savefile, condition_dirs)
    policy_kwargs = dict(
        net_arch=[1200, 1200],
        activation_fn=torch.nn.ReLU
    )

    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, learning_rate=1e-5, ent_coef=1e-4, vf_coef=0.3, max_grad_norm=0.3, policy_kwargs=policy_kwargs)
    if os.path.exists(f"{policy_savefile}.zip"):
        model.set_parameters(policy_savefile)

    for epoch in range(100):
        model.learn(total_timesteps=5000, progress_bar=True)
        model.save(policy_savefile)

    plt.ioff()
    plt.close()

if __name__=="__main__":
    main()