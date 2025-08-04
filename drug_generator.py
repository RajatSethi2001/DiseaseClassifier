import gymnasium as gym
import math
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
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
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

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def validate_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol)  # This will raise an exception if the molecule is invalid
        return True
    except:
        return False

def get_qed_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    qed_score = QED.qed(mol)
    return qed_score

def get_sa_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sa_score = sascorer.calculateScore(mol)
    return sa_score

def rapid_toxicity_screen(smiles):
    """
    Ultra-fast toxicity screening using simple rules
    Returns: (risk_level, risk_score, warnings)
    """
    mol = Chem.MolFromSmiles(smiles)
    risk_score = 0.0
    
    # Quick property checks
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Lipinski violations (not directly toxic but correlated)
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    if lipinski_violations >= 2:
        # "Multiple Lipinski violations"
        risk_score += 0.3
    
    # Extreme properties
    if logp > 6:
        # "Very high lipophilicity"
        risk_score += 0.4
    
    if mw > 800:
        # "Very high molecular weight"
        risk_score += 0.3
    
    if tpsa > 200:
        # "High polar surface area"
        risk_score += 0.2
    
    # Structural alerts (simplified)
    smiles_lower = smiles.lower()
    
    # Check for concerning patterns in SMILES string
    toxic_patterns = {
        # Highly reactive/electrophilic groups
        'aldehyde': {
            'patterns': ['c=o', 'cc=o', '[ch]=o'],
            'weight': 0.2,
            'description': 'Aldehyde group (reactive)'
        },
        'ketone_reactive': {
            'patterns': ['c(=o)c', 'cc(=o)c'],
            'weight': 0.1,
            'description': 'Ketone group'
        },
        'acyl_halide': {
            'patterns': ['c(=o)f', 'c(=o)cl', 'c(=o)br'],
            'weight': 0.5,
            'description': 'Acyl halide (highly reactive)'
        },
        'isocyanate': {
            'patterns': ['n=c=o', 'nc=o'],
            'weight': 0.4,
            'description': 'Isocyanate group (respiratory toxin)'
        },
        'epoxide': {
            'patterns': ['c1oc1', 'c1co1'],
            'weight': 0.3,
            'description': 'Epoxide (alkylating agent)'
        },
        
        # Nitro compounds and N-oxides
        'nitro_aromatic': {
            'patterns': ['c[n+]([o-])=o', 'c[n+](=o)[o-]'],
            'weight': 0.4,
            'description': 'Aromatic nitro compound'
        },
        'nitro_aliphatic': {
            'patterns': ['cc[n+]([o-])=o', 'c[n+]([o-])=o'],
            'weight': 0.3,
            'description': 'Aliphatic nitro compound'
        },
        'n_oxide': {
            'patterns': ['[n+]([o-])', 'n->o'],
            'weight': 0.2,
            'description': 'N-oxide group'
        },
        
        # Azo and diazo compounds
        'azo_aromatic': {
            'patterns': ['cn=nc', 'c-n=n-c'],
            'weight': 0.4,
            'description': 'Aromatic azo compound (potential carcinogen)'
        },
        'diazonium': {
            'patterns': ['c[n+]#n', 'cn+#n'],
            'weight': 0.5,
            'description': 'Diazonium salt (explosive)'
        },
        'azide': {
            'patterns': ['n=[n+]=[n-]', 'nn#n'],
            'weight': 0.4,
            'description': 'Azide group (explosive)'
        },
        
        # Aromatic amines (carcinogenic potential)
        'aniline': {
            'patterns': ['cn', 'c-n', 'cnc'],
            'weight': 0.3,
            'description': 'Aromatic amine (potential carcinogen)'
        },
        'benzidine_like': {
            'patterns': ['ncc1ccc(cc1)n', 'nc1ccc(cc1)n'],
            'weight': 0.5,
            'description': 'Benzidine-like structure (carcinogen)'
        },
        
        # Quinones and Michael acceptors
        'quinone': {
            'patterns': ['c1=cc(=o)c=cc1=o', 'o=c1c=cc(=o)c=c1'],
            'weight': 0.4,
            'description': 'Quinone (redox active, toxic)'
        },
        'michael_acceptor': {
            'patterns': ['c=cc=o', 'c=cc(=o)', 'c=ccn'],
            'weight': 0.3,
            'description': 'Michael acceptor (electrophilic)'
        },
        
        # Heavy metals and metalloids
        'heavy_metals': {
            'patterns': ['hg', 'pb', 'cd', 'as', 'tl', 'sb'],
            'weight': 0.6,
            'description': 'Heavy metal (highly toxic)'
        },
        'transition_metals': {
            'patterns': ['cr', 'ni', 'co', 'mn', 'cu'],
            'weight': 0.2,
            'description': 'Transition metal'
        },
        
        # Halogenated compounds
        'polyhalogenated': {
            'patterns': ['cfc', 'ccl2', 'cbr2', 'cf2', 'ccl3', 'cf3'],
            'weight': 0.3,
            'description': 'Polyhalogenated compound'
        },
        'halogenated_aromatic': {
            'patterns': ['cf', 'ccl', 'cbr', 'ci'],
            'weight': 0.2,
            'description': 'Halogenated aromatic'
        },
        
        # Peroxides and reactive oxygen
        'peroxide': {
            'patterns': ['coo', 'c-o-o', 'cooc'],
            'weight': 0.4,
            'description': 'Peroxide (explosive/oxidizing)'
        },
        'hydroperoxide': {
            'patterns': ['cooh', 'c-ooh'],
            'weight': 0.3,
            'description': 'Hydroperoxide (unstable)'
        },
        
        # Sulfur-containing toxic groups
        'sulfonyl_halide': {
            'patterns': ['s(=o)(=o)f', 's(=o)(=o)cl'],
            'weight': 0.4,
            'description': 'Sulfonyl halide (reactive)'
        },
        'sulfonate_ester': {
            'patterns': ['cos(=o)(=o)', 'cs(=o)(=o)o'],
            'weight': 0.2,
            'description': 'Sulfonate ester (alkylating)'
        },
        'thiol': {
            'patterns': ['cs', 'c-s', 'csh'],
            'weight': 0.1,
            'description': 'Thiol group (odorous, reactive)'
        },
        
        # Phosphorus compounds
        'organophosphate': {
            'patterns': ['p(=o)(o)(o)o', 'cop(=o)', 'p(=o)'],
            'weight': 0.4,
            'description': 'Organophosphate (neurotoxic)'
        },
        'phosphonate': {
            'patterns': ['cp(=o)', 'p(=o)c'],
            'weight': 0.3,
            'description': 'Phosphonate compound'
        },
        
        # Aromatic heterocycles (some problematic)
        'furan': {
            'patterns': ['c1ccoc1', 'c1occc1'],
            'weight': 0.2,
            'description': 'Furan ring (potential hepatotoxin)'
        },
        'thiophene_substituted': {
            'patterns': ['c1ccsc1c', 'c1sccc1c'],
            'weight': 0.1,
            'description': 'Substituted thiophene'
        },
        
        # Nitriles and cyanides
        'nitrile': {
            'patterns': ['c#n', 'cc#n'],
            'weight': 0.2,
            'description': 'Nitrile group (can release cyanide)'
        },
        'cyanide': {
            'patterns': ['[c-]#n+', 'c#n'],
            'weight': 0.4,
            'description': 'Cyanide (highly toxic)'
        },
        
        # Lactones and lactams (some toxic)
        'beta_lactam': {
            'patterns': ['c1ccn1c=o', 'n1cccc1=o'],
            'weight': 0.2,
            'description': 'Beta-lactam (can be allergenic)'
        },
        
        # Alkylating agents
        'mustard_like': {
            'patterns': ['clccnccl', 'brccnccbr'],
            'weight': 0.6,
            'description': 'Mustard-like alkylating agent'
        },
        'epichlorohydrin_like': {
            'patterns': ['clccc1oc1', 'c1oc1ccl'],
            'weight': 0.4,
            'description': 'Epichlorohydrin-like (carcinogen)'
        },
        
        # Aromatic polycyclics (PAH-like)
        'polycyclic_aromatic': {
            'patterns': ['c1ccc2c(c1)ccc3c2cccc3', 'c1cc2ccc3cccc4ccc(c1)c2c34'],
            'weight': 0.3,
            'description': 'Polycyclic aromatic (potential carcinogen)'
        },
        
        # Reactive carbonyls
        'anhydride': {
            'patterns': ['c(=o)oc(=o)', 'c(=o)oc=o'],
            'weight': 0.3,
            'description': 'Anhydride (reactive, irritant)'
        },
        'acid_chloride': {
            'patterns': ['c(=o)cl', 'cc(=o)cl'],
            'weight': 0.4,
            'description': 'Acid chloride (highly reactive)'
        },
        
        # Strained rings (reactive)
        'cyclopropane': {
            'patterns': ['c1cc1', 'ccc'],
            'weight': 0.1,
            'description': 'Cyclopropane (strained, reactive)'
        },
        'oxirane': {
            'patterns': ['c1oc1', 'coc'],
            'weight': 0.3,
            'description': 'Oxirane/epoxide (alkylating)'
        },
        
        # Miscellaneous toxic patterns
        'hydrazine': {
            'patterns': ['nn', 'n-n', 'nnh'],
            'weight': 0.3,
            'description': 'Hydrazine derivative (carcinogen)'
        },
        'hydroxylamine': {
            'patterns': ['no', 'n-o', 'noh'],
            'weight': 0.2,
            'description': 'Hydroxylamine (mutagenic)'
        },
        'nitrite_ester': {
            'patterns': ['con=o', 'cono'],
            'weight': 0.3,
            'description': 'Nitrite ester (vasodilator, toxic)'
        },
        'carbamate': {
            'patterns': ['nc(=o)o', 'oc(=o)n'],
            'weight': 0.1,
            'description': 'Carbamate (can be cholinesterase inhibitor)'
        }
    }
    
    # Check all patterns
    for pattern_name, pattern_info in toxic_patterns.items():
        patterns = pattern_info['patterns']
        weight = pattern_info['weight']
        
        for pattern in patterns:
            if pattern in smiles_lower:
                risk_score += weight
                break
    
    return min(risk_score, 1.0)

class DrugGenEnv(gym.Env):
    def __init__(self, gctx_savefile, autoencoder_savefile, condition_savefile, condition_dirs, max_selfies_len=50):
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

        condition_checkpoint = torch.load(condition_savefile)
        conditions = condition_checkpoint["conditions"]
        self.condition_model = ConditionModel(len(self.genes), len(conditions), 256)
        self.condition_model.load_state_dict(condition_checkpoint["model_state_dict"])
        self.condition_model.eval()

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
        if not validate_molecule(smiles):
            return self.current_obs_expr, 0, True, False, {}
        
        qed_score = get_qed_score(smiles)
        sa_score = get_sa_score(smiles)
        risk_score = rapid_toxicity_screen(smiles)

        with torch.no_grad():
            current_obs_expr_tensor = torch.tensor(self.current_obs_expr, dtype=torch.float32).unsqueeze(0)
            selfies_embedding_tensor = torch.tensor(selfies_embedding, dtype=torch.float32).unsqueeze(0)
            dosage_conc_tensor = torch.tensor(dosage_conc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            dosage_time_tensor = torch.tensor(dosage_time, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            new_expr = self.gctx_model(current_obs_expr_tensor, selfies_embedding_tensor, dosage_conc_tensor, dosage_time_tensor)

        original_health = (1 - np.mean(self.condition_model(current_obs_expr_tensor)[0].detach().cpu().numpy()))
        new_health = (1 - np.mean(self.condition_model(new_expr)[0].detach().cpu().numpy()))

        healthiness = new_health - original_health
        unnorm_dosage_conc = (np.e ** dosage_conc) - 1
        unnorm_dosage_time = (np.e ** dosage_time) - 1
        reward = max(0, healthiness) * sigmoid(10 * (qed_score - 0.4)) * sigmoid(10 * (0.5 - sa_score / 10)) * sigmoid(10 * (0.4 - risk_score))
        self.reward_list.append(reward)

        if reward > self.max_reward:
            print(f"File: {self.current_obs_file}")
            print(f"SMILES: {smiles}")
            print(f"Dosage Concentration: {unnorm_dosage_conc} uM")
            print(f"Dosage Time: {unnorm_dosage_time} h")
            print(f"Healthiness Improvement: {healthiness}")
            print(f"Drug QED Score: {qed_score}")
            print(f"Drug SA Score: {sa_score}")
            print(f"Drug Risk Score: {risk_score}")
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
    condition_savefile = "Models/condition_model.pth"

    condition_dirs = ["Conditions/Crohns_Disease"]
    policy_savefile = "Models/drug_generator"

    env = DrugGenEnv(gctx_savefile, ae_savefile, condition_savefile, condition_dirs)
    policy_kwargs = dict(
        net_arch=[1200, 1200],
        activation_fn=torch.nn.GELU
    )

    model = PPO("MlpPolicy", env, n_steps=512, batch_size=128, n_epochs=5, learning_rate=1e-5, ent_coef=1e-3, policy_kwargs=policy_kwargs)
    if os.path.exists(f"{policy_savefile}.zip"):
        model.set_parameters(policy_savefile)

    for epoch in range(100):
        model.learn(total_timesteps=5000, progress_bar=True)
        model.save(policy_savefile)

    plt.ioff()
    plt.close()

if __name__=="__main__":
    main()