import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch.nn as nn
from drug_generator import DrugGenEnv
from stable_baselines3 import PPO, A2C

def optimize_drug_gen(trial: optuna.trial.Trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_int('n_steps', 1, 256)
    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    ent_coef = trial.suggest_float('ent_coef', 1e-6, 0.1, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)
    arch_width = trial.suggest_categorical('arch_width', [400, 800, 1200, 1600])
    arch_depth = trial.suggest_categorical('arch_depth', [1, 2, 3])
    activation_fn = trial.suggest_categorical('activation_fn', [nn.ReLU, nn.GELU, nn.Tanh])
    
    gctx_savefile = "Models/gctx.pth"
    ae_savefile = "Models/selfies_autoencoder.pth"
    condition_savefile = "Models/health_model.pth"

    condition_dirs = ["Conditions/Unhealthy"]

    env = DrugGenEnv(gctx_savefile, ae_savefile, condition_savefile, condition_dirs)
    policy_kwargs = dict(
        net_arch=[arch_width] * arch_depth,
        activation_fn=activation_fn
    )

    model = A2C(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs
    )

    try:
        model.learn(5000)
    except:
        plt.close(env.fig)
        return 0.0
    slope = float(np.polyfit(list(range(len(env.reward_list))), env.reward_list, deg=1)[0])
    plt.close(env.fig)
    return slope

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_drug_gen, n_trials=50)
    print("Best trial:", study.best_trial.params)

if __name__=="__main__":
    main()