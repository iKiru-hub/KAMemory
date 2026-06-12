import wandb
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve1d
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys, os

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src/evolution_dir")
import models
import utils
from logger import logger
from _utils_ev import *
from evolution_search import *


"""
This is a sweep search (Weights and Biases).

--- N.B. ---
K_lat and K_ca3 assume the session has an input size
of 50 (or greater than 40)
"""



""" general settings """

THRESHOLD = 0.8
SESSION_IDX = 0
NUM_SAMPLES = 96
NUM_REPS = 3
K_STIMULUS = 5


""" sweep settings """

# Define sweep config
sweep_configuration = {
    "method": "bayes",

    "name": "param_search",

    "metric": {
        "goal": "maximize",
        "name": "accuracy"
    },

"parameters": {
        "K_lat": {"distribution": "int_uniform",
                  "min": 1,
                  "max": 100},
        "K_ca3": {"distribution": "int_uniform",
                  "min": 1,
                  "max": 100},
        "beta_eo": {"distribution": "uniform",
                   "min": 0.,
                   "max": 400.},
        "beta_is": {"distribution": "uniform",
                   "min": 0.,
                   "max": 400.},
        "beta_ca1": {"distribution": "uniform",
                    "min": 0.,
                    "max": 400.},
        "beta_ca3": {"distribution": "uniform",
                    "min": 0.,
                    "max": 400.},
        "alpha": {"distribution": "uniform",
                  "min": 0.,
                  "max": 1.},
    }
}


""" data and model settings """


def run_model(num_reps: int, num_samples: int, parameters: dict,
              settings: dict):

    # # baseline parameters
    # parameters = {"K_lat": 15,
    #               "K_ca3": 6.,
    #               "K_out": settings["K_out"],
    #               "beta_ca3": 14.,
    #               "beta_ca1": 14.,
    #               "beta_is": 10.,
    #               "beta_eo": 7.,
    #               "alpha": 0.1,
    #               "num_swaps_ca1": 1.0,
    #               "num_swaps_ca3": 1.0}

    K_lat = parameters['K_lat']
    K_ca3 = parameters['K_ca3']
    K_out = settings["K_out"]
    beta_eo = parameters['beta_eo']
    beta_is = parameters['beta_is']
    beta_ca1 = parameters['beta_ca1']
    beta_ca3 = parameters['beta_ca3']
    alpha = parameters['alpha']
    num_swaps_ca1 = parameters["num_swaps_ca1"]
    num_swaps_ca3 = parameters["num_swaps_ca3"]

    K_stim = K_STIMULUS
    dim_ei = settings["dim_ei"]
    W_ei_ca1 = settings["W_ei_ca1"]
    W_ca1_eo = settings["W_ca1_eo"]
    B_ei_ca1 = settings["B_ei_ca1"]
    B_ca1_eo = settings["B_ca1_eo"]

    accuracy = np.zeros((num_reps, num_samples, num_samples))

    for l in range(num_reps):

        stimuli = utils.sparse_stimulus_generator(N=num_samples,
                                                  K=K_stim,
                                                  size=dim_ei,
                                                  plot=False)

        datasets = []
        for k in range(num_samples):
            data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
            dataloader = DataLoader(TensorDataset(data),
                                    batch_size=1,
                                    shuffle=False)
            datasets.append(dataloader)

        for i in range(num_samples):

            model = models.MTLev(W_ei_ca1=W_ei_ca1,
                   W_ca1_eo=W_ca1_eo,
                   K_lat=K_lat,
                   K_out=K_out,
                   K_ca3=K_ca3,
                   dim_ca3=dim_ei,
                   beta_eo=beta_eo,
                   beta_is=beta_is,
                   beta_ca1=beta_ca1,
                   beta_ca3=beta_ca3,
                   alpha=alpha,
                   num_swaps_ca1=num_swaps_ca1,
                   num_swaps_ca3=num_swaps_ca3,
                   B_ei_ca1=B_ei_ca1,
                   B_ca1_eo=B_ca1_eo)

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    # forward
                    _ = model(batch[0].reshape(-1, 1))

            # test a dataset with pattern index 0.. i
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    x = batch[0].reshape(-1, 1)

                    # forward
                    y = model(x)
                    # if torch.norm(x).item() == 0.0:
                    #     logger.error(ValueError("zero norm for x"))
                    # if torch.norm(y).item() == 0.0:
                    #     logger.error(ValueError("zero norm for x"))
                    # if torch.isnan(x).any():
                    #     logger.error(ValueError("nan in x"))
                    # if torch.isnan(y).any():
                    #     sys.exit()
                    accuracy[l, i, j] = ((y.T @ x) / (torch.norm(x) * torch.norm(y))).item()

    score = np.mean(exp_eval(data=accuracy.mean(axis=0), sigma=SIGMA)).item()
    return score


""" training """

def main():

    # logger("<<< ---------------------- >>>")

    run = wandb.init()

    settings = load_autoencoder(index=SESSION_IDX)
    settings["num_samples"] = NUM_SAMPLES
    settings["sigma"] = SIGMA
    settings["genome_configs"] = GENOME_CONFIGS

    parameters = {
        "K_lat": wandb.config.K_lat,
        "K_ca3": wandb.config.K_ca3,
        "K_out": settings["K_out"],
        "beta_eo": wandb.config.beta_eo,
        "beta_is": wandb.config.beta_is,
        "beta_ca1": wandb.config.beta_ca1,
        "beta_ca3": wandb.config.beta_ca3,
        "alpha": wandb.config.alpha,
        "num_swaps_ca1": 1.0,
        "num_swaps_ca3": 1.0
    }

    accuracy = run_model(NUM_REPS, NUM_SAMPLES, parameters, settings)
    wandb.log({"accuracy": accuracy})




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="param search for MTL model, new")
    parser.add_argument('--count', type=int,
                        help='number of iterations',
                        default=10)
    parser.add_argument('--sweep-id', type=str, required=True,
                        help='wandb sweep id (e.g. username/project/sweep-id)')
    args = parser.parse_args()

    logger.info(f"%sweep id: {args.sweep_id}")
    logger.info(f"%count: {args.count}")

    wandb.agent(args.sweep_id,
                function=main,
                count=args.count)



