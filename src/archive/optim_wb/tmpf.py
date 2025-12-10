import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve1d
import torch
from torch.utils.data import DataLoader, TensorDataset

import models as models
import utils as utils
import main as main

import argparse

import sys

logger = utils.setup_logger(__name__)

"""
This is a sweep search (Weights and Biases).

--- N.B. ---
K_lat and K_ca3 assume the session has an input size
of 50 (or greater than 40)
"""



""" general settings """

NUM_VAR = 10
THRESHOLD = 0.8
SESSION_IDX = 0


""" sweep settings """

# Define sweep config
sweep_configuration = {
    "method": "bayes",

    "name": "param_search",

    "metric": {
        "goal": "maximize",
        "name": "capacity"
    },

    "parameters": {
        "alpha": {"distribution": "uniform",
                  "min": 0.01,
                  "max": 1.},
        "K_lat": {"distribution": "int_uniform",
                  "min": 1,
                  "max": 40},
        "K_ca3": {"distribution": "int_uniform",
                  "min": 1,
                  "max": 40},
        "beta": {"distribution": "int_uniform",
                 "min": 1,
                 "max": 100}
    }
}


""" data and model settings """


def make_data(num_samples: int,
              dim_ei: int,
              K: int) -> list:

    """
    Make datasets for training and testing.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    dim_ei : int
        Dimension of the input layer.
    K : int
        Number of output neurons.

    Returns
    -------
    datasets : list
        List of datasets.
    """

    datasets = []
    for i in range(num_samples):
        stimuli = utils.sparse_stimulus_generator(N=i+1,
                                                  K=K,
                                                  size=dim_ei,
                                                  plot=False)
        data = torch.tensor(stimuli, dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(data),
                                batch_size=1,
                                shuffle=False)
        datasets += [dataloader]

    return datasets



def train_model(model_params: dict,
                datasets: int,
                num_samples: int,
                random_lvl: float):

    """
    Train a model on a dataset.

    Parameters
    ----------
    model : dict
        Model parameters.
    dataset: torch.utils.data.DataLoader
        Dataset to train on.
    num_samples : int
        Number of samples.
    random_lvl : float
        Random level.

    Returns
    -------
    model : torch.nn.Module
        Trained model.
    """

    outputs = np.zeros((num_samples, num_samples))
    for i in range(num_samples):

        # make model
        model = models.MTL(**model_params)

        # train a dataset with pattern index 0.. i
        model.eval()
        with torch.no_grad():

            # one pattern at a time
            for batch in datasets[i]:
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

                value = (y.T @ x) / \
                            (torch.norm(x) * torch.norm(y))

                # outputs[i, j] = (value.item() - random_lvl) /\
                #     (1 - random_lvl)
                outputs[i, j] = (value.item() - 0.2) / 0.8

    return outputs


""" training """

def main():

    logger("<<< ---------------------- >>>")

    # load session
    info, autoencoder = models.load_session(idx=SESSION_IDX,
                                            verbose=False)

    logger(f"{info=}")
    logger(f"{autoencoder=}")

    # get session parameters
    dim_ei = info["dim_ei"]
    dim_ca3 = info["dim_ca3"]
    dim_ca1 = info["dim_ca1"]
    dim_eo = info["dim_eo"]
    K = info["K"]
    num_samples = 100


    # get parameters: w1, w2, b1, b2
    ae_params = autoencoder.get_weights(bias=True)

    logger(f"{type(ae_params[0])}")

    # make datasets
    datasets = make_data(num_samples=num_samples,
                         dim_ei=dim_ei,
                         K=K)

    logger(f"[2] len: {len(datasets)}, size: {datasets[0].dataset.tensors[0].shape}")

    logger("<<< Data generated >>>")

    RANDOM_LVL = 1 / K
    logger(f"{RANDOM_LVL=}")

    #
    model_params = {
        "W_ei_ca1": ae_params[0],
        "W_ca1_eo": ae_params[1],
        "B_ei_ca1": ae_params[2],
        "B_ca1_eo": ae_params[3],
        "dim_ca3": dim_ca3,
        "lr": 1.,
        "K_lat": 15,
        "K_ca3": 10, 
        "K_out": K,
        "beta": 60., 
        "alpha": 0.22
    }

    # run
    outputs = train_model(model_params=model_params,
                          datasets=datasets,
                          num_samples=num_samples,
                          random_lvl=RANDOM_LVL)
    capacity = utils.calc_capacity(outputs=outputs,
                                   threshold=THRESHOLD,
                                   nsmooth=20,
                                   idx_pattern=None)

    logger(f"capacity: {np.mean(capacity):.1f}")



if __name__ == "__main__":

    main()
