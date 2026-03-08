import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys, os
import warnings
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")
import models
import utils


""" data """

def make_datasets(num_samples: int=200, num_reps: int=2, dim_ei: int=50, K: int=10):

    """ create a dataset """

    datasets_rep = []

    for _ in range(num_reps):
        stimuli = utils.sparse_stimulus_generator(N=num_samples,
                                                  K=K,
                                                  size=dim_ei,
                                                  plot=False)

        datasets = []
        for k in range(num_samples):
            data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
            dataloader = DataLoader(TensorDataset(data),
                                    batch_size=1,
                                    shuffle=False)
            datasets += [dataloader]

        datasets_rep += [datasets]

    return datasets_rep


def load_autoencoder(index: int=0) -> dict:

    """ load the session of an autoencoder with its metadata """

    info, autoencoder = models.load_session(idx=index, verbose=False)

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(bias=True)

    return {"dim_ei": info["dim_ei"],
            "dim_ca1": info["dim_ca1"], "dim_eo": info["dim_eo"],
            "K": info["K"],
            "W_ei_ca1": W_ei_ca1, "W_ca1_eo": W_ca1_eo,
            "B_ei_ca1": B_ei_ca1, "B_ca1_eo": B_ca1_eo}


def exp_eval(data: np.ndarray, sigma: float):

    """
    evaluation of the results as weighted average
    with explonential weights

    Parameters
    ----------
    data: np.ndarray
        shape (num_stimuli, num_stimuli)
    sigma: float
        standard deviation of the exponential kernel

    Return
    ------
    float
    """

    n = len(data)
    out = np.zeros(n)
    for r in range(n):
        denom = 0.
        for c in range(r, -1, -1):
            w = np.exp(-((c-r)/sigma)**2)
            out[r] += w * np.clip(data[r, c], 0., 1.)
            denom += w

        out[r] = out[r] / denom

    return out

def evaluate_genome(genome: list, datasets: list, settings: dict):

    """
    evaluation of a genome

    Parameters
    ----------
    genome: list
        parameters for the model to evolve
    datasets: list
        list of Dataloader objects
    settings: dict
        autoencoder weights and run parameters

    Return
    ------
    float: accuracy
    """

    assert "sigma" in tuple(settings.keys()), "no" + \
        " 'sigma' provided in settings"
    num_samples = settings["num_samples"]
    W_ei_ca1 = settings["W_ei_ca1"]
    W_ca1_eo = settings["W_ca1_eo"]
    B_ei_ca1 = settings["B_ei_ca1"]
    B_ca1_eo = settings["B_ca1_eo"]
    num_reps = len(datasets)

    accuracy = np.zeros((num_reps, num_samples, num_samples))

    # -- run | repetition loop
    for l in tqdm(range(num_reps), disable=True):

        # run | main loop
        for i in tqdm(range(num_samples), disable=True):

            # make model
            model = models.MTL(W_ei_ca1=W_ei_ca1,
                               W_ca1_eo=W_ca1_eo,
                               B_ei_ca1=B_ei_ca1,
                               B_ca1_eo=B_ca1_eo,
                               dim_ca3=settings["dim_ei"],
                               K_lat=int(max((1, 10*genome[0]))),
                               K_ca3=int(max((1, 10*genome[1]))),
                               K_out=int(max((1, 10*genome[2]))),
                               beta=abs(100*genome[3]),
                               alpha=np.clip(genome[4], 0.01, 0.99))

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for j, batch in enumerate(datasets[l][i]):
                    # forward
                    _ = model(batch[0].reshape(-1, 1))

            # test a dataset with pattern index 0.. i 
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[l][i]):
                    x = batch[0].reshape(-1, 1)

                    # forward
                    y = model(x)

                    # record : cosine similarity
                    value = (y.T @ x) / (torch.norm(x) * torch.norm(y))
                    accuracy[l, i, j] = (value.item() - 0.2) / 0.8

    result = accuracy.mean(axis=0)
    if np.any(np.isnan(result)):
        return 0.

    return exp_eval(data=result, sigma=settings["sigma"]).mean()

def save_genome(info: dict, name: str):

    """ save an evolved genome """

    with open(f"logs/{name}.json", "w") as f:
        json.dump(info, f)


def load_genome(index: int):

    """ load an evolved and saved genome """

    name = ""
    path = os.path.abspath(__file__).split("src")[0] + \
        "src/evolution/logs"
    for i, f in enumerate(os.listdir(path)):
        if int(f.split('_')[1]) == index:
            name = f"{path}/{f}"
    if f == "":
        warnings.warn("index not found")
        return None

    with open(name, "r") as f:
        file = json.load(f)

    return file


if __name__ == "__main__":
    file = load_genome(0)
    print(f"loaded: {file}")
