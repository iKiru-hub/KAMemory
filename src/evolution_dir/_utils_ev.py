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


MAX_VAL = 1e4
MIN_VAL = 1


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
    # info = info["network_params"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(bias=True)

    return {"dim_ei": info["dim_ei"], "dim_ca1": info["dim_ca1"],
            "dim_eo": info["dim_eo"], "K_out": info["K"], "K_lat": info["K_lat"],
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


def _configure_genome(genome: list, genome_configs: dict):

    result = {}
    idx = 0
    for i, (param, conf) in enumerate(genome_configs.items()):
        if conf["active"]:
            result[param] = np.clip(genome[idx], conf["min"], conf["max"])
            idx += 1
        else:
            result[param] = conf["init"]

    return result


def evaluate_genome(genome: list, datasets: list, settings: dict):

    """
    evaluation of a genome

    Parameters
    ----------
    genome: list
        parameters for the model to evolve : [0.2, -0.4, 0.4]
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
    genome_configs = settings["genome_configs"]
    W_ei_ca1 = settings["W_ei_ca1"]
    W_ca1_eo = settings["W_ca1_eo"]
    B_ei_ca1 = settings["B_ei_ca1"]
    B_ca1_eo = settings["B_ca1_eo"]
    K_out = settings["K_out"]
    K_lat = settings["K_lat"]
    num_reps = len(datasets)

    accuracy = np.zeros((num_reps, num_samples, num_samples))

    # -- run | repetition loop
    for l in tqdm(range(num_reps), disable=True):

        # run | main loop
        for i in tqdm(range(num_samples), disable=True):

            # make model
            # model = models.MTL(W_ei_ca1=W_ei_ca1,
            #                    W_ca1_eo=W_ca1_eo,
            #                    B_ei_ca1=B_ei_ca1,
            #                    B_ca1_eo=B_ca1_eo,
            #                    dim_ca3=settings["dim_ei"],
            #                    K_lat=int(np.clip(10*genome[0], MIN_VAL, MAX_VAL)),
            #                    K_ca3=int(np.clip(10*genome[1], MIN_VAL, MAX_VAL)),
            #                    K_out=int(np.clip(10*genome[2], MIN_VAL, MAX_VAL)),
            #                    beta=abs(float(np.clip(100*genome[3],
            #                                           MIN_VAL, MAX_VAL))),
            #                    alpha=float(np.clip(genome[4], 0.01, 0.99)))

            # make model
            params = _configure_genome(genome=genome, genome_configs=genome_configs)
            model = models.MTLev(W_ei_ca1=W_ei_ca1,
                               W_ca1_eo=W_ca1_eo,
                               K_lat=params["K_lat"],
                               K_out=params["K_out"],
                               K_ca3=params["K_ca3"],
                               dim_ca3=genome_configs["K_ca3"]["max"],
                               beta_eo=params["beta_eo"],
                               beta_is=params["beta_is"],
                               beta_ca1=params["beta_ca1"],
                               beta_ca3=params["beta_ca3"],
                               alpha=params["alpha"],
                               num_swaps_ca1=params["num_swaps_ca1"],
                               num_swaps_ca3=params["num_swaps_ca3"],
                               B_ei_ca1=B_ei_ca1,
                               B_ca1_eo=B_ca1_eo)

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

    score = exp_eval(data=result, sigma=settings["sigma"]).mean()
    return score


def save_genome(info: dict, name: str):

    """ save an evolved genome """

    with open(f"logs/{name}.json", "w") as f:
        json.dump(info, f)


def load_genome(index: int=-2):

    """ load an evolved and saved genome """

    name = ""
    path = os.path.abspath(__file__).split("KAMemory")[0] + \
        "KAMemory/src/evolution_dir/logs"
    if index == -2:
        print("available:")
        for k, u in enumerate(os.listdir(path)): print(f"{k}:{u}")
        index = int(input(f"index: "))
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
    file = load_genome(-2)
    print(f"loaded: {file}")
